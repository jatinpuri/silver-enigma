import json, math
import numpy as np
import pandas as pd

START=pd.Timestamp('2012-01-01',tz='UTC')
TRAIN_END=pd.Timestamp('2022-01-01',tz='UTC')
END=pd.Timestamp('2026-08-09',tz='UTC')
HOLD=12
RRS=[0.30,0.40,0.60,0.75,1.00,1.50]
SLS=[1.50,2.00,2.25,2.50,3.00]
SLOTS={'LON_OPEN':7,'LON_H1':8,'LON_H2':9}

# Cost model in R at $-normalized sizing: EURUSD $4/lot RT commission + optional 0.2 pip stress.
# If stop is X pips, $4/lot = 0.4/X R; +0.2 pip = another 0.2/X R.
def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def pf(x):
    x=np.asarray(x,float); gp=x[x>0].sum(); gl=-x[x<0].sum()
    return gp/gl if gl>0 else np.nan

def indicators(df):
    h,l,c=df.high,df.low,df.close; pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=rma(tr,14)
    up=h.diff(); dn=-l.diff()
    pdm=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=df.index)
    mdm=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=df.index)
    base=rma(tr,14)
    pdi=100*rma(pdm,14)/base; mdi=100*rma(mdm,14)/base
    dx=100*(pdi-mdi).abs()/(pdi+mdi); adx=rma(dx,14)
    return atr,adx,pdi,mdi

arr=json.load(open('eurusd_h1_2011_2026.json'))
df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True)
df['atr'],df['adx'],df['pdi'],df['mdi']=indicators(df)
df['cdir']=np.sign(df.close-df.open).fillna(0).astype(int)
df['didir']=np.sign(df.pdi-df.mdi).fillna(0).astype(int)
df['gap']=(df.pdi-df.mdi).abs()
df['rng']=df.high-df.low
df['range_ratio']=df.rng/df.rng.shift(1)
df['bodyfrac']=(df.close-df.open).abs()/df.rng.replace(0,np.nan)
df['adx_rise']=df.adx>df.adx.shift(1)
df['adx_fall']=df.adx<df.adx.shift(1)
local=df.dt.dt.tz_convert('Europe/London')
df['ldn_hour']=local.dt.hour
df['ldn_wd']=local.dt.weekday
df['trade_day']=local.dt.date

high=df.high.to_numpy(float); low=df.low.to_numpy(float); close=df.close.to_numpy(float)

# Generate signal rows for the three London decision windows.
sigs=[]
for slot,hour in SLOTS.items():
    idxs=np.where((df.ldn_hour.to_numpy()==hour)&(df.ldn_wd.to_numpy()<=3)&(df.dt.to_numpy()>=START.to_datetime64())&(df.dt.to_numpy()<END.to_datetime64()))[0]
    for i in idxs:
        if i<1: continue
        r=df.iloc[i]
        if int(r.cdir)==0 or int(r.didir)==0 or not np.isfinite(r.atr) or not np.isfinite(r.adx) or not np.isfinite(r.gap): continue
        j=i+1
        if j>=len(df) or (df.loc[j,'dt']-r['dt'])>pd.Timedelta(hours=1,minutes=1): continue
        sigs.append(dict(slot=slot,i=i,j=j,dt=r.dt,day=r.trade_day,c=int(r.cdir),di=int(r.didir),atr=float(r.atr),adx=float(r.adx),gap=float(r.gap),rise=bool(r.adx_rise),fall=bool(r.adx_fall),rrange=float(r.range_ratio),body=float(r.bodyfrac)))
S=pd.DataFrame(sigs)

# Evaluate a given signal direction / SL / RR. Pending is valid only next H1.
def outcome(row,d,sl,rr):
    i=row.i; j=row.j; atr=row.atr
    entry=float(df.loc[i,'high'] if d>0 else df.loc[i,'low'])
    if (d>0 and high[j]<entry) or (d<0 and low[j]>entry): return None
    sd=sl*atr; st=entry-d*sd; tg=entry+d*sd*rr; end=min(j+HOLD-1,len(df)-1)
    rv=None
    for k in range(j,end+1):
        hit_sl=(low[k]<=st) if d>0 else (high[k]>=st)
        hit_tp=(high[k]>=tg) if d>0 else (low[k]<=tg)
        if hit_sl: rv=-1.0; break
        if hit_tp: rv=rr; break
    if rv is None: rv=((close[end]-entry)*d)/sd
    sp=sd/0.0001
    return float(rv), float(rv-0.4/sp), float(rv-0.6/sp)

# Existing fixed system at London open only. It gets first priority each day.
def base_trade(row):
    if row.slot!='LON_OPEN': return None
    if row.rise:
        return outcome(row,row.c,2.25,1.50)
    if row.fall and row.adx>=20 and row.gap>=5 and row.c!=row.di:
        return outcome(row,row.di,2.50,0.60)
    return None

base_rows=[]; base_days=set()
for row in S[S.slot=='LON_OPEN'].itertuples(index=False):
    o=base_trade(row)
    if o is not None:
        base_rows.append((row.dt,row.day,o[0],o[1],o[2],'BASE'))
        base_days.add(row.day)
base=pd.DataFrame(base_rows,columns=['dt','day','R','R_comm','R_stress','setup'])

# Candidate setup families. Direction function and condition deliberately simple.
def candidate_defs(row):
    c,di=row.c,row.di; gap=row.gap; adx=row.adx; rise=row.rise; fall=row.fall; rrng=row.rrange; body=row.body
    out=[]
    # Same original continuation logic, but at later London windows.
    if rise: out.append(('RISE_CANDLE',c))
    # Robust reversal family.
    if fall and adx>=20 and gap>=5 and c!=di: out.append(('FALL_REV_A20_G5',di))
    # Falling ADX but candle still agrees with dominant DI: trend may be decelerating, not reversing.
    if fall and c==di and gap>=5: out.append(('FALL_CONT_G5',di))
    if fall and c==di and gap>=10: out.append(('FALL_CONT_G10',di))
    # DI continuation regardless of ADX slope.
    if c==di and gap>=5 and adx>=20: out.append(('DI_CONT_A20_G5',di))
    if c==di and gap>=10 and adx>=20: out.append(('DI_CONT_A20_G10',di))
    # Range expansion + DI alignment.
    if rrng>=1.0 and c==di and gap>=5: out.append(('RANGE_DI_CONT_G5',di))
    # Strong expanding candle continuation.
    if rrng>=1.0 and body>=0.60: out.append(('EXPAND_BODY60',c))
    # Pullback to dominant DI: candle against DI, trade back with DI.
    if c!=di and gap>=5 and adx>=20: out.append(('PULLBACK_DI_A20_G5',di))
    if c!=di and gap>=10 and adx>=20: out.append(('PULLBACK_DI_A20_G10',di))
    return out

# Build all candidate strategies, but only on days base does not already trigger.
records=[]
for row in S.itertuples(index=False):
    if row.day in base_days: continue
    # LON_OPEN alternatives plus later slots are allowed; later slots are the main frequency fillers.
    for fam,d in candidate_defs(row):
        # Don't duplicate exact base definitions in same slot.
        if row.slot=='LON_OPEN' and fam in ('RISE_CANDLE','FALL_REV_A20_G5'): continue
        for sl in SLS:
            for rr in RRS:
                o=outcome(row,d,sl,rr)
                if o is None: continue
                records.append((row.slot,fam,sl,rr,row.dt,row.day,o[0],o[1],o[2]))
C=pd.DataFrame(records,columns=['slot','family','sl','rr','dt','day','R','R_comm','R_stress'])

def stats(g,a,b):
    q=g[(g.dt>=a)&(g.dt<b)]
    if len(q)==0: return dict(n=0,ev=np.nan,evc=np.nan,evs=np.nan,pf=np.nan,days=0,total=0.0)
    return dict(n=len(q),ev=q.R.mean(),evc=q.R_comm.mean(),evs=q.R_stress.mean(),pf=pf(q.R),days=q.day.nunique(),total=q.R.sum())

rows=[]
for key,g in C.groupby(['slot','family','sl','rr']):
    tr=stats(g,START,TRAIN_END); oo=stats(g,TRAIN_END,END); yy=stats(g,pd.Timestamp('2026-01-01',tz='UTC'),END); fu=stats(g,START,END)
    rows.append((*key,tr['n'],tr['evc'],tr['evs'],oo['n'],oo['evc'],oo['evs'],yy['n'],yy['evc'],yy['evs'],fu['n'],fu['evc'],fu['evs'],fu['pf'],fu['days']))
R=pd.DataFrame(rows,columns=['slot','family','sl','rr','n_train','evc_train','evs_train','n_oos','evc_oos','evs_oos','n_2026','evc_2026','evs_2026','n_full','evc_full','evs_full','pf_full','days_full'])
# Strict: positive after official commission in all three eras. Need enough old/OOS observations.
strict=R[(R.n_train>=60)&(R.n_oos>=20)&(R.n_2026>=3)&(R.evc_train>0)&(R.evc_oos>0)&(R.evc_2026>0)].copy()
strict['robust_score']=strict[['evc_train','evc_oos','evc_2026']].min(axis=1)*np.sqrt(strict.n_full)
strict['freq_score']=strict.n_2026 + 0.03*strict.n_full + 5*strict.robust_score
strict=strict.sort_values(['freq_score','robust_score'],ascending=False)
strict.to_csv('setup_b_extra_setups_strict_candidates.csv',index=False)
R.to_csv('setup_b_extra_setups_all_candidates.csv',index=False)

# Greedy portfolio: base first, then add up to 3 strict candidates maximizing unique no-base days.
# One trade max per day. Candidate priority only applies when earlier choices did not trade that day.
def candidate_trades(spec,blocked):
    slot,fam,sl,rr=spec
    q=C[(C.slot==slot)&(C.family==fam)&(C.sl==sl)&(C.rr==rr)&(~C.day.isin(blocked))].sort_values('dt')
    # Candidate itself only has one row/day per slot; keep earliest in case of odd duplicates.
    return q.drop_duplicates('day',keep='first')

selected=[]; blocked=set(base_days)
for step in range(3):
    best=None
    for r in strict.head(150).itertuples(index=False):
        spec=(r.slot,r.family,float(r.sl),float(r.rr))
        if spec in selected: continue
        q=candidate_trades(spec,blocked)
        # require some actual extra 2026 coverage and positive after-commission on incremental trades overall/OOS
        q26=q[(q.dt>=pd.Timestamp('2026-01-01',tz='UTC'))&(q.dt<END)]
        qoos=q[(q.dt>=TRAIN_END)&(q.dt<END)]
        qtrain=q[(q.dt>=START)&(q.dt<TRAIN_END)]
        if len(q26)<2 or len(qoos)<12 or len(qtrain)<35: continue
        if q.R_comm.mean()<=0 or qoos.R_comm.mean()<=0 or qtrain.R_comm.mean()<=0: continue
        score=20*len(q26)+0.2*len(q)+3*min(q.R_comm.mean(),qoos.R_comm.mean(),qtrain.R_comm.mean())
        if best is None or score>best[0]: best=(score,spec,q)
    if best is None: break
    _,spec,q=best; selected.append(spec); blocked.update(q.day.tolist())

portfolio_parts=[base.assign(priority=0)]
blocked2=set(base_days)
selection_rows=[]
for p,spec in enumerate(selected,1):
    q=candidate_trades(spec,blocked2)
    blocked2.update(q.day.tolist())
    qq=q[['dt','day','R','R_comm','R_stress']].copy(); qq['setup']='|'.join(map(str,spec)); qq['priority']=p
    portfolio_parts.append(qq)
    selection_rows.append(dict(priority=p,slot=spec[0],family=spec[1],sl=spec[2],rr=spec[3],trades_full=len(q),trades_2026=int(((q.dt>=pd.Timestamp('2026-01-01',tz='UTC'))&(q.dt<END)).sum()),ev_comm_full=q.R_comm.mean(),ev_stress_full=q.R_stress.mean()))
P=pd.concat(portfolio_parts,ignore_index=True).sort_values('dt')

def pstats(name,a,b):
    q=P[(P.dt>=a)&(P.dt<b)]
    return dict(period=name,trades=len(q),unique_days=q.day.nunique(),wr=(q.R>0).mean() if len(q) else np.nan,pf=pf(q.R) if len(q) else np.nan,ev=q.R.mean() if len(q) else np.nan,ev_comm=q.R_comm.mean() if len(q) else np.nan,ev_stress=q.R_stress.mean() if len(q) else np.nan,total_R=q.R.sum() if len(q) else 0.0)
PS=pd.DataFrame([pstats('TRAIN_2012_2021',START,TRAIN_END),pstats('OOS_2022_2026',TRAIN_END,END),pstats('YTD_2026',pd.Timestamp('2026-01-01',tz='UTC'),END),pstats('FULL_2012_2026',START,END)])
pd.DataFrame(selection_rows).to_csv('setup_b_extra_setups_selected.csv',index=False)
PS.to_csv('setup_b_extra_setups_portfolio_summary.csv',index=False)
P.to_csv('setup_b_extra_setups_portfolio_trades.csv',index=False)

base_summary=[]
for name,a,b in [('TRAIN_2012_2021',START,TRAIN_END),('OOS_2022_2026',TRAIN_END,END),('YTD_2026',pd.Timestamp('2026-01-01',tz='UTC'),END),('FULL_2012_2026',START,END)]:
    q=base[(base.dt>=a)&(base.dt<b)]
    base_summary.append(dict(period=name,trades=len(q),unique_days=q.day.nunique(),ev=q.R.mean() if len(q) else np.nan,ev_comm=q.R_comm.mean() if len(q) else np.nan,total_R=q.R.sum() if len(q) else 0.0))
pd.DataFrame(base_summary).to_csv('setup_b_extra_setups_base_summary.csv',index=False)

print('BASE')
print(pd.DataFrame(base_summary).to_string(index=False))
print('\nTOP STRICT CANDIDATES')
print(strict.head(30).to_string(index=False) if len(strict) else 'NONE')
print('\nSELECTED')
print(pd.DataFrame(selection_rows).to_string(index=False))
print('\nPORTFOLIO')
print(PS.to_string(index=False))
