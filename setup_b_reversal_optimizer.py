import json, itertools
import numpy as np
import pandas as pd

ATR_LEN=14; ADX_LEN=14; HOLD=12
START=pd.Timestamp('2012-01-01',tz='UTC'); TRAIN_END=pd.Timestamp('2022-01-01',tz='UTC'); END=pd.Timestamp('2026-08-09',tz='UTC')
RRS=[0.20,0.25,0.30,0.40,0.50,0.60,0.75,1.00]
SLS=[0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,3.00]

def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def pf(a):
    a=np.asarray(a,float); gp=a[a>0].sum(); gl=-a[a<0].sum(); return gp/gl if gl>0 else np.nan

def maxdd(a):
    a=np.asarray(a,float); eq=np.r_[0,np.cumsum(a)]; peak=np.maximum.accumulate(eq); return float(np.max(peak-eq))

def indicators(df):
    h,l,c=df.high,df.low,df.close; pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=rma(tr,ATR_LEN)
    up=h.diff(); dn=-l.diff()
    plus_dm=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=df.index)
    minus_dm=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=df.index)
    base=rma(tr,ADX_LEN)
    pdi=100*rma(plus_dm,ADX_LEN)/base; mdi=100*rma(minus_dm,ADX_LEN)/base
    dx=100*(pdi-mdi).abs()/(pdi+mdi); adx=rma(dx,ADX_LEN)
    return atr,adx,pdi,mdi

arr=json.load(open('eurusd_h1_2011_2026.json'))
df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True)
df['atr'],df['adx'],df['pdi'],df['mdi']=indicators(df)
df['cdir']=np.where(df.close>df.open,1,np.where(df.close<df.open,-1,0))
df['didir']=np.where(df.pdi>df.mdi,1,np.where(df.mdi>df.pdi,-1,0))
df['range']=df.high-df.low; df['body']=(df.close-df.open).abs(); df['bodyfrac']=df.body/df.range.replace(0,np.nan)
df['range_ratio']=df['range']/df['range'].shift(1)
df['di_gap']=(df.pdi-df.mdi).abs()
df['adx_drop']=df.adx.shift(1)-df.adx
local=df.dt.dt.tz_convert('Europe/London')
idxs=np.where((local.dt.hour.to_numpy()==7)&(local.dt.weekday.to_numpy()<=3)&(local.dt.year.to_numpy()>=2012)&(local.dt.year.to_numpy()<=2026))[0]

# Candidate filters. Each is interpretable and not too many dimensions.
def pass_filter(name,r):
    c=int(r.cdir); di=int(r.didir)
    if name=='NONE': return True
    if name=='DI_SUPPORT': return c!=di                 # reversal direction equals dominant DI
    if name=='DI_FADE': return c==di                    # reversal fades both candle and DI
    if name=='DI_GAP3_SUPPORT': return c!=di and r.di_gap>=3
    if name=='DI_GAP5_SUPPORT': return c!=di and r.di_gap>=5
    if name=='DI_GAP10_SUPPORT': return c!=di and r.di_gap>=10
    if name=='ADX20_SUPPORT': return c!=di and r.adx>=20
    if name=='ADX25_SUPPORT': return c!=di and r.adx>=25
    if name=='ADX30_SUPPORT': return c!=di and r.adx>=30
    if name=='DROP0P5_SUPPORT': return c!=di and r.adx_drop>=0.5
    if name=='DROP1_SUPPORT': return c!=di and r.adx_drop>=1.0
    if name=='RANGE_EXPAND_SUPPORT': return c!=di and r.range_ratio>=1.0
    if name=='BODY50_SUPPORT': return c!=di and r.bodyfrac>=0.5
    if name=='SMALLBODY40_SUPPORT': return c!=di and r.bodyfrac<=0.4
    if name=='ADX20_GAP5_SUPPORT': return c!=di and r.adx>=20 and r.di_gap>=5
    if name=='ADX20_GAP10_SUPPORT': return c!=di and r.adx>=20 and r.di_gap>=10
    if name=='DROP0P5_GAP5_SUPPORT': return c!=di and r.adx_drop>=0.5 and r.di_gap>=5
    if name=='RANGE_GAP5_SUPPORT': return c!=di and r.range_ratio>=1.0 and r.di_gap>=5
    return False
FILTERS=['NONE','DI_SUPPORT','DI_FADE','DI_GAP3_SUPPORT','DI_GAP5_SUPPORT','DI_GAP10_SUPPORT','ADX20_SUPPORT','ADX25_SUPPORT','ADX30_SUPPORT','DROP0P5_SUPPORT','DROP1_SUPPORT','RANGE_EXPAND_SUPPORT','BODY50_SUPPORT','SMALLBODY40_SUPPORT','ADX20_GAP5_SUPPORT','ADX20_GAP10_SUPPORT','DROP0P5_GAP5_SUPPORT','RANGE_GAP5_SUPPORT']

# Precompute falling-ADX signal candidates and future bars so grid runs quickly.
signals=[]
for i in idxs:
    if i<1: continue
    r=df.iloc[i]; p=df.iloc[i-1]; c=int(r.cdir)
    if c==0 or not np.isfinite(r.atr) or not np.isfinite(r.adx) or not np.isfinite(p.adx): continue
    if not (r.adx < p.adx): continue
    d=-c
    j=i+1
    if j>=len(df) or (df.loc[j,'dt']-r['dt'])>pd.Timedelta(hours=1,minutes=1): continue
    signals.append((i,j,d))

rows=[]; trade_rows=[]
for filt,sl_mult,rr in itertools.product(FILTERS,SLS,RRS):
    vals=[]
    for i,j,d in signals:
        r=df.iloc[i]
        if not pass_filter(filt,r): continue
        entry=float(r.high if d>0 else r.low)
        if d>0 and float(df.loc[j,'high'])<entry: continue
        if d<0 and float(df.loc[j,'low'])>entry: continue
        sd=sl_mult*float(r.atr); stop=entry-d*sd; target=entry+d*(sl_mult*rr)*float(r.atr)
        end=min(j+HOLD-1,len(df)-1); rv=None; ex=end
        for k in range(j,end+1):
            hi=float(df.loc[k,'high']); lo=float(df.loc[k,'low'])
            hit_sl=(lo<=stop) if d>0 else (hi>=stop)
            hit_tp=(hi>=target) if d>0 else (lo<=target)
            if hit_sl: rv=-1.0; ex=k; break
            if hit_tp: rv=rr; ex=k; break
        if rv is None: rv=((float(df.loc[end,'close'])-entry)*d)/sd
        stop_pips=sd/0.0001
        comm_r=0.4/stop_pips
        stress_r=0.6/stop_pips
        vals.append((r['dt'],float(rv),float(rv-comm_r),float(rv-stress_r)))
    if not vals: continue
    z=pd.DataFrame(vals,columns=['dt','R','Rc','Rs'])
    for period,start,end in [('TRAIN_2012_2021',START,TRAIN_END),('OOS_2022_2026',TRAIN_END,END),('YTD_2026',pd.Timestamp('2026-01-01',tz='UTC'),END),('FULL_2012_2026',START,END)]:
        q=z[(z.dt>=start)&(z.dt<end)]
        if len(q)==0: continue
        yrs=q.groupby(q.dt.dt.year).R.sum()
        rows.append(dict(filter=filt,sl_atr=sl_mult,rr=rr,period=period,trades=len(q),wr=float((q.R>0).mean()),pf=float(pf(q.R)),ev=float(q.R.mean()),ev_comm=float(q.Rc.mean()),ev_stress=float(q.Rs.mean()),total=float(q.R.sum()),maxdd=maxdd(q.R),positive_years=int((yrs>0).sum()),years=len(yrs)))

out=pd.DataFrame(rows)
out.to_csv('setup_b_reversal_optimizer_all.csv',index=False)

# Select train-positive candidates with adequate frequency, then rank robustness on OOS.
wide=out.pivot_table(index=['filter','sl_atr','rr'],columns='period',values=['trades','ev','ev_comm','ev_stress','pf','total'],aggfunc='first')
wide.columns=['__'.join(c) for c in wide.columns]; wide=wide.reset_index()
for col in wide.columns:
    if col not in ['filter','sl_atr','rr']: wide[col]=pd.to_numeric(wide[col],errors='coerce')
sel=wide[(wide['trades__TRAIN_2012_2021']>=120)&(wide['ev_stress__TRAIN_2012_2021']>0)&(wide['trades__OOS_2022_2026']>=40)&(wide['ev_stress__OOS_2022_2026']>0)].copy()
if len(sel):
    # robustness score rewards OOS stress EV and trade frequency, penalizes train/OOS mismatch
    sel['score']=sel['ev_stress__OOS_2022_2026']*np.sqrt(sel['trades__OOS_2022_2026']) - 0.5*(sel['ev_stress__TRAIN_2012_2021']-sel['ev_stress__OOS_2022_2026']).abs()
    sel=sel.sort_values('score',ascending=False)
sel.to_csv('setup_b_reversal_optimizer_robust.csv',index=False)
print('ROBUST CANDIDATES')
print(sel.head(30).to_string(index=False) if len(sel) else 'NONE')
