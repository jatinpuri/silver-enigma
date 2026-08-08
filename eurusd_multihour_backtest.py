import json, itertools
import numpy as np
import pandas as pd

ATR_LEN=14
ADX_LEN=14
SL_ATR=2.25
TP_ATR=3.375
HOLD=12
RR=TP_ATR/SL_ATR
START_YEAR=2012
END_YEAR=2026
FULL_YEARS=list(range(2012,2026))
DAY_POLICIES={'MonThu':{0,1,2,3},'MonFri':{0,1,2,3,4}}

WINDOWS={
    'LON_OPEN': ('Europe/London',7,'08:00 London decision / 07-08 signal'),
    'LON_H1':   ('Europe/London',8,'09:00 London decision / 08-09 signal'),
    'LON_H2':   ('Europe/London',9,'10:00 London decision / 09-10 signal'),
    'LON_H3':   ('Europe/London',10,'11:00 London decision / 10-11 signal'),
    'NY_OPEN':  ('America/New_York',7,'08:00 New York decision / 07-08 signal'),
    'NY_H1':    ('America/New_York',8,'09:00 New York decision / 08-09 signal'),
    'NY_H2':    ('America/New_York',9,'10:00 New York decision / 09-10 signal'),
}

def pf(rs):
    a=np.asarray(rs,dtype=float)
    gp=a[a>0].sum(); gl=-a[a<0].sum()
    return gp/gl if gl>0 else (np.inf if gp>0 else np.nan)

def maxdd(trades):
    if len(trades)==0: return np.nan
    x=trades.sort_values('exit_dt')['R'].cumsum().to_numpy()
    eq=np.r_[0.0,x]
    peak=np.maximum.accumulate(eq)
    return float(np.max(peak-eq))

def metrics(trades):
    if len(trades)==0:
        return dict(n=0,ev=np.nan,pf=np.nan,wr=np.nan,totalR=0,maxDD=np.nan,annual_freq=0,ytd2026=0,pos_years=0,recent_pf=np.nan,recent_ev=np.nan,old_pf=np.nan,old_ev=np.nan)
    r=trades.R.to_numpy(float)
    counts=trades.groupby('year').size().to_dict()
    annual_freq=float(np.mean([counts.get(y,0) for y in FULL_YEARS]))
    yrR=trades.groupby('year').R.sum().to_dict()
    pos_years=sum(yrR.get(y,0)>0 for y in FULL_YEARS)
    recent=trades[trades.year>=2022]
    old=trades[trades.year<=2015]
    return dict(
        n=len(trades), ev=float(r.mean()), pf=float(pf(r)), wr=float((r>0).mean()), totalR=float(r.sum()),
        maxDD=maxdd(trades), annual_freq=annual_freq, ytd2026=int(counts.get(2026,0)), pos_years=int(pos_years),
        recent_pf=float(pf(recent.R.to_numpy(float))) if len(recent) else np.nan,
        recent_ev=float(recent.R.mean()) if len(recent) else np.nan,
        old_pf=float(pf(old.R.to_numpy(float))) if len(old) else np.nan,
        old_ev=float(old.R.mean()) if len(old) else np.nan,
    )

def rma(s,n):
    return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def indicators(df):
    h,l,c=df.high,df.low,df.close
    pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=rma(tr,ATR_LEN)
    up=h.diff(); down=-l.diff()
    plus=pd.Series(np.where((up>down)&(up>0),up,0.0),index=df.index)
    minus=pd.Series(np.where((down>up)&(down>0),down,0.0),index=df.index)
    atr_adx=rma(tr,ADX_LEN)
    pdi=100*rma(plus,ADX_LEN)/atr_adx.replace(0,np.nan)
    mdi=100*rma(minus,ADX_LEN)/atr_adx.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    adx=rma(dx,ADX_LEN)
    return atr,adx

arr=json.load(open('eurusd_h1_2011_2026.json'))
df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
df=df[(df.high!=df.low) | (df.open!=df.close)].reset_index(drop=True)
df['atr'],df['adx']=indicators(df)
df['adx_prev']=df.adx.shift(1)
df['dir']=np.where(df.close>df.open,1,np.where(df.close<df.open,-1,0))

all_trades=[]
for window,(tz,sig_hour,desc) in WINDOWS.items():
    local=df.dt.dt.tz_convert(tz)
    lh=local.dt.hour.to_numpy(); wd=local.dt.weekday.to_numpy(); yr=local.dt.year.to_numpy()
    for i,row in df.iterrows():
        if yr[i]<START_YEAR or yr[i]>END_YEAR: continue
        if lh[i]!=sig_hour or wd[i]>4: continue
        if row['dir']==0 or not np.isfinite(row.atr) or not np.isfinite(row.adx) or not np.isfinite(row.adx_prev): continue
        if not (row.adx>row.adx_prev): continue
        j=i+1
        if j>=len(df): continue
        if (df.loc[j,'dt']-row['dt']) > pd.Timedelta(hours=1,minutes=1): continue
        d=int(row['dir']); entry=float(row.high if d>0 else row.low)
        if d>0 and float(df.loc[j,'high']) < entry: continue
        if d<0 and float(df.loc[j,'low']) > entry: continue
        risk=SL_ATR*float(row.atr)
        if risk<=0: continue
        stop=entry-d*risk; target=entry+d*TP_ATR*float(row.atr)
        end=min(j+HOLD-1,len(df)-1)
        rv=None; reason='TIME'; exit_idx=end
        for k in range(j,end+1):
            hi=float(df.loc[k,'high']); lo=float(df.loc[k,'low'])
            hit_sl=(lo<=stop) if d>0 else (hi>=stop)
            hit_tp=(hi>=target) if d>0 else (lo<=target)
            if hit_sl:
                rv=-1.0; reason='SL'; exit_idx=k; break
            if hit_tp:
                rv=RR; reason='TP'; exit_idx=k; break
        if rv is None:
            rv=((float(df.loc[end,'close'])-entry)*d)/risk
        all_trades.append(dict(window=window,description=desc,local_weekday=int(wd[i]),signal_dt=row['dt'],entry_bar_dt=df.loc[j,'dt'],exit_dt=df.loc[exit_idx,'dt'],year=int(yr[i]),R=float(rv),reason=reason))

tr=pd.DataFrame(all_trades)
tr.to_csv('eurusd_multihour_trades.csv',index=False)

window_rows=[]; combo_rows=[]; names=list(WINDOWS.keys())
for policy,allowed_days in DAY_POLICIES.items():
    tp=tr[tr.local_weekday.isin(allowed_days)].copy()
    for w in WINDOWS:
        z=tp[tp.window==w].copy(); m=metrics(z)
        m.update(day_policy=policy,window=w,description=WINDOWS[w][2]); window_rows.append(m)
    for k in range(1,len(names)+1):
        for combo in itertools.combinations(names,k):
            z=tp[tp.window.isin(combo)].copy(); m=metrics(z)
            m.update(day_policy=policy,combo='+'.join(combo),windows=k,distance260=abs(m['annual_freq']-260))
            combo_rows.append(m)

hour_df=pd.DataFrame(window_rows)
hour_df=hour_df[['day_policy','window','description','n','annual_freq','ytd2026','ev','pf','wr','totalR','maxDD','pos_years','old_pf','old_ev','recent_pf','recent_ev']]
hour_df.to_csv('eurusd_multihour_windows.csv',index=False)
print('\nINDIVIDUAL WINDOWS')
print(hour_df.to_string(index=False))

combo=pd.DataFrame(combo_rows)
combo['quality']=((combo.ev>0)&(combo.pf>1)&(combo.recent_ev>0)&(combo.recent_pf>1))
combo=combo.sort_values(['distance260','quality','pf','ev'],ascending=[True,False,False,False]).reset_index(drop=True)
combo.to_csv('eurusd_multihour_combos.csv',index=False)
print('\nCLOSEST TO 260/YEAR - BOTH DAY POLICIES')
print(combo.head(30)[['day_policy','combo','windows','annual_freq','distance260','n','ev','pf','wr','totalR','maxDD','pos_years','old_pf','recent_pf','recent_ev','quality']].to_string(index=False))

near=combo[(combo.annual_freq>=220)&(combo.annual_freq<=300)&(combo.ev>0)&(combo.pf>1)&(combo.recent_ev>0)&(combo.recent_pf>1)].copy()
if len(near):
    near['score']=near.ev*100 + (near.pf-1)*10 - near.distance260/20 + near.pos_years/10
    near=near.sort_values('score',ascending=False)
near.head(50).to_csv('eurusd_multihour_best_near260.csv',index=False)
print('\nBEST QUALITY NEAR 260')
if len(near):
    print(near.head(25)[['day_policy','combo','annual_freq','distance260','n','ev','pf','wr','totalR','maxDD','pos_years','old_pf','recent_pf','recent_ev','score']].to_string(index=False))
else:
    print('No candidate met positive full/recent constraints in 220-300 trades/year range.')

# Yearly detail for the 15 best near-target quality candidates.
yrrows=[]
for _,cr in near.head(15).iterrows():
    allowed=DAY_POLICIES[cr['day_policy']]; selected=cr['combo'].split('+')
    z=tr[tr.local_weekday.isin(allowed) & tr.window.isin(selected)]
    for y in range(2012,2027):
        q=z[z.year==y]
        yrrows.append(dict(day_policy=cr['day_policy'],combo=cr['combo'],year=y,trades=len(q),R=float(q.R.sum()) if len(q) else 0,EV=float(q.R.mean()) if len(q) else np.nan,PF=float(pf(q.R.to_numpy(float))) if len(q) else np.nan))
pd.DataFrame(yrrows).to_csv('eurusd_multihour_yearly_top10.csv',index=False)
