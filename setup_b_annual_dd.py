import json
import numpy as np
import pandas as pd

ATR_LEN=14; ADX_LEN=14; SL_ATR=2.25; TP_ATR=3.375; HOLD=12
START=pd.Timestamp('2012-01-01',tz='UTC'); END=pd.Timestamp('2026-08-09',tz='UTC')

def rma(s,n):
    return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def indicators(df):
    h,l,c=df.high,df.low,df.close
    pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=rma(tr,ATR_LEN)
    up=h.diff(); dn=-l.diff()
    plus_dm=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=df.index)
    minus_dm=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=df.index)
    atr_dm=rma(tr,ADX_LEN)
    plus_di=100*rma(plus_dm,ADX_LEN)/atr_dm
    minus_di=100*rma(minus_dm,ADX_LEN)/atr_dm
    dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di)
    adx=rma(dx,ADX_LEN)
    return atr,adx

arr=json.load(open('eurusd_h1_2011_2026.json'))
df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True)
df['atr'],df['adx']=indicators(df)
df['cdir']=np.where(df.close>df.open,1,np.where(df.close<df.open,-1,0))

local=df.dt.dt.tz_convert('Europe/London')
idxs=np.where((local.dt.hour.to_numpy()==7)&(local.dt.weekday.to_numpy()<=3)&(local.dt.year.to_numpy()>=2012)&(local.dt.year.to_numpy()<=2026))[0]
trades=[]
for i in idxs:
    if i<1: continue
    row=df.iloc[i]
    d=int(row.cdir)
    if d==0 or not np.isfinite(row.atr) or not np.isfinite(row.adx) or not np.isfinite(df.iloc[i-1].adx): continue
    if not (row.adx > df.iloc[i-1].adx): continue
    j=i+1
    if j>=len(df) or (df.loc[j,'dt']-row['dt'])>pd.Timedelta(hours=1,minutes=1): continue
    entry=float(row.high if d>0 else row.low)
    if d>0 and float(df.loc[j,'high'])<entry: continue
    if d<0 and float(df.loc[j,'low'])>entry: continue
    risk=SL_ATR*float(row.atr)
    stop=entry-d*risk; target=entry+d*TP_ATR*float(row.atr)
    end=min(j+HOLD-1,len(df)-1)
    rv=None; exit_idx=end
    for k in range(j,end+1):
        hi=float(df.loc[k,'high']); lo=float(df.loc[k,'low'])
        hit_sl=(lo<=stop) if d>0 else (hi>=stop)
        hit_tp=(hi>=target) if d>0 else (lo<=target)
        if hit_sl:
            rv=-1.0; exit_idx=k; break
        if hit_tp:
            rv=1.5; exit_idx=k; break
    if rv is None:
        rv=((float(df.loc[end,'close'])-entry)*d)/risk
    trades.append({'signal_dt':row['dt'],'exit_dt':df.loc[exit_idx,'dt'],'R':float(rv)})

tr=pd.DataFrame(trades).sort_values('exit_dt').reset_index(drop=True)
tr=tr[(tr.signal_dt>=START)&(tr.signal_dt<END)]

def maxdd(a):
    a=np.asarray(a,float)
    eq=np.concatenate([[0.0],np.cumsum(a)])
    peak=np.maximum.accumulate(eq)
    return float(np.max(peak-eq))

rows=[]
for y,g in tr.groupby(tr.signal_dt.dt.year):
    rows.append({'year':int(y),'trades':len(g),'total_R':g.R.sum(),'max_dd_R':maxdd(g.R.to_numpy())})
out=pd.DataFrame(rows)
out.to_csv('setup_b_annual_dd_results.csv',index=False)
completed=out[out.year<=2025]
summary=pd.DataFrame([{
    'completed_years':len(completed),
    'avg_annual_max_dd_R':completed.max_dd_R.mean(),
    'median_annual_max_dd_R':completed.max_dd_R.median(),
    'worst_annual_max_dd_R':completed.max_dd_R.max(),
    'worst_year':int(completed.loc[completed.max_dd_R.idxmax(),'year']),
    'best_annual_max_dd_R':completed.max_dd_R.min(),
    'continuous_max_dd_R':maxdd(tr.R.to_numpy()),
    'total_trades':len(tr),
    'total_R':tr.R.sum(),
    'ytd_2026_max_dd_R':float(out.loc[out.year==2026,'max_dd_R'].iloc[0]) if (out.year==2026).any() else np.nan,
    'ytd_2026_total_R':float(out.loc[out.year==2026,'total_R'].iloc[0]) if (out.year==2026).any() else np.nan,
}])
summary.to_csv('setup_b_annual_dd_summary.csv',index=False)
print(out.to_string(index=False))
print('\nSUMMARY')
print(summary.to_string(index=False))
