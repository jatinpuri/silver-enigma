import json
import numpy as np
import pandas as pd

ATR_LEN=14; ADX_LEN=14; SL_ATR=2.25; TP_ATR=3.375; RR=1.5; HOLD=12
START=pd.Timestamp('2012-01-01',tz='UTC'); RECENT=pd.Timestamp('2022-01-01',tz='UTC'); Y26=pd.Timestamp('2026-01-01',tz='UTC'); END=pd.Timestamp('2026-08-09',tz='UTC')

def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def indicators(df):
    h,l,c=df.high,df.low,df.close
    pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=rma(tr,ATR_LEN)
    up=h.diff(); dn=-l.diff()
    plus_dm=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=df.index)
    minus_dm=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=df.index)
    base=rma(tr,ADX_LEN)
    plus_di=100*rma(plus_dm,ADX_LEN)/base
    minus_di=100*rma(minus_dm,ADX_LEN)/base
    dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di)
    adx=rma(dx,ADX_LEN)
    return atr,adx,plus_di,minus_di

def pf(a):
    a=np.asarray(a,float); gp=a[a>0].sum(); gl=-a[a<0].sum()
    return gp/gl if gl>0 else np.nan

def maxdd(a):
    a=np.asarray(a,float); eq=np.r_[0,np.cumsum(a)]; peak=np.maximum.accumulate(eq)
    return float(np.max(peak-eq))

arr=json.load(open('eurusd_h1_2011_2026.json'))
df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True)
df['atr'],df['adx'],df['pdi'],df['mdi']=indicators(df)
df['cdir']=np.where(df.close>df.open,1,np.where(df.close<df.open,-1,0))
df['didir']=np.where(df.pdi>df.mdi,1,np.where(df.mdi>df.pdi,-1,0))

local=df.dt.dt.tz_convert('Europe/London')
idxs=np.where((local.dt.hour.to_numpy()==7)&(local.dt.weekday.to_numpy()<=3)&(local.dt.year.to_numpy()>=2012)&(local.dt.year.to_numpy()<=2026))[0]

# selector returns trade direction (+1/-1) or 0 to skip

def getdir(name,i):
    if i<2: return 0
    r=df.iloc[i]; prev=df.iloc[i-1]; prev2=df.iloc[i-2]
    c=int(r.cdir); di=int(r.didir)
    if c==0 or di==0 or not np.isfinite(r.adx) or not np.isfinite(prev.adx): return 0
    rise = r.adx > prev.adx
    fall = r.adx < prev.adx
    if name=='RISE_CANDLE': return c if rise else 0
    if name=='FALL_CANDLE_SAME': return c if fall else 0
    if name=='FALL_CANDLE_REVERSE': return -c if fall else 0
    if name=='FALL_DI': return di if fall else 0
    if name=='FALL_OPP_DI': return -di if fall else 0
    if name=='FALL_AGREE_SAME': return c if fall and c==di else 0
    if name=='FALL_AGREE_REVERSE': return -c if fall and c==di else 0
    if name=='FALL_DISAGREE_CANDLE': return c if fall and c!=di else 0
    if name=='FALL_DISAGREE_DI': return di if fall and c!=di else 0
    if name=='FALL2_CANDLE_SAME': return c if fall and prev.adx < prev2.adx else 0
    if name=='FALL2_CANDLE_REVERSE': return -c if fall and prev.adx < prev2.adx else 0
    for th in [1,2,3,5]:
        if name==f'FALL{th}PT_CANDLE_SAME': return c if (prev.adx-r.adx)>=th else 0
        if name==f'FALL{th}PT_CANDLE_REVERSE': return -c if (prev.adx-r.adx)>=th else 0
    return 0

variants=['RISE_CANDLE','FALL_CANDLE_SAME','FALL_CANDLE_REVERSE','FALL_DI','FALL_OPP_DI',
          'FALL_AGREE_SAME','FALL_AGREE_REVERSE','FALL_DISAGREE_CANDLE','FALL_DISAGREE_DI',
          'FALL2_CANDLE_SAME','FALL2_CANDLE_REVERSE']
for th in [1,2,3,5]:
    variants += [f'FALL{th}PT_CANDLE_SAME',f'FALL{th}PT_CANDLE_REVERSE']

alltr=[]
for name in variants:
    for i in idxs:
        d=getdir(name,i)
        if d==0: continue
        row=df.iloc[i]
        if not np.isfinite(row.atr) or row.atr<=0: continue
        j=i+1
        if j>=len(df) or (df.loc[j,'dt']-row['dt'])>pd.Timedelta(hours=1,minutes=1): continue
        entry=float(row.high if d>0 else row.low)
        # pending order is valid only during next H1 bar
        if d>0 and float(df.loc[j,'high'])<entry: continue
        if d<0 and float(df.loc[j,'low'])>entry: continue
        stop_dist=SL_ATR*float(row.atr); stop_pips=stop_dist/0.0001
        stop=entry-d*stop_dist; target=entry+d*TP_ATR*float(row.atr)
        end=min(j+HOLD-1,len(df)-1); rv=None; exit_idx=end
        for k in range(j,end+1):
            hi=float(df.loc[k,'high']); lo=float(df.loc[k,'low'])
            hit_sl=(lo<=stop) if d>0 else (hi>=stop)
            hit_tp=(hi>=target) if d>0 else (lo<=target)
            if hit_sl:
                rv=-1.0; exit_idx=k; break
            if hit_tp:
                rv=RR; exit_idx=k; break
        if rv is None:
            rv=((float(df.loc[end,'close'])-entry)*d)/stop_dist
        # $4/lot round trip => R cost = 0.4 / stop_pips.
        # Extra 0.2 pip execution stress => another 0.2 / stop_pips R.
        comm_r=0.4/stop_pips
        stress_r=0.6/stop_pips
        alltr.append(dict(variant=name,signal_dt=row['dt'],exit_dt=df.loc[exit_idx,'dt'],R=float(rv),R_after_comm=float(rv-comm_r),R_after_comm_0p2=float(rv-stress_r),stop_pips=stop_pips,dir=d,cdir=int(row.cdir),didir=int(row.didir),adx=float(row.adx),adx_prev=float(df.iloc[i-1].adx)))

tr=pd.DataFrame(alltr)
tr=tr[(tr.signal_dt>=START)&(tr.signal_dt<END)].copy()
tr.to_csv('setup_b_adx_falling_variant_trades.csv',index=False)

rows=[]
periods=[('FULL_2012_2026',START),('RECENT_2022_2026',RECENT),('YTD_2026',Y26)]
for name in variants:
    z0=tr[tr.variant==name].sort_values('signal_dt')
    for plabel,pstart in periods:
        z=z0[z0.signal_dt>=pstart]
        if not len(z): continue
        yearly=z.groupby(z.signal_dt.dt.year).R.sum()
        rows.append(dict(
            variant=name,period=plabel,trades=len(z),win_rate=float((z.R>0).mean()),PF=float(pf(z.R)),EV_R=float(z.R.mean()),
            EV_after_comm_R=float(z.R_after_comm.mean()),EV_after_comm_0p2_R=float(z.R_after_comm_0p2.mean()),total_R=float(z.R.sum()),
            max_DD_R=maxdd(z.R.to_numpy()),avg_stop_pips=float(z.stop_pips.mean()),positive_years=int((yearly>0).sum()),years=int(len(yearly))
        ))
out=pd.DataFrame(rows)
out.to_csv('setup_b_adx_falling_variants_results.csv',index=False)
print(out.to_string(index=False))
