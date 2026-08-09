import json
import numpy as np
import pandas as pd

ATR_LEN=14
SL_ATR=2.25
RR=0.25
RISK_USD=2700.0
COMMISSION_PER_LOT_RT=4.0
END=pd.Timestamp('2026-08-09',tz='UTC')
START=pd.Timestamp('2012-01-01',tz='UTC')
Y26=pd.Timestamp('2026-01-01',tz='UTC')
WINDOWS={
 'LON_OPEN':('Europe/London',7),
 'LON_H1':('Europe/London',8),
 'NY_H1':('America/New_York',8),
}

def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
arr=json.load(open('eurusd_h1_2011_2026.json'))
df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True)
h,l,c=df.high,df.low,df.close
pc=c.shift(1)
tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
df['atr']=rma(tr,ATR_LEN)
df['cdir']=np.where(c>df.open,1,np.where(c<df.open,-1,0))

trades=[]
for w,(tz,sig_hour) in WINDOWS.items():
    local=df.dt.dt.tz_convert(tz)
    lh=local.dt.hour.to_numpy(); wd=local.dt.weekday.to_numpy(); yr=local.dt.year.to_numpy()
    idxs=np.where((lh==sig_hour)&(wd<=4)&(yr>=2012)&(yr<=2026))[0]
    for i in idxs:
        row=df.iloc[i]; d=int(row['cdir'])
        if d==0 or not np.isfinite(row['atr']): continue
        j=i+1
        if j>=len(df) or (df.loc[j,'dt']-row['dt'])>pd.Timedelta(hours=1,minutes=1): continue
        entry=float(row['high'] if d>0 else row['low'])
        if d>0 and float(df.loc[j,'high'])<entry: continue
        if d<0 and float(df.loc[j,'low'])>entry: continue
        risk_price=SL_ATR*float(row['atr'])
        if risk_price<=0: continue
        stop=entry-d*risk_price
        target=entry+d*(risk_price*RR)
        end=min(j+11,len(df)-1)
        rv=None
        for k in range(j,end+1):
            hi=float(df.loc[k,'high']); lo=float(df.loc[k,'low'])
            hit_sl=(lo<=stop) if d>0 else (hi>=stop)
            hit_tp=(hi>=target) if d>0 else (lo<=target)
            if hit_sl: rv=-1.0; break
            if hit_tp: rv=RR; break
        if rv is None:
            rv=((float(df.loc[end,'close'])-entry)*d)/risk_price
        stop_pips=risk_price/0.0001
        lots=RISK_USD/(stop_pips*10.0)
        commission=lots*COMMISSION_PER_LOT_RT
        trades.append(dict(window=w,signal_dt=row['dt'],trade_day=row['dt'].date(),entry_bar_dt=df.loc[j,'dt'],R=float(rv),stop_pips=stop_pips,lots=lots,commission_usd=commission))
tr=pd.DataFrame(trades).sort_values(['trade_day','entry_bar_dt','signal_dt','window']).reset_index(drop=True)

def select(windows,start):
    z=tr[(tr.signal_dt>=start)&(tr.signal_dt<END)&tr.window.isin(windows)].copy()
    out=[]
    for day,g in z.groupby('trade_day',sort=True):
        g=g.sort_values(['entry_bar_dt','signal_dt','window']).head(2)
        out.append(g)
    return pd.concat(out,ignore_index=True) if out else z.iloc[:0].copy()

def row(name,windows,start,label):
    z=select(windows,start)
    gross_ev_r=float(z.R.mean())
    gross_ev_usd=gross_ev_r*RISK_USD
    avg_stop=float(z.stop_pips.mean())
    med_stop=float(z.stop_pips.median())
    avg_lots=float(z.lots.mean())
    avg_comm=float(z.commission_usd.mean())
    comm_r=avg_comm/RISK_USD
    comm_ev_r=gross_ev_r-comm_r
    comm_ev_usd=comm_ev_r*RISK_USD
    base=dict(method=name,period=label,trades=len(z),gross_ev_r=gross_ev_r,gross_ev_usd=gross_ev_usd,avg_stop_pips=avg_stop,median_stop_pips=med_stop,avg_lots=avg_lots,avg_commission_usd=avg_comm,commission_cost_r=comm_r,ev_after_commission_r=comm_ev_r,ev_after_commission_usd=comm_ev_usd)
    for pips in [0.1,0.2,0.3,0.5]:
        # extra all-in price cost in pips beyond commission, proportional to lot size
        extra_usd=(z.lots*10.0*pips).mean()
        evusd=gross_ev_usd-avg_comm-extra_usd
        base[f'extra_{pips:.1f}pip_usd']=extra_usd
        base[f'ev_after_comm_plus_{pips:.1f}pip_usd']=evusd
        base[f'ev_after_comm_plus_{pips:.1f}pip_r']=evusd/RISK_USD
    return base
rows=[]
for start,label in [(START,'2012_2026'),(Y26,'2026_YTD')]:
    rows.append(row('2_WINDOWS_LONDON',['LON_OPEN','LON_H1'],start,label))
    rows.append(row('3_WINDOWS_FIRST2',['LON_OPEN','LON_H1','NY_H1'],start,label))
out=pd.DataFrame(rows)
out.to_csv('real_life_ev_negative_rr_results.csv',index=False)
print(out.to_string(index=False))