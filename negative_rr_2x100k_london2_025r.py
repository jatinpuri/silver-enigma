import json
from collections import defaultdict
import numpy as np
import pandas as pd

ATR_LEN=14
SL_ATR=2.25
RR=0.25
TP_ATR=SL_ATR*RR
HOLD=12
RISK_PCT=0.027
N_ACCTS=2
SIZE=100000.0
SPLIT=0.80
COST=549.0
TRAIL=0.06*SIZE
PROF_DAY_MIN=0.005*SIZE
WAIT_DAYS=14
CONS_MAX=0.20
FULL_START=pd.Timestamp('2012-01-01',tz='UTC')
Y26_START=pd.Timestamp('2026-01-01',tz='UTC')
END=pd.Timestamp('2026-08-09',tz='UTC')
WINDOWS={
    'LON_OPEN':('Europe/London',7),
    'LON_H1':('Europe/London',8),
}

def rma(s,n):
    return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def indicators(df):
    h,l,c=df.high,df.low,df.close
    pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return rma(tr,ATR_LEN)

arr=json.load(open('eurusd_h1_2011_2026.json'))
df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True)
df['atr']=indicators(df)
df['cdir']=np.where(df.close>df.open,1,np.where(df.close<df.open,-1,0))

trades=[]
for w,(tz,sig_hour) in WINDOWS.items():
    local=df.dt.dt.tz_convert(tz)
    lh=local.dt.hour.to_numpy(); wd=local.dt.weekday.to_numpy(); yr=local.dt.year.to_numpy()
    idxs=np.where((lh==sig_hour)&(wd<=4)&(yr>=2012)&(yr<=2026))[0]
    for i in idxs:
        row=df.iloc[i]
        d=int(row['cdir'])
        if d==0 or not np.isfinite(row['atr']):
            continue
        j=i+1
        if j>=len(df) or (df.loc[j,'dt']-row['dt'])>pd.Timedelta(hours=1,minutes=1):
            continue
        entry=float(row['high'] if d>0 else row['low'])
        if d>0 and float(df.loc[j,'high'])<entry:
            continue
        if d<0 and float(df.loc[j,'low'])>entry:
            continue
        risk=SL_ATR*float(row['atr'])
        if risk<=0:
            continue
        stop=entry-d*risk
        target=entry+d*TP_ATR*float(row['atr'])
        end=min(j+HOLD-1,len(df)-1)
        rv=None; exit_idx=end; reason='TIME'
        for k in range(j,end+1):
            hi=float(df.loc[k,'high']); lo=float(df.loc[k,'low'])
            hit_sl=(lo<=stop) if d>0 else (hi>=stop)
            hit_tp=(hi>=target) if d>0 else (lo<=target)
            if hit_sl:
                rv=-1.0; exit_idx=k; reason='SL'; break
            if hit_tp:
                rv=RR; exit_idx=k; reason='TP'; break
        if rv is None:
            rv=((float(df.loc[end,'close'])-entry)*d)/risk
        trades.append(dict(window=w,signal_dt=row['dt'],entry_bar_dt=df.loc[j,'dt'],exit_dt=df.loc[exit_idx,'dt'],trade_day=row['dt'].date(),R=float(rv),reason=reason))

tr=pd.DataFrame(trades).sort_values(['trade_day','entry_bar_dt','signal_dt','window']).reset_index(drop=True)

def pf(a):
    a=np.asarray(a,float)
    gp=a[a>0].sum(); gl=-a[a<0].sum()
    return gp/gl if gl>0 else (np.inf if gp>0 else np.nan)

def market_metrics(z):
    a=z.R.to_numpy(float)
    return dict(signals=len(z),wr=float((a>0).mean()) if len(a) else np.nan,ev=float(a.mean()) if len(a) else np.nan,pf=float(pf(a)) if len(a) else np.nan,totalR=float(a.sum()) if len(a) else 0.0)

def simulate(start):
    z=tr[(tr.signal_dt>=start)&(tr.signal_dt<END)].copy().sort_values(['trade_day','entry_bar_dt','signal_dt','window'])
    states=[]
    for _ in range(N_ACCTS):
        states.append(dict(bal=SIZE,floor=SIZE-TRAIL,cycle_start=None,account_start=None,daily=defaultdict(float),paid_current=False,payouts=0,breaches=0,accounts=1,user=0.0))
    ptr=0; trades_used=0; paydates=[]; first_days=[]; resolved_total=0; resolved_success=0
    by_window=defaultdict(int)
    for day,dg in z.groupby('trade_day',sort=True):
        used=set()
        for _,r in dg.sort_values(['entry_bar_dt','signal_dt','window']).iterrows():
            if len(used)>=N_ACCTS: break
            chosen=None
            for _ in range(N_ACCTS):
                idx=ptr%N_ACCTS; ptr=(ptr+1)%N_ACCTS
                if idx not in used:
                    chosen=idx; break
            if chosen is None: break
            used.add(chosen)
            s=states[chosen]
            trades_used+=1; by_window[r.window]+=1
            if s['account_start'] is None: s['account_start']=day
            if s['cycle_start'] is None: s['cycle_start']=day
            pnl=float(r.R)*SIZE*RISK_PCT
            s['bal']+=pnl; s['daily'][day]+=pnl
            if s['bal']<s['floor']-1e-9:
                resolved_total+=1
                if s['paid_current']: resolved_success+=1
                s['breaches']+=1; s['accounts']+=1
                s.update(bal=SIZE,floor=SIZE-TRAIL,cycle_start=day,account_start=day,daily=defaultdict(float),paid_current=False)
                continue
            if (day-s['cycle_start']).days<WAIT_DAYS: continue
            profit=s['bal']-SIZE
            if profit<=0: continue
            pdays=sum(v>=PROF_DAY_MIN-1e-9 for v in s['daily'].values())
            best=max([0.0]+list(s['daily'].values()))
            cons=best/profit
            if pdays<3 or cons>CONS_MAX+1e-12: continue
            user=profit*SPLIT
            s['user']+=user; s['payouts']+=1; paydates.append(pd.Timestamp(day))
            if not s['paid_current']:
                s['paid_current']=True
                first_days.append((day-s['account_start']).days)
            s['bal']=SIZE; s['floor']=SIZE; s['cycle_start']=day; s['daily']=defaultdict(float)
    for s in states:
        if s['paid_current']:
            resolved_total+=1; resolved_success+=1
    payouts=sum(s['payouts'] for s in states)
    breaches=sum(s['breaches'] for s in states)
    accounts=sum(s['accounts'] for s in states)
    user=sum(s['user'] for s in states)
    paydates=sorted(paydates)
    gaps=[(paydates[i]-paydates[i-1]).days for i in range(1,len(paydates))]
    return dict(trades_used=trades_used,lon_open_used=by_window['LON_OPEN'],lon_h1_used=by_window['LON_H1'],payouts=payouts,breaches=breaches,accounts_used=accounts,user_paid=user,account_costs=accounts*COST,net_cash=user-accounts*COST,avg_payout=(user/payouts if payouts else np.nan),payout_before_breach_rate=(resolved_success/resolved_total if resolved_total else np.nan),resolved_accounts=resolved_total,successful_accounts=resolved_success,avg_first_payout_days=(float(np.mean(first_days)) if first_days else np.nan),median_first_payout_days=(float(np.median(first_days)) if first_days else np.nan),first_portfolio_payout_date=(str(paydates[0].date()) if paydates else ''),first_portfolio_payout_days=((paydates[0]-pd.Timestamp(start.date())).days if paydates else np.nan),median_gap_days=(float(np.median(gaps)) if gaps else np.nan))

rows=[]
for label,start in [('FULL_2012_2026',FULL_START),('YTD_2026',Y26_START)]:
    mm=market_metrics(tr[(tr.signal_dt>=start)&(tr.signal_dt<END)])
    sm=simulate(start)
    row={'period':label,'rr':RR,'risk_pct':RISK_PCT,'account_size':SIZE,'accounts':N_ACCTS,'tp_atr':TP_ATR,'sl_atr':SL_ATR}
    row.update(mm); row.update(sm); rows.append(row)

out=pd.DataFrame(rows)
out.to_csv('negative_rr_2x100k_london2_025r_results.csv',index=False)
print(out.to_string(index=False))
