import pandas as pd
import numpy as np
from collections import defaultdict

START=200000.0
TRAIL=0.06*START
PROFIT_DAY_MIN=0.005*START
CONSISTENCY_MAX=0.20
WAIT_DAYS=14
SPLIT=0.80
START_DATE=pd.Timestamp('2012-01-01',tz='UTC')
END_DATE=pd.Timestamp('2026-08-09',tz='UTC')
RISKS=[x/10000 for x in range(250,296,5)]  # 2.50% to 2.95% in 0.05% steps
CONFIGS={
 'ALL3_MF':({'LON_OPEN','LON_H1','NY_H1'},set(range(5))),
 'LONDON_MF':({'LON_OPEN','LON_H1'},set(range(5))),
 'ALL3_MTH':({'LON_OPEN','LON_H1','NY_H1'},{0,1,2,3}),
 'ALL3_TTH':({'LON_OPEN','LON_H1','NY_H1'},{1,2,3}),
}

raw=pd.read_csv('eurusd_multihour_trades.csv')
for c in ['signal_dt','entry_bar_dt','exit_dt']:
    raw[c]=pd.to_datetime(raw[c],utc=True)
raw=raw[(raw.signal_dt>=START_DATE)&(raw.signal_dt<END_DATE)].copy()
raw['trade_day']=raw.signal_dt.dt.date

def select_trades(windows, weekdays):
    t=raw[raw.window.isin(windows)&raw.local_weekday.isin(weekdays)].copy()
    t=t.sort_values(['trade_day','entry_bar_dt','signal_dt','window'])
    t=t.groupby('trade_day',as_index=False,sort=True).first()
    return t.sort_values(['exit_dt','signal_dt','window']).reset_index(drop=True)

def simulate(tr,rp):
    risk=rp*START
    bal=START; floor=START-TRAIL; acct=1
    account_start=None; cycle_start=None; daily=defaultdict(float)
    first={}; payouts=[]; payout_dates=[]; breaches=0; lastpay=None; gaps=[]
    for _,r in tr.iterrows():
        d=r.trade_day
        if account_start is None: account_start=d
        if cycle_start is None: cycle_start=d
        pnl=float(r.R)*risk
        bal+=pnl; daily[d]+=pnl
        if bal<floor-1e-9:
            breaches+=1; acct+=1; bal=START; floor=START-TRAIL
            account_start=d; cycle_start=d; daily=defaultdict(float)
            continue
        if (d-cycle_start).days<WAIT_DAYS: continue
        profit=bal-START
        if profit<=0: continue
        pdays=sum(v>=PROFIT_DAY_MIN-1e-9 for v in daily.values())
        best=max([0.0]+list(daily.values()))
        cons=best/profit if profit>0 else np.inf
        if pdays<3 or cons>CONSISTENCY_MAX+1e-12: continue
        user=profit*SPLIT; payouts.append(user); payout_dates.append(d)
        if acct not in first: first[acct]=(d-account_start).days
        if lastpay is not None: gaps.append((d-lastpay).days)
        lastpay=d
        bal=START; floor=START; cycle_start=d; daily=defaultdict(float)
    fd=list(first.values())
    return {
      'trades':len(tr),'risk_pct':rp,'payouts':len(payouts),'breaches':breaches,'accounts_used':acct,
      'accounts_with_first_payout':len(fd),'payout_before_breach_rate':len(fd)/acct if acct else np.nan,
      'avg_days_first_payout_success':np.mean(fd) if fd else np.nan,
      'median_days_first_payout_success':np.median(fd) if fd else np.nan,
      'p25_days_first_payout_success':np.percentile(fd,25) if fd else np.nan,
      'p75_days_first_payout_success':np.percentile(fd,75) if fd else np.nan,
      'min_days_first_payout_success':np.min(fd) if fd else np.nan,
      'avg_days_between_payouts':np.mean(gaps) if gaps else np.nan,
      'median_days_between_payouts':np.median(gaps) if gaps else np.nan,
      'user_paid_200k':sum(payouts)
    }

rows=[]
for name,(wins,wds) in CONFIGS.items():
    tr=select_trades(wins,wds)
    for rp in RISKS:
        d=simulate(tr,rp); d['config']=name; rows.append(d)
out=pd.DataFrame(rows)
out['rank_median']=out['median_days_first_payout_success'].rank(method='min')
out=out.sort_values(['median_days_first_payout_success','avg_days_first_payout_success','payout_before_breach_rate'],ascending=[True,True,False])
out.to_csv('turbo_fastest_payout_finegrid_2012_2026.csv',index=False)
print(out.head(30).to_string(index=False))
