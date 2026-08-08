import pandas as pd
import numpy as np
from collections import defaultdict

START=200000.0; TRAIL=0.06*START; PROFIT_DAY_MIN=0.005*START
CONSISTENCY_MAX=0.20; WAIT_DAYS=14; SPLIT=0.80
WINDOWS={'LON_OPEN','LON_H1','NY_H1'}
START_DATE=pd.Timestamp('2012-01-01',tz='UTC'); END_DATE=pd.Timestamp('2026-08-09',tz='UTC')
RISK_PCTS=[0.024,0.025,0.026,0.027,0.028,0.029]
COST50=349.0
tr=pd.read_csv('eurusd_multihour_trades.csv')
for c in ['signal_dt','entry_bar_dt','exit_dt']: tr[c]=pd.to_datetime(tr[c],utc=True)
tr=tr[(tr.signal_dt>=START_DATE)&(tr.signal_dt<END_DATE)&tr.window.isin(WINDOWS)&(tr.local_weekday<=4)].copy()
tr['trade_day']=tr.signal_dt.dt.date
tr=tr.sort_values(['trade_day','entry_bar_dt','signal_dt','window']).groupby('trade_day',as_index=False,sort=True).first()
tr=tr.sort_values(['exit_dt','signal_dt','window']).reset_index(drop=True)
years=(END_DATE-START_DATE).days/365.25
rows=[]
for rp in RISK_PCTS:
    risk=rp*START; bal=START; floor=START-TRAIL; acct=1; account_start=None
    cycle_start=None; daily=defaultdict(float); first={}; pays=[]; paydates=[]; breaches=0; gaps=[]; last=None
    for _,r in tr.iterrows():
        d=r.trade_day
        if account_start is None: account_start=d
        if cycle_start is None: cycle_start=d
        pnl=float(r.R)*risk; bal+=pnl; daily[d]+=pnl
        if bal<floor-1e-9:
            breaches+=1; acct+=1; bal=START; floor=START-TRAIL; account_start=d; cycle_start=d; daily=defaultdict(float); continue
        if (d-cycle_start).days<WAIT_DAYS: continue
        profit=bal-START
        if profit<=0: continue
        pdays=sum(v>=PROFIT_DAY_MIN-1e-9 for v in daily.values()); best=max([0.0]+list(daily.values())); cons=best/profit
        if pdays<3 or cons>CONSISTENCY_MAX+1e-12: continue
        gross=bal-START; user=gross*SPLIT; pays.append(user); paydates.append(d)
        if acct not in first: first[acct]=(d-account_start).days
        if last is not None: gaps.append((d-last).days)
        last=d; bal=START; floor=START; cycle_start=d; daily=defaultdict(float)
    fd=list(first.values()); user200=sum(pays); user50=user200*.25; costs=acct*COST50
    rows.append({'risk_pct':rp,'payouts':len(pays),'breaches':breaches,'accounts_used':acct,'accounts_with_first_payout':len(fd),'payout_before_breach_rate':len(fd)/acct,'avg_days_first_payout_success':np.mean(fd) if fd else np.nan,'median_days_first_payout_success':np.median(fd) if fd else np.nan,'avg_days_between_payouts':np.mean(gaps) if gaps else np.nan,'median_days_between_payouts':np.median(gaps) if gaps else np.nan,'payouts_per_year':len(pays)/years,'user_paid_50k':user50,'account_costs_50k':costs,'net_cash_50k':user50-costs,'net_cash_per_year_50k':(user50-costs)/years})
pd.DataFrame(rows).to_csv('turbo_quick_payout_highrisk_grid_2012_2026.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
