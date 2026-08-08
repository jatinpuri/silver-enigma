import pandas as pd
import numpy as np
from collections import defaultdict

START=200000.0
TRAIL=0.06*START
PROFIT_DAY_MIN=0.005*START
CONSISTENCY_MAX=0.20
WAIT_DAYS=14
START_DATE=pd.Timestamp('2012-01-01',tz='UTC')
END_DATE=pd.Timestamp('2026-08-09',tz='UTC')

tr0=pd.read_csv('eurusd_multihour_trades.csv')
for c in ['signal_dt','entry_bar_dt','exit_dt']:
    tr0[c]=pd.to_datetime(tr0[c],utc=True)
tr0=tr0[(tr0.signal_dt>=START_DATE)&(tr0.signal_dt<END_DATE)&(tr0.local_weekday<=4)].copy()
tr0['trade_day']=tr0['signal_dt'].dt.date

configs={
 'ALL3_MF':({'LON_OPEN','LON_H1','NY_H1'},{0,1,2,3,4}),
 'LONDON_MF':({'LON_OPEN','LON_H1'},{0,1,2,3,4}),
 'LONOPEN_MF':({'LON_OPEN'},{0,1,2,3,4}),
 'ALL3_MTH':({'LON_OPEN','LON_H1','NY_H1'},{0,1,2,3}),
 'LONDON_MTH':({'LON_OPEN','LON_H1'},{0,1,2,3}),
 'ALL3_TTH':({'LON_OPEN','LON_H1','NY_H1'},{1,2,3}),
}
risks=[0.006,0.008,0.010,0.012]


def prep(windows,weekdays):
    tr=tr0[tr0.window.isin(windows)&tr0.local_weekday.isin(weekdays)].copy()
    tr=tr.sort_values(['trade_day','entry_bar_dt','signal_dt','window'])
    tr=tr.groupby('trade_day',as_index=False,sort=True).first()
    return tr.sort_values(['exit_dt','signal_dt','window']).reset_index(drop=True)


def analyze(tr,risk_pct):
    risk=START*risk_pct
    bal=START; floor=START-TRAIL; acct=1
    account_start=tr.iloc[0].trade_day if len(tr) else None
    cycle_start=account_start; cycle_base=START; daily=defaultdict(float)
    first_payout_date={}
    account_start_dates={1:account_start}
    breached_accounts=set()
    payouts=0; breaches=0

    for _,r in tr.iterrows():
        day=r.trade_day
        if cycle_start is None:
            cycle_start=day
        pnl=float(r.R)*risk
        bal += pnl; daily[day]+=pnl
        if bal < floor-1e-9:
            breaches += 1; breached_accounts.add(acct)
            acct += 1; bal=START; floor=START-TRAIL
            cycle_start=day; cycle_base=START; daily=defaultdict(float)
            account_start_dates[acct]=day
            continue
        if (day-cycle_start).days < WAIT_DAYS:
            continue
        profit=bal-cycle_base
        if profit<=0: continue
        profdays=sum(v>=PROFIT_DAY_MIN-1e-9 for v in daily.values())
        best=max([0.0]+list(daily.values()))
        cons=best/profit if profit>0 else np.inf
        if profdays<3 or cons>CONSISTENCY_MAX+1e-12: continue
        payouts += 1
        if acct not in first_payout_date:
            first_payout_date[acct]=day
        bal=START; floor=START; cycle_start=day; cycle_base=START; daily=defaultdict(float)

    # accounts observed includes final active account
    accounts=acct
    successful=[]
    for a,d in first_payout_date.items():
        st=account_start_dates[a]
        successful.append((d-st).days)
    success_rate=len(first_payout_date)/accounts if accounts else np.nan
    return {
      'config':None,'risk_pct':risk_pct,'trades':len(tr),'accounts_observed':accounts,
      'accounts_with_first_payout':len(first_payout_date),'payout_before_breach_rate':success_rate,
      'avg_days_to_first_payout_successful':float(np.mean(successful)) if successful else np.nan,
      'median_days_to_first_payout_successful':float(np.median(successful)) if successful else np.nan,
      'min_days_to_first_payout_successful':float(np.min(successful)) if successful else np.nan,
      'max_days_to_first_payout_successful':float(np.max(successful)) if successful else np.nan,
      'total_payout_events':payouts,'breaches':breaches
    }

rows=[]
for name,(wins,days) in configs.items():
    tr=prep(wins,days)
    for rp in risks:
        s=analyze(tr,rp); s['config']=name; rows.append(s)

df=pd.DataFrame(rows)
df['score']=df['median_days_to_first_payout_successful'] + (1-df['payout_before_breach_rate'])*365
# lower score balances speed and survival
print(df.sort_values(['score','median_days_to_first_payout_successful']).to_string(index=False))
df.to_csv('turbo_payout_speed_grid_2012_2026.csv',index=False)
