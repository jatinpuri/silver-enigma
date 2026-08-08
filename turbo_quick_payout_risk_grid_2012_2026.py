import pandas as pd
import numpy as np
from collections import defaultdict

START=200000.0
TRAIL=0.06*START
PROFIT_DAY_MIN=0.005*START
CONSISTENCY_MAX=0.20
WAIT_DAYS=14
SPLIT=0.80
WINDOWS={'LON_OPEN','LON_H1','NY_H1'}
START_DATE=pd.Timestamp('2012-01-01',tz='UTC')
END_DATE=pd.Timestamp('2026-08-09',tz='UTC')
RISK_PCTS=[0.006,0.008,0.010,0.012,0.014,0.016,0.018,0.020,0.0225,0.025]

# Current total activation-cost assumptions used in prior analysis, USD.
ACCOUNT_COSTS={50000:349.0,100000:549.0,200000:1098.0}

tr=pd.read_csv('eurusd_multihour_trades.csv')
for c in ['signal_dt','entry_bar_dt','exit_dt']:
    tr[c]=pd.to_datetime(tr[c],utc=True)
tr=tr[(tr.signal_dt>=START_DATE)&(tr.signal_dt<END_DATE)&tr.window.isin(WINDOWS)&(tr.local_weekday<=4)].copy()
tr['trade_day']=tr['signal_dt'].dt.date
tr=tr.sort_values(['trade_day','entry_bar_dt','signal_dt','window'])
tr=tr.groupby('trade_day',as_index=False,sort=True).first()
tr=tr.sort_values(['exit_dt','signal_dt','window']).reset_index(drop=True)

rows=[]
for risk_pct in RISK_PCTS:
    risk=risk_pct*START
    bal=START; floor=START-TRAIL; acct=1
    cycle_start=None; cycle_base=START; daily=defaultdict(float)
    account_start=None
    first_payout_by_acct={}
    payout_dates=[]; payout_amounts=[]; breaches=[]
    last_payout_date=None; gap_days=[]

    for _,r in tr.iterrows():
        day=r.trade_day
        if cycle_start is None: cycle_start=day
        if account_start is None: account_start=day
        pnl=float(r.R)*risk
        bal+=pnl; daily[day]+=pnl

        if bal < floor-1e-9:
            breaches.append(day)
            acct+=1; bal=START; floor=START-TRAIL
            cycle_start=day; cycle_base=START; daily=defaultdict(float)
            account_start=day
            continue

        if (day-cycle_start).days < WAIT_DAYS: continue
        cycle_profit=bal-cycle_base
        if cycle_profit<=0: continue
        profdays=sum(v>=PROFIT_DAY_MIN-1e-9 for v in daily.values())
        best=max([0.0]+list(daily.values()))
        cons=best/cycle_profit if cycle_profit>0 else np.inf
        if profdays<3 or cons>CONSISTENCY_MAX+1e-12: continue

        gross=bal-START
        if gross<=0: continue
        user=gross*SPLIT
        payout_dates.append(day); payout_amounts.append(user)
        if acct not in first_payout_by_acct:
            first_payout_by_acct[acct]=(day-account_start).days
        if last_payout_date is not None:
            gap_days.append((day-last_payout_date).days)
        last_payout_date=day

        bal=START; floor=START
        cycle_start=day; cycle_base=START; daily=defaultdict(float)

    first_days=list(first_payout_by_acct.values())
    accounts_used=acct
    total_user_paid=sum(payout_amounts)
    # Scale dollar payouts to each nominal account size; timing/breaches are percentage-identical.
    base={
        'risk_pct':risk_pct,
        'trades':len(tr),
        'payouts':len(payout_dates),
        'breaches':len(breaches),
        'accounts_used':accounts_used,
        'accounts_with_first_payout':len(first_days),
        'payout_before_breach_rate':len(first_days)/accounts_used if accounts_used else np.nan,
        'avg_days_first_payout_success':np.mean(first_days) if first_days else np.nan,
        'median_days_first_payout_success':np.median(first_days) if first_days else np.nan,
        'avg_days_between_payouts':np.mean(gap_days) if gap_days else np.nan,
        'median_days_between_payouts':np.median(gap_days) if gap_days else np.nan,
        'payouts_per_year':len(payout_dates)/((END_DATE-START_DATE).days/365.25),
        'user_paid_200k':total_user_paid,
    }
    for size,cost in ACCOUNT_COSTS.items():
        scale=size/START
        paid=total_user_paid*scale
        total_cost=accounts_used*cost
        base[f'user_paid_{size//1000}k']=paid
        base[f'account_costs_{size//1000}k']=total_cost
        base[f'net_cash_{size//1000}k']=paid-total_cost
        years=(END_DATE-START_DATE).days/365.25
        base[f'net_cash_per_year_{size//1000}k']=(paid-total_cost)/years
        base[f'net_per_account_cost_{size//1000}k']=(paid-total_cost)/total_cost if total_cost>0 else np.nan
    rows.append(base)

out=pd.DataFrame(rows)
out.to_csv('turbo_quick_payout_risk_grid_2012_2026.csv',index=False)
print(out.to_string(index=False))
