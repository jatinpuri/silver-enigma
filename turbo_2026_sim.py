import pandas as pd
import numpy as np
from collections import defaultdict

START_BAL=50000.0
RISK_PCT=0.025
RISK_DOLLARS=START_BAL*RISK_PCT
TRAIL_PCT=0.06
PROFIT_DAY_MIN=START_BAL*0.005
CONSISTENCY_MAX=0.20
PAYOUT_WAIT_DAYS=14
SPLIT=0.80
TARGET_WINDOWS={'LON_OPEN','LON_H1','NY_H1'}

tr=pd.read_csv('eurusd_multihour_trades.csv')
for c in ['signal_dt','entry_bar_dt','exit_dt']:
    tr[c]=pd.to_datetime(tr[c],utc=True)
tr=tr[(tr['year']==2026)&(tr['window'].isin(TARGET_WINDOWS))&(tr['local_weekday']<=4)].copy()
tr=tr.sort_values(['exit_dt','signal_dt','window']).reset_index(drop=True)

# Approximation note: drawdown is checked on realised/closed P&L events. The official rule is equity-based,
# so floating adverse excursion between H1 closes is not observable from this trade-result file.

def simulate(df, label):
    balance=START_BAL
    high_water=START_BAL
    floor=START_BAL*(1-TRAIL_PCT)
    floor_locked=False
    cycle_start=None
    cycle_daily=defaultdict(float)
    payouts=[]
    breaches=[]
    account_no=1
    gross_paid=0.0
    user_paid=0.0
    trades_count=0

    def reset_account(next_date):
        nonlocal balance,high_water,floor,floor_locked,cycle_start,cycle_daily,account_no
        balance=START_BAL
        high_water=START_BAL
        floor=START_BAL*(1-TRAIL_PCT)
        floor_locked=False
        cycle_start=next_date
        cycle_daily=defaultdict(float)
        account_no += 1

    for _,r in df.iterrows():
        dt=r['exit_dt']
        day=dt.date()
        if cycle_start is None:
            cycle_start=day
        trades_count+=1
        pnl=float(r['R'])*RISK_DOLLARS
        pre=balance
        balance += pnl
        cycle_daily[day] += pnl

        # Breach check against the floor that existed before this close.
        if balance < floor - 1e-9:
            breaches.append(dict(label=label,account=account_no,date=str(day),balance=balance,floor=floor,pnl=pnl,window=r['window']))
            reset_account(day)
            continue

        # Trail only on new closed-balance highs, then lock at starting balance once +6% is achieved.
        if balance > high_water:
            high_water=balance
            if high_water >= START_BAL*(1+TRAIL_PCT):
                floor=START_BAL
                floor_locked=True
            elif not floor_locked:
                floor=max(START_BAL*(1-TRAIL_PCT), high_water-START_BAL*TRAIL_PCT)

        # Check payout after at least 14 calendar days in current cycle.
        days_elapsed=(day-cycle_start).days
        if days_elapsed < PAYOUT_WAIT_DAYS:
            continue
        gross_profit=balance-START_BAL
        if gross_profit <= 0:
            continue
        profitable_days=sum(1 for v in cycle_daily.values() if v >= PROFIT_DAY_MIN-1e-9)
        best_day=max([v for v in cycle_daily.values()]+[0.0])
        consistency=(best_day/gross_profit) if gross_profit>0 else np.inf
        if profitable_days < 3 or consistency > CONSISTENCY_MAX+1e-12:
            continue

        # Request immediately and withdraw all profit above initial balance.
        gross=gross_profit
        user=gross*SPLIT
        payouts.append(dict(label=label,account=account_no,date=str(day),gross_profit=gross,user_payout=user,
                            profitable_days=profitable_days,best_day=best_day,consistency=consistency,
                            balance_before=balance,floor_before=floor))
        gross_paid += gross
        user_paid += user
        balance=START_BAL
        # Official rule: once payout is requested the DD limit locks at the initial account balance.
        floor=START_BAL
        high_water=max(high_water,START_BAL)
        floor_locked=True
        cycle_start=day
        cycle_daily=defaultdict(float)

    return dict(label=label,trades=trades_count,payouts=len(payouts),breaches=len(breaches),accounts_used=account_no,
                gross_paid=gross_paid,user_paid=user_paid,ending_balance=balance,ending_floor=floor), payouts, breaches

# Scenario A: one 50k funded account trades all three windows.
a,pa,ba=simulate(tr,'ONE_ACCOUNT_ALL_WINDOWS')

# Scenario B: user's historical split style — one 50k account for London windows, one 50k for NY window.
lon=tr[tr.window.isin({'LON_OPEN','LON_H1'})].copy()
ny=tr[tr.window.eq('NY_H1')].copy()
b1,pb1,bb1=simulate(lon,'SPLIT_LONDON_ACCOUNT')
b2,pb2,bb2=simulate(ny,'SPLIT_NY_ACCOUNT')
b=dict(label='TWO_ACCOUNT_SPLIT_TOTAL',trades=b1['trades']+b2['trades'],payouts=b1['payouts']+b2['payouts'],
       breaches=b1['breaches']+b2['breaches'],accounts_used=b1['accounts_used']+b2['accounts_used'],
       gross_paid=b1['gross_paid']+b2['gross_paid'],user_paid=b1['user_paid']+b2['user_paid'],
       ending_balance=b1['ending_balance']+b2['ending_balance'],ending_floor=b1['ending_floor']+b2['ending_floor'])

pd.DataFrame([a,b1,b2,b]).to_csv('turbo_2026_summary.csv',index=False)
pd.DataFrame(pa+pb1+pb2).to_csv('turbo_2026_payouts.csv',index=False)
pd.DataFrame(ba+bb1+bb2).to_csv('turbo_2026_breaches.csv',index=False)
print(pd.DataFrame([a,b1,b2,b]).to_string(index=False))
print('\nPAYOUTS')
print(pd.DataFrame(pa+pb1+pb2).to_string(index=False))
print('\nBREACHES')
print(pd.DataFrame(ba+bb1+bb2).to_string(index=False))
