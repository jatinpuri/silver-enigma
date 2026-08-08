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

# Drawdown checks use realised/closed P&L because the trade-result file does not contain full intratrade equity paths.
def simulate(df, label, trail_before_payout):
    balance=START_BAL
    high_water=START_BAL
    floor=START_BAL*(1-TRAIL_PCT)
    payout_seen=False
    cycle_start=None
    cycle_daily=defaultdict(float)
    payouts=[]
    breaches=[]
    account_no=1
    gross_paid=0.0
    user_paid=0.0
    trades_count=0

    def reset_account(next_date):
        nonlocal balance,high_water,floor,payout_seen,cycle_start,cycle_daily,account_no
        balance=START_BAL
        high_water=START_BAL
        floor=START_BAL*(1-TRAIL_PCT)
        payout_seen=False
        cycle_start=next_date
        cycle_daily=defaultdict(float)
        account_no += 1

    for _,r in df.iterrows():
        dt=r['exit_dt']; day=dt.date()
        if cycle_start is None:
            cycle_start=day
        trades_count+=1
        pnl=float(r['R'])*RISK_DOLLARS
        balance += pnl
        cycle_daily[day] += pnl

        if balance < floor - 1e-9:
            breaches.append(dict(label=label,account=account_no,date=str(day),balance=balance,floor=floor,pnl=pnl,window=r['window']))
            reset_account(day)
            continue

        # Official mode: continuously trail closed-balance highs until +6%, then lock at initial balance.
        # Custom mode: keep a fixed 6% floor before the first payout; after payout the floor locks at initial balance.
        if trail_before_payout and not payout_seen and balance > high_water:
            high_water=balance
            if high_water >= START_BAL*(1+TRAIL_PCT):
                floor=START_BAL
            else:
                floor=max(START_BAL*(1-TRAIL_PCT), high_water-START_BAL*TRAIL_PCT)

        days_elapsed=(day-cycle_start).days
        if days_elapsed < PAYOUT_WAIT_DAYS:
            continue
        gross_profit=balance-START_BAL
        if gross_profit <= 0:
            continue
        profitable_days=sum(1 for v in cycle_daily.values() if v >= PROFIT_DAY_MIN-1e-9)
        best_day=max([v for v in cycle_daily.values()]+[0.0])
        consistency=best_day/gross_profit
        if profitable_days < 3 or consistency > CONSISTENCY_MAX+1e-12:
            continue

        gross=gross_profit; user=gross*SPLIT
        payouts.append(dict(label=label,account=account_no,date=str(day),gross_profit=gross,user_payout=user,
                            profitable_days=profitable_days,best_day=best_day,consistency=consistency,
                            balance_before=balance,floor_before=floor))
        gross_paid += gross; user_paid += user
        balance=START_BAL
        floor=START_BAL
        high_water=max(high_water,START_BAL)
        payout_seen=True
        cycle_start=day
        cycle_daily=defaultdict(float)

    # End-of-period eligibility snapshot.
    gross_profit=balance-START_BAL
    best_day=max([v for v in cycle_daily.values()]+[0.0])
    prof_days=sum(1 for v in cycle_daily.values() if v >= PROFIT_DAY_MIN-1e-9)
    consistency=(best_day/gross_profit) if gross_profit>0 else np.inf
    return dict(label=label,trades=trades_count,payouts=len(payouts),breaches=len(breaches),accounts_used=account_no,
                gross_paid=gross_paid,user_paid=user_paid,ending_balance=balance,ending_floor=floor,
                ending_cycle_profit=gross_profit,ending_profitable_days=prof_days,ending_best_day=best_day,
                ending_consistency=consistency), payouts, breaches

lon=tr[tr.window.isin({'LON_OPEN','LON_H1'})].copy()
ny=tr[tr.window.eq('NY_H1')].copy()

rows=[]; all_payouts=[]; all_breaches=[]
for mode,trail in [('OFFICIAL_CONTINUOUS_TRAIL',True),('CUSTOM_FIXED6_UNTIL_PAYOUT',False)]:
    a,pa,ba=simulate(tr,f'{mode}_ONE_ACCOUNT',trail)
    l,pl,bl=simulate(lon,f'{mode}_LONDON_ACCOUNT',trail)
    n,pn,bn=simulate(ny,f'{mode}_NY_ACCOUNT',trail)
    total=dict(label=f'{mode}_TWO_ACCOUNT_SPLIT_TOTAL',trades=l['trades']+n['trades'],payouts=l['payouts']+n['payouts'],
               breaches=l['breaches']+n['breaches'],accounts_used=l['accounts_used']+n['accounts_used'],
               gross_paid=l['gross_paid']+n['gross_paid'],user_paid=l['user_paid']+n['user_paid'],
               ending_balance=l['ending_balance']+n['ending_balance'],ending_floor=l['ending_floor']+n['ending_floor'],
               ending_cycle_profit=l['ending_cycle_profit']+n['ending_cycle_profit'],
               ending_profitable_days=l['ending_profitable_days']+n['ending_profitable_days'],
               ending_best_day=max(l['ending_best_day'],n['ending_best_day']),ending_consistency=np.nan)
    rows += [a,l,n,total]
    all_payouts += pa+pl+pn
    all_breaches += ba+bl+bn

pd.DataFrame(rows).to_csv('turbo_2026_summary.csv',index=False)
pd.DataFrame(all_payouts).to_csv('turbo_2026_payouts.csv',index=False)
pd.DataFrame(all_breaches).to_csv('turbo_2026_breaches.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
print('\nPAYOUTS')
print(pd.DataFrame(all_payouts).to_string(index=False))
print('\nBREACHES')
print(pd.DataFrame(all_breaches).to_string(index=False))
