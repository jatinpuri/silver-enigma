import pandas as pd
import numpy as np
from collections import defaultdict

START=200000.0
RISK_PCT=0.027
RISK=RISK_PCT*START
TRAIL=0.06*START
PROFIT_DAY_MIN=0.005*START
CONSISTENCY_MAX=0.20
WAIT_DAYS=14
SPLIT=0.80
WINDOWS={'LON_OPEN','LON_H1','NY_H1'}
START_DATE=pd.Timestamp('2012-01-01',tz='UTC')
END_DATE=pd.Timestamp('2026-08-09',tz='UTC')
COST50=349.0

tr=pd.read_csv('eurusd_multihour_trades.csv')
for c in ['signal_dt','entry_bar_dt','exit_dt']:
    tr[c]=pd.to_datetime(tr[c],utc=True)
tr=tr[(tr.signal_dt>=START_DATE)&(tr.signal_dt<END_DATE)&tr.window.isin(WINDOWS)&(tr.local_weekday<=4)].copy()
tr['trade_day']=tr.signal_dt.dt.date
tr=tr.sort_values(['trade_day','entry_bar_dt','signal_dt','window'])
tr=tr.groupby('trade_day',as_index=False,sort=True).first()
tr=tr.sort_values(['exit_dt','signal_dt','window']).reset_index(drop=True)

bal=START; floor=START-TRAIL; acct=1
account_start=None; cycle_start=None; daily=defaultdict(float)
accounts={}
payout_events=[]

def ensure_account(a,d):
    if a not in accounts:
        accounts[a]={'account':a,'start_date':str(d),'first_payout_date':'','first_payout_days':np.nan,'payout_count':0,'user_payout_total_200k':0.0,'breach_date':'','status':'active'}

for _,r in tr.iterrows():
    d=r.trade_day
    if account_start is None:
        account_start=d
    if cycle_start is None:
        cycle_start=d
    ensure_account(acct,account_start)
    pnl=float(r.R)*RISK
    bal += pnl
    daily[d] += pnl

    if bal < floor-1e-9:
        accounts[acct]['breach_date']=str(d)
        accounts[acct]['status']='breached_after_payout' if accounts[acct]['payout_count']>0 else 'breached_before_payout'
        acct += 1
        bal=START; floor=START-TRAIL
        account_start=d; cycle_start=d; daily=defaultdict(float)
        ensure_account(acct,account_start)
        continue

    if (d-cycle_start).days < WAIT_DAYS:
        continue
    profit=bal-START
    if profit <= 0:
        continue
    pdays=sum(v>=PROFIT_DAY_MIN-1e-9 for v in daily.values())
    best=max([0.0]+list(daily.values()))
    cons=best/profit
    if pdays<3 or cons>CONSISTENCY_MAX+1e-12:
        continue

    gross=bal-START
    user=gross*SPLIT
    accounts[acct]['payout_count'] += 1
    accounts[acct]['user_payout_total_200k'] += user
    if not accounts[acct]['first_payout_date']:
        accounts[acct]['first_payout_date']=str(d)
        accounts[acct]['first_payout_days']=(d-account_start).days
    payout_events.append({'account':acct,'date':str(d),'gross_200k':gross,'user_200k':user,'user_50k':user*0.25})
    bal=START; floor=START; cycle_start=d; daily=defaultdict(float)

# Mark final account if it has payout but is still active
for a,v in accounts.items():
    if v['status']=='active' and v['payout_count']>0:
        v['status']='active_after_payout'

accdf=pd.DataFrame(list(accounts.values())).sort_values('account')
accdf['success']=accdf['payout_count']>0
accdf['failed_before_payout']=accdf['status']=='breached_before_payout'

# Acquisition cycles: fresh accounts bought until one reaches its first payout.
cycles=[]; attempts=0; failures=0; cycle_start_account=None
for _,r in accdf.iterrows():
    if cycle_start_account is None:
        cycle_start_account=int(r.account)
    attempts += 1
    if bool(r.success):
        cycles.append({'cycle':len(cycles)+1,'start_account':cycle_start_account,'success_account':int(r.account),'accounts_to_first_payout':attempts,'failed_accounts_before_success':failures,'capital_cost_50k_usd':attempts*COST50,'first_payout_days_on_success_account':r.first_payout_days})
        attempts=0; failures=0; cycle_start_account=None
    elif r.status=='breached_before_payout':
        failures += 1
    else:
        # final censored active account with no payout: do not count it as a completed failure
        pass
cycledf=pd.DataFrame(cycles)

# Consecutive completed failed-account streaks in the actual account sequence.
fail_streaks=[]; cur=0
for _,r in accdf.iterrows():
    if r.status=='breached_before_payout':
        cur+=1
    elif bool(r.success):
        fail_streaks.append(cur); cur=0
# don't append trailing censored active account as a completed failure streak

completed_accounts=accdf[accdf.status!='active']
completed_successes=int((completed_accounts.payout_count>0).sum())
completed_failures=int((completed_accounts.status=='breached_before_payout').sum())
p_success=completed_successes/len(completed_accounts) if len(completed_accounts) else np.nan

summary={
    'risk_pct':RISK_PCT,
    'trades':len(tr),
    'accounts_observed':len(accdf),
    'completed_accounts':len(completed_accounts),
    'completed_success_accounts':completed_successes,
    'completed_failed_before_payout_accounts':completed_failures,
    'completed_account_success_rate':p_success,
    'successful_acquisition_cycles':len(cycledf),
    'avg_accounts_to_first_payout':cycledf.accounts_to_first_payout.mean(),
    'median_accounts_to_first_payout':cycledf.accounts_to_first_payout.median(),
    'p75_accounts_to_first_payout':cycledf.accounts_to_first_payout.quantile(.75),
    'p90_accounts_to_first_payout':cycledf.accounts_to_first_payout.quantile(.90),
    'max_accounts_to_first_payout':cycledf.accounts_to_first_payout.max(),
    'avg_failed_before_success':cycledf.failed_accounts_before_success.mean(),
    'median_failed_before_success':cycledf.failed_accounts_before_success.median(),
    'max_failed_before_success':cycledf.failed_accounts_before_success.max(),
    'max_consecutive_failed_accounts_historical':max(fail_streaks) if fail_streaks else 0,
    'empirical_pct_success_within_1_account':(cycledf.accounts_to_first_payout<=1).mean(),
    'empirical_pct_success_within_2_accounts':(cycledf.accounts_to_first_payout<=2).mean(),
    'empirical_pct_success_within_3_accounts':(cycledf.accounts_to_first_payout<=3).mean(),
    'empirical_pct_success_within_4_accounts':(cycledf.accounts_to_first_payout<=4).mean(),
    'empirical_pct_success_within_5_accounts':(cycledf.accounts_to_first_payout<=5).mean(),
    'empirical_pct_success_within_6_accounts':(cycledf.accounts_to_first_payout<=6).mean(),
    'cost_50k_one_account_usd':COST50,
    'median_bankroll_to_success_50k_usd':cycledf.accounts_to_first_payout.median()*COST50,
    'p90_bankroll_to_success_50k_usd':cycledf.accounts_to_first_payout.quantile(.90)*COST50,
    'worst_historical_bankroll_to_success_50k_usd':cycledf.accounts_to_first_payout.max()*COST50,
    'total_payout_events':len(payout_events),
    'total_user_payout_50k_usd':sum(x['user_50k'] for x in payout_events),
    'total_account_cost_50k_usd':len(accdf)*COST50,
    'net_cash_50k_usd':sum(x['user_50k'] for x in payout_events)-len(accdf)*COST50,
}

pd.DataFrame([summary]).to_csv('turbo_27pct_bankroll_streaks_summary.csv',index=False)
cycledf.to_csv('turbo_27pct_bankroll_streaks_cycles.csv',index=False)
accdf.to_csv('turbo_27pct_bankroll_streaks_accounts.csv',index=False)
pd.DataFrame(payout_events).to_csv('turbo_27pct_bankroll_streaks_payouts.csv',index=False)

print('SUMMARY')
print(pd.DataFrame([summary]).to_string(index=False))
print('\nCYCLES')
print(cycledf.to_string(index=False))
print('\nACCOUNT STATUS COUNTS')
print(accdf.status.value_counts().to_string())
