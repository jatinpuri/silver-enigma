import pandas as pd
import numpy as np
from collections import defaultdict

START_DATE=pd.Timestamp('2012-01-01',tz='UTC')
END_DATE=pd.Timestamp('2026-08-09',tz='UTC')
RISK_PCT=0.027
SPLIT=0.80
WAIT_DAYS=14
CONSISTENCY_MAX=0.20
WINDOWS={'LON_OPEN','LON_H1','NY_H1'}

CONFIGS=[
    {'name':'1x100k','n':1,'size':100000.0,'cost':549.0},
    {'name':'2x50k','n':2,'size':50000.0,'cost':349.0},
    {'name':'4x25k','n':4,'size':25000.0,'cost':189.0},
]

tr=pd.read_csv('eurusd_multihour_trades.csv')
for c in ['signal_dt','entry_bar_dt','exit_dt']:
    tr[c]=pd.to_datetime(tr[c],utc=True)
tr=tr[(tr.signal_dt>=START_DATE)&(tr.signal_dt<END_DATE)&tr.window.isin(WINDOWS)&(tr.local_weekday<=4)].copy()
tr['trade_day']=tr.signal_dt.dt.date
tr=tr.sort_values(['trade_day','entry_bar_dt','signal_dt','window']).reset_index(drop=True)

years=(END_DATE-START_DATE).days/365.25
summary=[]
slot_rows=[]

for cfg in CONFIGS:
    n=cfg['n']; size=cfg['size']; cost=cfg['cost']
    risk=size*RISK_PCT
    trail=size*0.06
    profitable_day_min=size*0.005
    states=[]
    for i in range(n):
        states.append({
            'bal':size,'floor':size-trail,'cycle_start':None,'account_start':None,
            'daily':defaultdict(float),'account_num':1,'accounts_used':1,
            'payouts':0,'breaches':0,'user_paid':0.0,'first_payout_days':[],
            'first_paid_current':False,'trade_count':0
        })
    rr_ptr=0
    portfolio_paydates=[]
    source_trades_used=0

    for day, dg in tr.groupby('trade_day', sort=True):
        dg=dg.sort_values(['entry_bar_dt','signal_dt','window'])
        used=set()
        # Assign as many triggered setups as there are accounts, max one trade/account/day.
        for _,r in dg.iterrows():
            if len(used)>=n:
                break
            # choose next round-robin slot not already used today
            chosen=None
            for _try in range(n):
                idx=rr_ptr % n
                rr_ptr=(rr_ptr+1)%n
                if idx not in used:
                    chosen=idx
                    break
            if chosen is None:
                break
            used.add(chosen)
            s=states[chosen]
            source_trades_used += 1
            s['trade_count'] += 1
            if s['account_start'] is None:
                s['account_start']=day
            if s['cycle_start'] is None:
                s['cycle_start']=day

            pnl=float(r.R)*risk
            s['bal'] += pnl
            s['daily'][day] += pnl

            # Established closed-P&L simulation: fixed 6% floor before payout; floor locks to initial after payout.
            if s['bal'] < s['floor'] - 1e-9:
                s['breaches'] += 1
                s['account_num'] += 1
                s['accounts_used'] += 1
                s['bal']=size
                s['floor']=size-trail
                s['cycle_start']=day
                s['account_start']=day
                s['daily']=defaultdict(float)
                s['first_paid_current']=False
                continue

            if (day-s['cycle_start']).days < WAIT_DAYS:
                continue
            profit=s['bal']-size
            if profit <= 0:
                continue
            pdays=sum(v >= profitable_day_min-1e-9 for v in s['daily'].values())
            best=max([0.0]+list(s['daily'].values()))
            consistency=best/profit if profit>0 else np.inf
            if pdays<3 or consistency>CONSISTENCY_MAX+1e-12:
                continue

            gross=profit
            user=gross*SPLIT
            s['payouts'] += 1
            s['user_paid'] += user
            portfolio_paydates.append(pd.Timestamp(day))
            if not s['first_paid_current']:
                s['first_payout_days'].append((day-s['account_start']).days)
                s['first_paid_current']=True
            # Withdraw all profit; locked floor at initial after payout.
            s['bal']=size
            s['floor']=size
            s['cycle_start']=day
            s['daily']=defaultdict(float)

    portfolio_paydates=sorted(portfolio_paydates)
    gaps=[(portfolio_paydates[i]-portfolio_paydates[i-1]).days for i in range(1,len(portfolio_paydates))]
    payouts=sum(s['payouts'] for s in states)
    breaches=sum(s['breaches'] for s in states)
    accounts_used=sum(s['accounts_used'] for s in states)
    user_paid=sum(s['user_paid'] for s in states)
    account_cost=accounts_used*cost
    net=user_paid-account_cost
    all_first=[x for s in states for x in s['first_payout_days']]

    summary.append({
        'setup':cfg['name'],'total_nominal_capital':n*size,'accounts_live':n,
        'account_size':size,'risk_pct':RISK_PCT,'risk_usd_per_trade_per_account':risk,
        'source_trades_used':source_trades_used,'payouts':payouts,'breaches':breaches,
        'accounts_purchased_total':accounts_used,'user_payout_usd':user_paid,
        'account_cost_usd':account_cost,'net_cash_usd':net,'net_cash_per_year_usd':net/years,
        'avg_user_payout_usd':user_paid/payouts if payouts else np.nan,
        'avg_days_between_portfolio_payouts':np.mean(gaps) if gaps else np.nan,
        'median_days_between_portfolio_payouts':np.median(gaps) if gaps else np.nan,
        'avg_first_payout_days_successful_accounts':np.mean(all_first) if all_first else np.nan,
        'median_first_payout_days_successful_accounts':np.median(all_first) if all_first else np.nan,
        'first_portfolio_payout_date':portfolio_paydates[0].date() if portfolio_paydates else None,
        'first_portfolio_payout_days_from_2012_start':(portfolio_paydates[0]-pd.Timestamp('2012-01-01')).days if portfolio_paydates else np.nan
    })

    for i,s in enumerate(states,1):
        slot_rows.append({
            'setup':cfg['name'],'slot':i,'trades':s['trade_count'],'payouts':s['payouts'],
            'breaches':s['breaches'],'accounts_used':s['accounts_used'],'user_paid_usd':s['user_paid'],
            'avg_first_payout_days_successful_accounts':np.mean(s['first_payout_days']) if s['first_payout_days'] else np.nan,
            'median_first_payout_days_successful_accounts':np.median(s['first_payout_days']) if s['first_payout_days'] else np.nan
        })

out=pd.DataFrame(summary)
out.to_csv('turbo_100k_split_compare_27pct_summary.csv',index=False)
pd.DataFrame(slot_rows).to_csv('turbo_100k_split_compare_27pct_slots.csv',index=False)
print(out.to_string(index=False))
