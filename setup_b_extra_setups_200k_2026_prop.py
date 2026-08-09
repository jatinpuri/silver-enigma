from collections import defaultdict
import numpy as np
import pandas as pd

SIZE=200000.0; TRAIL=0.06*SIZE; DAILY_LIMIT=0.03*SIZE
PROFIT_DAY_MIN=0.005*SIZE; WAIT_DAYS=14; CONS_MAX=0.20; SPLIT=0.80; ACCOUNT_COST=1098.0
RISK_PCTS=[0.005,0.006,0.0075,0.008,0.01,0.0125,0.015,0.0175,0.02,0.0225,0.025,0.027,0.028,0.029]
START=pd.Timestamp('2026-01-01',tz='UTC'); END=pd.Timestamp('2026-08-09',tz='UTC')

tr=pd.read_csv('setup_b_extra_setups_portfolio_trades.csv')
tr['dt']=pd.to_datetime(tr['dt'],utc=True)
tr=tr[(tr.dt>=START)&(tr.dt<END)].sort_values('dt').reset_index(drop=True)

def sim(risk_pct,col):
    bal=SIZE; peak=SIZE; floor=SIZE-TRAIL; aid=1; account_start=None; cycle_start=None; daily=defaultdict(float); paid_current=False
    payouts=[]; breaches=[]; first=[]; resolved=0; successful=0
    risk=SIZE*risk_pct
    for r in tr.itertuples(index=False):
        day=pd.Timestamp(r.dt).date()
        if account_start is None: account_start=day
        if cycle_start is None: cycle_start=day
        pnl=float(getattr(r,col))*risk
        bal+=pnl; daily[day]+=pnl
        daily_breach=daily[day] <= -DAILY_LIMIT + 1e-9
        trail_breach=bal <= floor + 1e-9
        if daily_breach or trail_breach:
            resolved+=1
            if paid_current: successful+=1
            breaches.append((day,aid,'DAILY' if daily_breach else 'TRAIL'))
            aid+=1; bal=SIZE; peak=SIZE; floor=SIZE-TRAIL; account_start=day; cycle_start=day; daily=defaultdict(float); paid_current=False
            continue
        if bal>peak: peak=bal
        floor=min(SIZE,peak-TRAIL)
        if (day-cycle_start).days<WAIT_DAYS: continue
        profit=bal-SIZE
        if profit<=0: continue
        profdays=sum(v>=PROFIT_DAY_MIN-1e-9 for v in daily.values())
        best=max([0.0]+list(daily.values())); cons=best/profit if profit>0 else np.inf
        if profdays<3 or cons>CONS_MAX+1e-12: continue
        user=profit*SPLIT; payouts.append((day,aid,user,profit,profdays,cons))
        if not paid_current:
            paid_current=True; first.append((day-account_start).days)
        bal=SIZE; peak=SIZE+TRAIL; floor=SIZE; cycle_start=day; daily=defaultdict(float)
    if paid_current: resolved+=1; successful+=1
    paid=sum(x[2] for x in payouts); pdates=[pd.Timestamp(x[0]) for x in payouts]; gaps=[(pdates[i]-pdates[i-1]).days for i in range(1,len(pdates))]
    return dict(risk_pct=risk_pct,trades=len(tr),payouts=len(payouts),breaches=len(breaches),accounts_used=aid,user_payouts=paid,activation_costs=aid*ACCOUNT_COST,net_cash=paid-aid*ACCOUNT_COST,avg_payout=(paid/len(payouts) if payouts else np.nan),payout_before_breach=(successful/resolved if resolved else np.nan),median_first_payout_days=(np.median(first) if first else np.nan),first_payout_date=(str(pdates[0].date()) if pdates else ''),median_gap_days=(np.median(gaps) if gaps else np.nan))

rows=[]
for model,col in [('COMMISSION','R_comm'),('COMM_PLUS_0P2','R_stress')]:
    for rp in RISK_PCTS:
        rows.append(dict(cost_model=model,**sim(rp,col)))
out=pd.DataFrame(rows)
out.to_csv('setup_b_extra_setups_200k_2026_prop_results.csv',index=False)
print(out.to_string(index=False))
