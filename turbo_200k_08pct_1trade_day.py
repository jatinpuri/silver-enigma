import pandas as pd
import numpy as np
from collections import defaultdict

START=200000.0
RISK=0.008*START
TRAIL=0.06*START
PROFIT_DAY_MIN=0.005*START
CONSISTENCY_MAX=0.20
WAIT=14
SPLIT=0.80
WINDOWS={'LON_OPEN','LON_H1','NY_H1'}

tr=pd.read_csv('eurusd_multihour_trades.csv')
for c in ['signal_dt','entry_bar_dt','exit_dt']:
    tr[c]=pd.to_datetime(tr[c],utc=True)
tr=tr[(tr.year==2026)&(tr.window.isin(WINDOWS))&(tr.local_weekday<=4)].copy()
tr['trade_day']=tr['signal_dt'].dt.date
tr=tr.sort_values(['trade_day','entry_bar_dt','signal_dt','window'])
tr=tr.groupby('trade_day',as_index=False,sort=True).first()
tr=tr.sort_values(['exit_dt','signal_dt','window']).reset_index(drop=True)

def sim(mode):
    bal=START; high=START; floor=START-TRAIL; locked=False
    acct=1; cycle_start=None; daily=defaultdict(float)
    payouts=[]; breaches=[]; used=[]
    for _,r in tr.iterrows():
        day=r.trade_day
        if cycle_start is None: cycle_start=day
        pnl=float(r.R)*RISK
        bal += pnl; daily[day]+=pnl
        used.append((str(day),r.window,float(r.R),pnl,str(r.exit_dt)))
        if bal < floor-1e-9:
            breaches.append((acct,str(day),bal,floor,pnl,r.window))
            acct+=1; bal=START; high=START; floor=START-TRAIL; locked=False
            cycle_start=day; daily=defaultdict(float)
            continue
        if mode=='official':
            if bal>high:
                high=bal
                if high>=START+TRAIL:
                    floor=START; locked=True
                elif not locked:
                    floor=max(START-TRAIL,high-TRAIL)
        if (day-cycle_start).days < WAIT: continue
        profit=bal-START
        if profit<=0: continue
        profdays=sum(v>=PROFIT_DAY_MIN-1e-9 for v in daily.values())
        best=max([0.0]+list(daily.values()))
        cons=best/profit if profit>0 else np.inf
        if profdays<3 or cons>CONSISTENCY_MAX+1e-12: continue
        user=profit*SPLIT
        payouts.append((acct,str(day),profit,user,profdays,best,cons,bal,floor))
        bal=START; floor=START; high=max(high,START); locked=True
        cycle_start=day; daily=defaultdict(float)
    profit=bal-START
    best=max([0.0]+list(daily.values()))
    profdays=sum(v>=PROFIT_DAY_MIN-1e-9 for v in daily.values())
    cons=best/profit if profit>0 else np.inf
    return {
      'mode':mode,'trades':len(tr),'payouts':len(payouts),'breaches':len(breaches),'accounts_used':acct,
      'gross_paid':sum(x[2] for x in payouts),'user_paid':sum(x[3] for x in payouts),
      'ending_balance':bal,'ending_floor':floor,'ending_cycle_profit':profit,
      'ending_profitable_days':profdays,'ending_best_day':best,'ending_consistency':cons
    }, payouts, breaches, used

rows=[]; pays=[]; br=[]; trades_used=None
for mode in ['custom','official']:
    s,p,b,u=sim(mode); rows.append(s)
    if trades_used is None: trades_used=u
    for x in p:
        pays.append({'mode':mode,'account':x[0],'date':x[1],'gross_profit':x[2],'user_payout':x[3],'profitable_days':x[4],'best_day':x[5],'consistency':x[6],'balance_before':x[7],'floor_before':x[8]})
    for x in b:
        br.append({'mode':mode,'account':x[0],'date':x[1],'balance':x[2],'floor':x[3],'pnl':x[4],'window':x[5]})
pd.DataFrame(rows).to_csv('turbo_200k_08pct_1trade_day_summary.csv',index=False)
pd.DataFrame(pays).to_csv('turbo_200k_08pct_1trade_day_payouts.csv',index=False)
pd.DataFrame(br).to_csv('turbo_200k_08pct_1trade_day_breaches.csv',index=False)
pd.DataFrame(trades_used,columns=['trade_day','window','R','pnl','exit_dt']).to_csv('turbo_200k_08pct_1trade_day_trades.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
print('\nPAYOUTS\n',pd.DataFrame(pays).to_string(index=False))
print('\nBREACHES\n',pd.DataFrame(br).to_string(index=False))
