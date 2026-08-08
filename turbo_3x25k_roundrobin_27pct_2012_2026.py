import pandas as pd
import numpy as np
from collections import defaultdict

START=25000.0
RISK_PCT=0.027
RISK=START*RISK_PCT
TRAIL=START*0.06
PROFIT_DAY_MIN=START*0.005
CONSISTENCY_MAX=0.20
WAIT_DAYS=14
SPLIT=0.80
ACCOUNT_COST=189.0
WINDOWS={'LON_OPEN','LON_H1','NY_H1'}
START_DATE=pd.Timestamp('2012-01-01',tz='UTC')
END_DATE=pd.Timestamp('2026-08-09',tz='UTC')

tr=pd.read_csv('eurusd_multihour_trades.csv')
for c in ['signal_dt','entry_bar_dt','exit_dt']:
    tr[c]=pd.to_datetime(tr[c],utc=True)
tr=tr[(tr.signal_dt>=START_DATE)&(tr.signal_dt<END_DATE)&tr.window.isin(WINDOWS)&(tr.local_weekday<=4)].copy()
tr['trade_day']=tr.signal_dt.dt.date
tr=tr.sort_values(['trade_day','entry_bar_dt','signal_dt','window']).reset_index(drop=True)

# Round-robin every triggered trade across 3 simultaneous slots. With max 3 windows/day,
# each slot receives at most one trade per trading day.
tr['slot']=[i%3 for i in range(len(tr))]

class Slot:
    def __init__(self, slot):
        self.slot=slot
        self.balance=START
        self.floor=START-TRAIL
        self.account_no=1
        self.accounts_used=1
        self.breaches=0
        self.payouts=[]
        self.payout_dates=[]
        self.daily=defaultdict(float)
        self.cycle_start=None
        self.account_start=None
        self.first_payout_recorded=False
        self.first_payout_days=[]
        self.trade_count=0
    def replace(self, d):
        self.balance=START
        self.floor=START-TRAIL
        self.account_no+=1
        self.accounts_used+=1
        self.daily=defaultdict(float)
        self.cycle_start=d
        self.account_start=d
        self.first_payout_recorded=False
    def process(self, r):
        d=r.trade_day
        if self.account_start is None: self.account_start=d
        if self.cycle_start is None: self.cycle_start=d
        self.trade_count+=1
        pnl=float(r.R)*RISK
        self.balance += pnl
        self.daily[d] += pnl
        if self.balance < self.floor-1e-9:
            self.breaches += 1
            self.replace(d)
            return None
        if (d-self.cycle_start).days < WAIT_DAYS:
            return None
        profit=self.balance-START
        if profit<=0:
            return None
        pdays=sum(v>=PROFIT_DAY_MIN-1e-9 for v in self.daily.values())
        best=max([0.0]+list(self.daily.values()))
        cons=best/profit
        if pdays<3 or cons>CONSISTENCY_MAX+1e-12:
            return None
        gross=self.balance-START
        user=gross*SPLIT
        rec={'slot':self.slot+1,'account_no':self.account_no,'date':d,'gross':gross,'user_payout':user,
             'profitable_days':pdays,'best_day':best,'consistency':cons,'balance_before':self.balance}
        self.payouts.append(rec)
        self.payout_dates.append(d)
        if not self.first_payout_recorded:
            self.first_payout_days.append((d-self.account_start).days)
            self.first_payout_recorded=True
        self.balance=START
        self.floor=START
        self.cycle_start=d
        self.daily=defaultdict(float)
        return rec

slots=[Slot(i) for i in range(3)]
all_payouts=[]
for _,r in tr.iterrows():
    rec=slots[int(r.slot)].process(r)
    if rec: all_payouts.append(rec)

# baseline: one $25k slot, earliest triggered trade only each day
base=tr.sort_values(['trade_day','entry_bar_dt','signal_dt','window']).groupby('trade_day',as_index=False,sort=True).first()
base=base.sort_values(['trade_day','entry_bar_dt']).reset_index(drop=True)
b=Slot(99)
base_payouts=[]
for _,r in base.iterrows():
    rec=b.process(r)
    if rec: base_payouts.append(rec)

def payout_gap_stats(recs):
    ds=sorted([pd.Timestamp(x['date']) for x in recs])
    if len(ds)<2: return (np.nan,np.nan)
    gaps=[(ds[i]-ds[i-1]).days for i in range(1,len(ds))]
    return float(np.mean(gaps)),float(np.median(gaps))

portfolio_gap_avg,portfolio_gap_med=payout_gap_stats(all_payouts)
base_gap_avg,base_gap_med=payout_gap_stats(base_payouts)
portfolio_user=sum(x['user_payout'] for x in all_payouts)
portfolio_accounts=sum(s.accounts_used for s in slots)
portfolio_cost=portfolio_accounts*ACCOUNT_COST
base_user=sum(x['user_payout'] for x in base_payouts)
base_cost=b.accounts_used*ACCOUNT_COST

summary=[{
    'setup':'3x25k_roundrobin_all_triggered',
    'source_trades_used':len(tr),
    'payouts':len(all_payouts),
    'breaches':sum(s.breaches for s in slots),
    'accounts_used':portfolio_accounts,
    'user_payout_usd':portfolio_user,
    'account_cost_usd':portfolio_cost,
    'net_cash_usd':portfolio_user-portfolio_cost,
    'avg_days_between_portfolio_payouts':portfolio_gap_avg,
    'median_days_between_portfolio_payouts':portfolio_gap_med,
    'first_portfolio_payout_date':min([x['date'] for x in all_payouts]) if all_payouts else None,
    'first_portfolio_payout_days_from_start':(pd.Timestamp(min([x['date'] for x in all_payouts]))-pd.Timestamp(START_DATE.date())).days if all_payouts else np.nan
},{
    'setup':'1x25k_first_trade_per_day',
    'source_trades_used':len(base),
    'payouts':len(base_payouts),
    'breaches':b.breaches,
    'accounts_used':b.accounts_used,
    'user_payout_usd':base_user,
    'account_cost_usd':base_cost,
    'net_cash_usd':base_user-base_cost,
    'avg_days_between_portfolio_payouts':base_gap_avg,
    'median_days_between_portfolio_payouts':base_gap_med,
    'first_portfolio_payout_date':min([x['date'] for x in base_payouts]) if base_payouts else None,
    'first_portfolio_payout_days_from_start':(pd.Timestamp(min([x['date'] for x in base_payouts]))-pd.Timestamp(START_DATE.date())).days if base_payouts else np.nan
}]

slotrows=[]
for s in slots:
    avgfp=np.mean(s.first_payout_days) if s.first_payout_days else np.nan
    medfp=np.median(s.first_payout_days) if s.first_payout_days else np.nan
    gapavg,gapmed=payout_gap_stats(s.payouts)
    slotrows.append({'slot':s.slot+1,'trades':s.trade_count,'payouts':len(s.payouts),'breaches':s.breaches,'accounts_used':s.accounts_used,
                     'user_payout_usd':sum(x['user_payout'] for x in s.payouts),'avg_first_payout_days_successful_accounts':avgfp,
                     'median_first_payout_days_successful_accounts':medfp,'avg_days_between_slot_payouts':gapavg,'median_days_between_slot_payouts':gapmed})

pd.DataFrame(summary).to_csv('turbo_3x25k_roundrobin_27pct_summary.csv',index=False)
pd.DataFrame(slotrows).to_csv('turbo_3x25k_roundrobin_27pct_slots.csv',index=False)
pd.DataFrame(all_payouts).to_csv('turbo_3x25k_roundrobin_27pct_payouts.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
print(pd.DataFrame(slotrows).to_string(index=False))
