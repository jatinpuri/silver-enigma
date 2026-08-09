import json
from collections import defaultdict
import numpy as np
import pandas as pd

# Exact Setup B rule reconstruction on current Dukascopy H1 source.
ATR_LEN=14; ADX_LEN=14; SL_ATR=2.25; TP_ATR=3.375; RR=1.5; HOLD=12
START=pd.Timestamp('2012-01-01',tz='UTC'); Y26=pd.Timestamp('2026-01-01',tz='UTC'); END=pd.Timestamp('2026-08-09',tz='UTC')

# Current Turbo funded assumptions.
SIZE=100000.0
TRAIL=0.06*SIZE
DAILY_LIMIT=0.03*SIZE
PROFIT_DAY_MIN=0.005*SIZE
WAIT_DAYS=14
CONS_MAX=0.20
SPLIT=0.80
ACCOUNT_COST=549.0
COMMISSION_RT_PER_LOT=4.0
EXEC_STRESS_PIPS=0.2

RISK_PCTS=[0.005,0.006,0.0075,0.008,0.010,0.0125,0.015,0.0175,0.020,0.0225,0.025,0.027,0.028,0.029]

def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def indicators(df):
    h,l,c=df.high,df.low,df.close
    pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=rma(tr,ATR_LEN)
    up=h.diff(); dn=-l.diff()
    plus_dm=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=df.index)
    minus_dm=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=df.index)
    base=rma(tr,ADX_LEN)
    plus_di=100*rma(plus_dm,ADX_LEN)/base
    minus_di=100*rma(minus_dm,ADX_LEN)/base
    dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di)
    adx=rma(dx,ADX_LEN)
    return atr,adx

def pf(a):
    a=np.asarray(a,float); gp=a[a>0].sum(); gl=-a[a<0].sum()
    return gp/gl if gl>0 else np.nan

arr=json.load(open('eurusd_h1_2011_2026.json'))
df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True)
df['atr'],df['adx']=indicators(df)
df['cdir']=np.where(df.close>df.open,1,np.where(df.close<df.open,-1,0))

local=df.dt.dt.tz_convert('Europe/London')
idxs=np.where((local.dt.hour.to_numpy()==7)&(local.dt.weekday.to_numpy()<=3)&(local.dt.year.to_numpy()>=2012)&(local.dt.year.to_numpy()<=2026))[0]
trades=[]
for i in idxs:
    if i<1: continue
    row=df.iloc[i]; d=int(row.cdir)
    if d==0 or not np.isfinite(row.atr) or not np.isfinite(row.adx) or not np.isfinite(df.iloc[i-1].adx): continue
    if not (row.adx > df.iloc[i-1].adx): continue
    j=i+1
    if j>=len(df) or (df.loc[j,'dt']-row['dt'])>pd.Timedelta(hours=1,minutes=1): continue
    entry=float(row.high if d>0 else row.low)
    if d>0 and float(df.loc[j,'high'])<entry: continue
    if d<0 and float(df.loc[j,'low'])>entry: continue
    stop_dist=SL_ATR*float(row.atr)
    if stop_dist<=0: continue
    stop=entry-d*stop_dist; target=entry+d*TP_ATR*float(row.atr)
    end=min(j+HOLD-1,len(df)-1); rv=None; exit_idx=end; reason='TIME'
    for k in range(j,end+1):
        hi=float(df.loc[k,'high']); lo=float(df.loc[k,'low'])
        hit_sl=(lo<=stop) if d>0 else (hi>=stop)
        hit_tp=(hi>=target) if d>0 else (lo<=target)
        if hit_sl:
            rv=-1.0; exit_idx=k; reason='SL'; break
        if hit_tp:
            rv=RR; exit_idx=k; reason='TP'; break
    if rv is None:
        rv=((float(df.loc[end,'close'])-entry)*d)/stop_dist
    trades.append(dict(signal_dt=row['dt'],entry_dt=df.loc[j,'dt'],exit_dt=df.loc[exit_idx,'dt'],trade_day=row['dt'].date(),R=float(rv),stop_pips=stop_dist/0.0001,reason=reason))
tr=pd.DataFrame(trades).sort_values(['signal_dt','entry_dt']).reset_index(drop=True)
tr=tr[(tr.signal_dt>=START)&(tr.signal_dt<END)].copy()

raw_summary=pd.DataFrame([{
    'trades':len(tr),'win_rate':float((tr.R>0).mean()),'ev_R':float(tr.R.mean()),'PF':float(pf(tr.R)),'total_R':float(tr.R.sum()),
    'avg_stop_pips':float(tr.stop_pips.mean()),'median_stop_pips':float(tr.stop_pips.median()),
    'ytd_2026_trades':int((tr.signal_dt>=Y26).sum()),'ytd_2026_ev_R':float(tr.loc[tr.signal_dt>=Y26,'R'].mean()),
}])
raw_summary.to_csv('setup_b_turbo_1x100k_reconstruction_stats.csv',index=False)

def pnl_after_cost(r, risk_pct, stress=False):
    risk_usd=SIZE*risk_pct
    lots=risk_usd/(float(r.stop_pips)*10.0)
    cost=lots*COMMISSION_RT_PER_LOT
    if stress: cost += lots*10.0*EXEC_STRESS_PIPS
    return float(r.R)*risk_usd-cost, cost, lots

def simulate(z,risk_pct,stress=False):
    # One live funded account at a time. Closed-event approximation to equity rules.
    bal=SIZE; peak=SIZE; floor=SIZE-TRAIL
    account_id=1; account_start=None; cycle_start=None
    daily=defaultdict(float); paid_current=False
    payouts=[]; breaches=[]; first_days=[]
    resolved=0; successful=0; total_cost_trade=0.0; max_lots=0.0
    for _,r in z.sort_values(['signal_dt','entry_dt']).iterrows():
        day=r.trade_day
        if account_start is None: account_start=day
        if cycle_start is None: cycle_start=day
        pnl,cost,lots=pnl_after_cost(r,risk_pct,stress)
        total_cost_trade+=cost; max_lots=max(max_lots,lots)
        bal += pnl; daily[day]+=pnl
        # Official rule is equity-based; H1 source can only check event/close path.
        daily_breach = daily[day] <= -DAILY_LIMIT + 1e-9
        trailing_breach = bal <= floor + 1e-9
        if daily_breach or trailing_breach:
            resolved+=1
            if paid_current: successful+=1
            breaches.append(dict(date=day,account_id=account_id,balance=bal,floor=floor,reason='DAILY' if daily_breach else 'TRAIL'))
            account_id+=1; bal=SIZE; peak=SIZE; floor=SIZE-TRAIL
            account_start=day; cycle_start=day; daily=defaultdict(float); paid_current=False
            continue
        if bal>peak: peak=bal
        floor=min(SIZE,peak-TRAIL)
        if (day-cycle_start).days < WAIT_DAYS: continue
        profit=bal-SIZE
        if profit<=0: continue
        profitable_days=sum(v>=PROFIT_DAY_MIN-1e-9 for v in daily.values())
        best=max([0.0]+list(daily.values()))
        consistency=best/profit if profit>0 else np.inf
        if profitable_days<3 or consistency>CONS_MAX+1e-12: continue
        user=profit*SPLIT
        payouts.append(dict(date=day,account_id=account_id,gross_profit=profit,user_payout=user,profitable_days=profitable_days,consistency=consistency))
        if not paid_current:
            paid_current=True; first_days.append((day-account_start).days)
        # Withdraw all profit. After first payout, max DD is locked at initial balance.
        bal=SIZE; peak=SIZE+TRAIL; floor=SIZE
        cycle_start=day; daily=defaultdict(float)
    # Treat active paid account as successful resolved observation; active unpaid is unresolved.
    if paid_current:
        resolved+=1; successful+=1
    pdates=[pd.Timestamp(x['date']) for x in payouts]
    gaps=[(pdates[i]-pdates[i-1]).days for i in range(1,len(pdates))]
    accounts_used=account_id
    user_paid=sum(x['user_payout'] for x in payouts)
    costs=accounts_used*ACCOUNT_COST
    years=(pd.Timestamp(z.signal_dt.max())-pd.Timestamp(z.signal_dt.min())).days/365.25 if len(z)>1 else np.nan
    return dict(
        risk_pct=risk_pct,trades=len(z),payouts=len(payouts),breaches=len(breaches),accounts_used=accounts_used,
        resolved_accounts=resolved,successful_accounts=successful,payout_before_breach_rate=(successful/resolved if resolved else np.nan),
        user_payouts=user_paid,activation_costs=costs,net_cash=user_paid-costs,avg_payout=(user_paid/len(payouts) if payouts else np.nan),
        avg_first_payout_days=(float(np.mean(first_days)) if first_days else np.nan),median_first_payout_days=(float(np.median(first_days)) if first_days else np.nan),
        median_payout_gap_days=(float(np.median(gaps)) if gaps else np.nan),first_payout_date=(str(pdates[0].date()) if pdates else ''),
        total_trading_costs=total_cost_trade,avg_trading_cost_per_trade=(total_cost_trade/len(z) if len(z) else np.nan),max_lots=max_lots,
        net_cash_per_year=((user_paid-costs)/years if years and years>0 else np.nan)
    )

rows=[]
for label,start in [('FULL_2012_2026',START),('YTD_2026',Y26)]:
    z=tr[(tr.signal_dt>=start)&(tr.signal_dt<END)].copy()
    for cost_label,stress in [('COMMISSION',False),('COMM_PLUS_0P2',True)]:
        for rp in RISK_PCTS:
            rows.append(dict(period=label,cost_model=cost_label,**simulate(z,rp,stress)))
out=pd.DataFrame(rows)
out.to_csv('setup_b_turbo_1x100k_risk_sweep_results.csv',index=False)
print('RECONSTRUCTION')
print(raw_summary.to_string(index=False))
print('\nFULL COMMISSION')
print(out[(out.period=='FULL_2012_2026')&(out.cost_model=='COMMISSION')].to_string(index=False))
print('\n2026 COMMISSION')
print(out[(out.period=='YTD_2026')&(out.cost_model=='COMMISSION')].to_string(index=False))
