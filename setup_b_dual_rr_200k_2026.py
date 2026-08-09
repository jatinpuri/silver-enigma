import json
from collections import defaultdict
import numpy as np
import pandas as pd

ATR_LEN=14; ADX_LEN=14; SL_ATR=2.25; HOLD=12
START=pd.Timestamp('2026-01-01',tz='UTC'); END=pd.Timestamp('2026-08-09',tz='UTC')
SIZE=200000.0; TRAIL=0.06*SIZE; DAILY_LIMIT=0.03*SIZE
PROFIT_DAY_MIN=0.005*SIZE; WAIT_DAYS=14; CONS_MAX=0.20; SPLIT=0.80
ACCOUNT_COST=1098.0; COMMISSION_RT_PER_LOT=4.0; EXEC_STRESS_PIPS=0.2
RISK_PCTS=[0.005,0.006,0.0075,0.008,0.01,0.0125,0.015,0.0175,0.02,0.0225,0.025,0.027,0.028,0.029]

def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def indicators(df):
    h,l,c=df.high,df.low,df.close; pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=rma(tr,ATR_LEN)
    up=h.diff(); dn=-l.diff()
    plus_dm=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=df.index)
    minus_dm=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=df.index)
    base=rma(tr,ADX_LEN)
    pdi=100*rma(plus_dm,ADX_LEN)/base; mdi=100*rma(minus_dm,ADX_LEN)/base
    dx=100*(pdi-mdi).abs()/(pdi+mdi); adx=rma(dx,ADX_LEN)
    return atr,adx

def pf(a):
    a=np.asarray(a,float); gp=a[a>0].sum(); gl=-a[a<0].sum()
    return gp/gl if gl>0 else np.nan

def maxdd(a):
    a=np.asarray(a,float); eq=np.r_[0,np.cumsum(a)]; peak=np.maximum.accumulate(eq)
    return float(np.max(peak-eq))

arr=json.load(open('eurusd_h1_2011_2026.json'))
df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True)
df['atr'],df['adx']=indicators(df)
df['cdir']=np.where(df.close>df.open,1,np.where(df.close<df.open,-1,0))
local=df.dt.dt.tz_convert('Europe/London')
idxs=np.where((local.dt.hour.to_numpy()==7)&(local.dt.weekday.to_numpy()<=3)&(local.dt.year.to_numpy()==2026))[0]

trades=[]
for i in idxs:
    if i<1: continue
    r=df.iloc[i]; p=df.iloc[i-1]; c=int(r.cdir)
    if c==0 or not np.isfinite(r.atr) or not np.isfinite(r.adx) or not np.isfinite(p.adx): continue
    if r.adx>p.adx:
        d=c; regime='RISE_1P5R'; rr=1.5
    elif r.adx<p.adx:
        d=-c; regime='FALL_REVERSE_0P25R'; rr=0.25
    else: continue
    j=i+1
    if j>=len(df) or (df.loc[j,'dt']-r['dt'])>pd.Timedelta(hours=1,minutes=1): continue
    entry=float(r.high if d>0 else r.low)
    if d>0 and float(df.loc[j,'high'])<entry: continue
    if d<0 and float(df.loc[j,'low'])>entry: continue
    stop_dist=SL_ATR*float(r.atr); stop_pips=stop_dist/0.0001
    stop=entry-d*stop_dist; target=entry+d*(SL_ATR*rr)*float(r.atr)
    end=min(j+HOLD-1,len(df)-1); rv=None; exit_idx=end; reason='TIME'
    for k in range(j,end+1):
        hi=float(df.loc[k,'high']); lo=float(df.loc[k,'low'])
        hit_sl=(lo<=stop) if d>0 else (hi>=stop)
        hit_tp=(hi>=target) if d>0 else (lo<=target)
        if hit_sl: rv=-1.0; exit_idx=k; reason='SL'; break
        if hit_tp: rv=rr; exit_idx=k; reason='TP'; break
    if rv is None:
        rv=((float(df.loc[end,'close'])-entry)*d)/stop_dist
    trades.append(dict(signal_dt=r['dt'],entry_dt=df.loc[j,'dt'],exit_dt=df.loc[exit_idx,'dt'],trade_day=r['dt'].date(),R=float(rv),stop_pips=stop_pips,regime=regime,rr=rr,reason=reason))

tr=pd.DataFrame(trades).sort_values(['signal_dt','entry_dt']).reset_index(drop=True)
tr=tr[(tr.signal_dt>=START)&(tr.signal_dt<END)].copy()
tr.to_csv('setup_b_dual_rr_2026_trades.csv',index=False)

def pnl_after_cost(row,risk_pct,stress=False):
    risk=SIZE*risk_pct; lots=risk/(float(row.stop_pips)*10.0)
    cost=lots*COMMISSION_RT_PER_LOT
    if stress: cost+=lots*10.0*EXEC_STRESS_PIPS
    return float(row.R)*risk-cost,cost,lots

def simulate(risk_pct,stress=False):
    bal=SIZE; peak=SIZE; floor=SIZE-TRAIL
    account_id=1; account_start=None; cycle_start=None; daily=defaultdict(float); paid_current=False
    payouts=[]; breaches=[]; first_days=[]; resolved=0; successful=0; trade_costs=0.0; max_lots=0.0
    for _,r in tr.iterrows():
        day=r.trade_day
        if account_start is None: account_start=day
        if cycle_start is None: cycle_start=day
        pnl,cost,lots=pnl_after_cost(r,risk_pct,stress); trade_costs+=cost; max_lots=max(max_lots,lots)
        bal+=pnl; daily[day]+=pnl
        daily_breach=daily[day] <= -DAILY_LIMIT + 1e-9
        trailing_breach=bal <= floor + 1e-9
        if daily_breach or trailing_breach:
            resolved+=1
            if paid_current: successful+=1
            breaches.append((day,account_id,'DAILY' if daily_breach else 'TRAIL'))
            account_id+=1; bal=SIZE; peak=SIZE; floor=SIZE-TRAIL
            account_start=day; cycle_start=day; daily=defaultdict(float); paid_current=False
            continue
        if bal>peak: peak=bal
        floor=min(SIZE,peak-TRAIL)
        if (day-cycle_start).days<WAIT_DAYS: continue
        profit=bal-SIZE
        if profit<=0: continue
        profdays=sum(v>=PROFIT_DAY_MIN-1e-9 for v in daily.values())
        best=max([0.0]+list(daily.values())); cons=best/profit if profit>0 else np.inf
        if profdays<3 or cons>CONS_MAX+1e-12: continue
        user=profit*SPLIT
        payouts.append((day,account_id,user,profit,profdays,cons))
        if not paid_current:
            paid_current=True; first_days.append((day-account_start).days)
        bal=SIZE; peak=SIZE+TRAIL; floor=SIZE; cycle_start=day; daily=defaultdict(float)
    if paid_current: resolved+=1; successful+=1
    pdates=[pd.Timestamp(x[0]) for x in payouts]; gaps=[(pdates[i]-pdates[i-1]).days for i in range(1,len(pdates))]
    paid=sum(x[2] for x in payouts); acct_cost=account_id*ACCOUNT_COST
    return dict(risk_pct=risk_pct,trades=len(tr),payouts=len(payouts),breaches=len(breaches),accounts_used=account_id,
                resolved_accounts=resolved,successful_accounts=successful,payout_before_breach_rate=(successful/resolved if resolved else np.nan),
                user_payouts=paid,activation_costs=acct_cost,net_cash=paid-acct_cost,avg_payout=(paid/len(payouts) if payouts else np.nan),
                avg_first_payout_days=(np.mean(first_days) if first_days else np.nan),median_first_payout_days=(np.median(first_days) if first_days else np.nan),
                median_payout_gap_days=(np.median(gaps) if gaps else np.nan),first_payout_date=(str(pdates[0].date()) if pdates else ''),
                total_trading_costs=trade_costs,avg_trading_cost_per_trade=(trade_costs/len(tr) if len(tr) else np.nan),max_lots=max_lots)

rows=[]
for label,stress in [('COMMISSION',False),('COMM_PLUS_0P2',True)]:
    for rp in RISK_PCTS: rows.append(dict(cost_model=label,**simulate(rp,stress)))
out=pd.DataFrame(rows); out.to_csv('setup_b_dual_rr_200k_2026_results.csv',index=False)

sumrows=[]
for regime,z in tr.groupby('regime'):
    sumrows.append(dict(regime=regime,trades=len(z),win_rate=float((z.R>0).mean()),PF=float(pf(z.R)),EV_R=float(z.R.mean()),total_R=float(z.R.sum()),max_DD_R=maxdd(z.R.to_numpy())))
sumrows.append(dict(regime='COMBINED',trades=len(tr),win_rate=float((tr.R>0).mean()),PF=float(pf(tr.R)),EV_R=float(tr.R.mean()),total_R=float(tr.R.sum()),max_DD_R=maxdd(tr.R.to_numpy())))
summary=pd.DataFrame(sumrows); summary.to_csv('setup_b_dual_rr_2026_summary.csv',index=False)
print(summary.to_string(index=False)); print(out.to_string(index=False))
