import json
from collections import defaultdict
import numpy as np
import pandas as pd

ATR_LEN=14; SL_ATR=2.25; RR=0.25; TP_ATR=SL_ATR*RR; HOLD=12
RISK_PCT=0.027; SPLIT=0.80; COMM_RT=4.0; EXEC_PIPS=0.2
WAIT_DAYS=14; CONS_MAX=0.20
FULL_START=pd.Timestamp('2012-01-01',tz='UTC'); Y26_START=pd.Timestamp('2026-01-01',tz='UTC'); END=pd.Timestamp('2026-08-09',tz='UTC')
SLOTS={
 'LON_OPEN':('Europe/London',7),
 'LON_H1':('Europe/London',8),
 'NY_OPEN':('America/New_York',7),
 'NY_H1':('America/New_York',8),
 'NY_H2':('America/New_York',9),
}

def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def atr14(df):
    pc=df.close.shift(1)
    tr=pd.concat([(df.high-df.low).abs(),(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    return rma(tr,ATR_LEN)

arr=json.load(open('eurusd_h1_2011_2026.json'))
df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True)
df['atr']=atr14(df)
df['cdir']=np.where(df.close>df.open,1,np.where(df.close<df.open,-1,0))

trades=[]
for slot,(tz,sig_hour) in SLOTS.items():
    local=df.dt.dt.tz_convert(tz)
    idxs=np.where((local.dt.hour.to_numpy()==sig_hour)&(local.dt.weekday.to_numpy()<=4)&(local.dt.year.to_numpy()>=2012)&(local.dt.year.to_numpy()<=2026))[0]
    for i in idxs:
        row=df.iloc[i]; d=int(row.cdir)
        if d==0 or not np.isfinite(row.atr): continue
        j=i+1
        if j>=len(df) or (df.loc[j,'dt']-row.dt)>pd.Timedelta(hours=1,minutes=1): continue
        entry=float(row.high if d>0 else row.low)
        if d>0 and float(df.loc[j,'high'])<entry: continue
        if d<0 and float(df.loc[j,'low'])>entry: continue
        stop_dist=SL_ATR*float(row.atr)
        if stop_dist<=0: continue
        stop=entry-d*stop_dist; target=entry+d*TP_ATR*float(row.atr)
        end=min(j+HOLD-1,len(df)-1); rv=None; exit_idx=end
        for k in range(j,end+1):
            hi=float(df.loc[k,'high']); lo=float(df.loc[k,'low'])
            hit_sl=(lo<=stop) if d>0 else (hi>=stop)
            hit_tp=(hi>=target) if d>0 else (lo<=target)
            if hit_sl: rv=-1.0; exit_idx=k; break
            if hit_tp: rv=RR; exit_idx=k; break
        if rv is None: rv=((float(df.loc[end,'close'])-entry)*d)/stop_dist
        trades.append(dict(slot=slot,signal_dt=row.dt,entry_bar_dt=df.loc[j,'dt'],exit_dt=df.loc[exit_idx,'dt'],trade_day=row.dt.date(),raw_R=float(rv),stop_pips=stop_dist/0.0001))
tr=pd.DataFrame(trades).sort_values(['trade_day','entry_bar_dt','signal_dt','slot']).reset_index(drop=True)

def pnl_after_cost(raw_R,stop_pips,size,stress=False):
    risk=size*RISK_PCT
    lots=risk/(stop_pips*10.0)
    cost=lots*COMM_RT + (lots*10.0*EXEC_PIPS if stress else 0.0)
    return raw_R*risk-cost

def new_state(size,day=None):
    trail=.06*size
    return dict(size=size,bal=size,peak=size,floor=size-trail,cycle_start=day,account_start=day,daily=defaultdict(float),paid_current=False,payouts=0,breaches=0,accounts=1,user=0.0)

def reset_state(s,day):
    size=s['size']; trail=.06*size
    s.update(bal=size,peak=size,floor=size-trail,cycle_start=day,account_start=day,daily=defaultdict(float),paid_current=False)
    s['breaches']+=1; s['accounts']+=1

def update_floor(s):
    size=s['size']; trail=.06*size
    if s['bal']>s['peak']: s['peak']=s['bal']
    s['floor']=min(size,s['peak']-trail)

def maybe_payout(s,day,paydates,first_days):
    size=s['size']
    if (day-s['cycle_start']).days<WAIT_DAYS: return
    profit=s['bal']-size
    if profit<=0: return
    pdays=sum(v>=.005*size-1e-9 for v in s['daily'].values())
    best=max([0.0]+list(s['daily'].values()))
    if pdays<3 or best/profit>CONS_MAX+1e-12: return
    user=profit*SPLIT; s['user']+=user; s['payouts']+=1; paydates.append(pd.Timestamp(day))
    if not s['paid_current']:
        s['paid_current']=True; first_days.append((day-s['account_start']).days)
    s['bal']=size; s['peak']=size+.06*size; s['floor']=size; s['cycle_start']=day; s['daily']=defaultdict(float)

def summarize(states,paydates,first_days,resolved_total,resolved_success,start):
    for s in states:
        if s['paid_current']:
            resolved_total+=1; resolved_success+=1
    paydates=sorted(paydates); gaps=[(paydates[i]-paydates[i-1]).days for i in range(1,len(paydates))]
    payouts=sum(s['payouts'] for s in states); breaches=sum(s['breaches'] for s in states); accounts=sum(s['accounts'] for s in states); user=sum(s['user'] for s in states)
    return dict(payouts=payouts,breaches=breaches,accounts_used=accounts,user_paid=user,avg_payout=(user/payouts if payouts else np.nan),payout_before_breach_rate=(resolved_success/resolved_total if resolved_total else np.nan),resolved_accounts=resolved_total,successful_accounts=resolved_success,avg_first_payout_days=(float(np.mean(first_days)) if first_days else np.nan),median_first_payout_days=(float(np.median(first_days)) if first_days else np.nan),first_portfolio_payout_date=(str(paydates[0].date()) if paydates else ''),first_portfolio_payout_days=((paydates[0]-pd.Timestamp(start.date())).days if paydates else np.nan),median_gap_days=(float(np.median(gaps)) if gaps else np.nan))

def process_trade(s,r,day,stress,paydates,first_days):
    if s['account_start'] is None: s['account_start']=day
    if s['cycle_start'] is None: s['cycle_start']=day
    pnl=pnl_after_cost(float(r.raw_R),float(r.stop_pips),s['size'],stress)
    s['bal']+=pnl; s['daily'][day]+=pnl
    daily_breach=s['daily'][day] <= -.03*s['size']+1e-9
    trail_breach=s['bal'] <= s['floor']+1e-9
    if daily_breach or trail_breach: return True
    update_floor(s); maybe_payout(s,day,paydates,first_days); return False

def sim_5x20(start,stress=False):
    z=tr[(tr.signal_dt>=start)&(tr.signal_dt<END)].copy()
    states={slot:new_state(20000.0) for slot in SLOTS}
    paydates=[]; first_days=[]; resolved_total=0; resolved_success=0; used=0
    for day,dg in z.groupby('trade_day',sort=True):
        for _,r in dg.sort_values(['entry_bar_dt','signal_dt','slot']).iterrows():
            s=states[r.slot]; used+=1
            if process_trade(s,r,day,stress,paydates,first_days):
                resolved_total+=1
                if s['paid_current']: resolved_success+=1
                reset_state(s,day)
    out=summarize(list(states.values()),paydates,first_days,resolved_total,resolved_success,start)
    out['trades_used']=used
    return out

def sim_1x100_earliest(start,stress=False):
    z=tr[(tr.signal_dt>=start)&(tr.signal_dt<END)].copy()
    s=new_state(100000.0); paydates=[]; first_days=[]; resolved_total=0; resolved_success=0; used=0; slot_used=defaultdict(int)
    for day,dg in z.groupby('trade_day',sort=True):
        r=dg.sort_values(['entry_bar_dt','signal_dt','slot']).iloc[0]
        used+=1; slot_used[r.slot]+=1
        if process_trade(s,r,day,stress,paydates,first_days):
            resolved_total+=1
            if s['paid_current']: resolved_success+=1
            reset_state(s,day)
    out=summarize([s],paydates,first_days,resolved_total,resolved_success,start); out['trades_used']=used
    out['slot_mix']=';'.join(f'{k}:{slot_used[k]}' for k in SLOTS)
    return out

def sim_1x100_h1(start,stress=False):
    z=tr[(tr.slot=='LON_H1')&(tr.signal_dt>=start)&(tr.signal_dt<END)].copy()
    s=new_state(100000.0); paydates=[]; first_days=[]; resolved_total=0; resolved_success=0; used=0
    for _,r in z.sort_values(['trade_day','entry_bar_dt']).iterrows():
        day=r.trade_day; used+=1
        if process_trade(s,r,day,stress,paydates,first_days):
            resolved_total+=1
            if s['paid_current']: resolved_success+=1
            reset_state(s,day)
    out=summarize([s],paydates,first_days,resolved_total,resolved_success,start); out['trades_used']=used
    return out

rows=[]
for label,start in [('FULL_2012_2026',FULL_START),('YTD_2026',Y26_START)]:
    for costlabel,stress in [('COMMISSION',False),('COMM_PLUS_0P2',True)]:
        for method,fn in [('5x20k_5_FIXED_SLOTS',sim_5x20),('1x100k_EARLIEST_OF_5',sim_1x100_earliest),('1x100k_LON_H1_ONLY',sim_1x100_h1)]:
            sm=fn(start,stress); rows.append(dict(period=label,cost_model=costlabel,method=method,**sm))
out=pd.DataFrame(rows)
# Official current 100k successful activation = $549. 20k is not currently listed; leave its fee unknown.
out['official_activation_cost_per_account']=np.where(out.method.str.startswith('1x100k'),549.0,np.nan)
out['known_account_costs']=np.where(out.method.str.startswith('1x100k'),out.accounts_used*549.0,np.nan)
out['known_net_cash']=np.where(out.method.str.startswith('1x100k'),out.user_paid-out.known_account_costs,np.nan)
# Max hypothetical 20k activation fee per account such that 5x20k beats 1x100k earliest on net cash.
for (period,cost_model),g in out.groupby(['period','cost_model']):
    base=g[g.method=='1x100k_EARLIEST_OF_5'].iloc[0]
    idx=out[(out.period==period)&(out.cost_model==cost_model)&(out.method=='5x20k_5_FIXED_SLOTS')].index[0]
    out.loc[idx,'break_even_20k_fee_vs_1x100k_earliest']=(out.loc[idx,'user_paid']-base['known_net_cash'])/out.loc[idx,'accounts_used']
out.to_csv('turbo_5x20k_vs_1x100k_five_slots_025r_results.csv',index=False)
print(out.to_string(index=False))
