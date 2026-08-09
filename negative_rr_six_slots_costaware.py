import json
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd

ATR_LEN=14
SL_ATR=2.25
RR=0.25
TP_ATR=SL_ATR*RR
HOLD=12
SIZE=100000.0
RISK_PCT=0.027
RISK_USD=SIZE*RISK_PCT
SPLIT=0.80
COST_ACCOUNT=549.0
COMMISSION_PER_LOT_RT=4.0
EXEC_PIPS_STRESS=0.2
TRAIL_USD=0.06*SIZE
DAILY_DD_USD=0.03*SIZE
PROF_DAY_MIN=0.005*SIZE
WAIT_DAYS=14
CONS_MAX=0.20
FULL_START=pd.Timestamp('2012-01-01',tz='UTC')
Y26_START=pd.Timestamp('2026-01-01',tz='UTC')
END=pd.Timestamp('2026-08-09',tz='UTC')

# Signal candle hour -> decision/entry begins one hour later.
SLOTS={
    'LON_OPEN':('Europe/London',7),   # 07-08 signal, decide 08:00 London
    'LON_H1':('Europe/London',8),     # 08-09 signal, decide 09:00 London
    'LON_H2':('Europe/London',9),     # 09-10 signal, decide 10:00 London
    'NY_OPEN':('America/New_York',7), # 07-08 signal, decide 08:00 NY
    'NY_H1':('America/New_York',8),   # 08-09 signal, decide 09:00 NY
    'NY_H2':('America/New_York',9),   # 09-10 signal, decide 10:00 NY
}

def rma(s,n):
    return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def atr14(df):
    h,l,c=df.high,df.low,df.close
    pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return rma(tr,ATR_LEN)

def pf(a):
    a=np.asarray(a,float)
    gp=a[a>0].sum(); gl=-a[a<0].sum()
    return gp/gl if gl>0 else (np.inf if gp>0 else np.nan)

arr=json.load(open('eurusd_h1_2011_2026.json'))
df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True)
df['atr']=atr14(df)
df['cdir']=np.where(df.close>df.open,1,np.where(df.close<df.open,-1,0))

all_trades=[]
for slot,(tz,sig_hour) in SLOTS.items():
    local=df.dt.dt.tz_convert(tz)
    lh=local.dt.hour.to_numpy(); wd=local.dt.weekday.to_numpy(); yr=local.dt.year.to_numpy()
    idxs=np.where((lh==sig_hour)&(wd<=4)&(yr>=2012)&(yr<=2026))[0]
    for i in idxs:
        row=df.iloc[i]
        d=int(row['cdir'])
        if d==0 or not np.isfinite(row['atr']):
            continue
        j=i+1
        if j>=len(df) or (df.loc[j,'dt']-row['dt'])>pd.Timedelta(hours=1,minutes=1):
            continue
        entry=float(row['high'] if d>0 else row['low'])
        # Pending order valid only in next H1 bar.
        if d>0 and float(df.loc[j,'high'])<entry: continue
        if d<0 and float(df.loc[j,'low'])>entry: continue
        stop_dist=SL_ATR*float(row['atr'])
        if stop_dist<=0: continue
        stop=entry-d*stop_dist
        target=entry+d*TP_ATR*float(row['atr'])
        end=min(j+HOLD-1,len(df)-1)
        rv=None; exit_idx=end; reason='TIME'
        for k in range(j,end+1):
            hi=float(df.loc[k,'high']); lo=float(df.loc[k,'low'])
            hit_sl=(lo<=stop) if d>0 else (hi>=stop)
            hit_tp=(hi>=target) if d>0 else (lo<=target)
            # conservative same-bar ambiguity: SL first
            if hit_sl:
                rv=-1.0; exit_idx=k; reason='SL'; break
            if hit_tp:
                rv=RR; exit_idx=k; reason='TP'; break
        if rv is None:
            rv=((float(df.loc[end,'close'])-entry)*d)/stop_dist
        stop_pips=stop_dist/0.0001
        lots=RISK_USD/(stop_pips*10.0)
        comm=lots*COMMISSION_PER_LOT_RT
        exec_cost=lots*10.0*EXEC_PIPS_STRESS
        r_comm=float(rv)-comm/RISK_USD
        r_stress=float(rv)-(comm+exec_cost)/RISK_USD
        all_trades.append(dict(slot=slot,signal_dt=row['dt'],entry_bar_dt=df.loc[j,'dt'],exit_dt=df.loc[exit_idx,'dt'],trade_day=row['dt'].date(),raw_R=float(rv),R_comm=r_comm,R_stress=r_stress,reason=reason,stop_pips=stop_pips,lots=lots,commission=comm,exec02=exec_cost))

tr=pd.DataFrame(all_trades).sort_values(['signal_dt','entry_bar_dt','slot']).reset_index(drop=True)

def market_metrics(z,rcol):
    a=z[rcol].to_numpy(float)
    if len(a)==0:
        return dict(n=0,wr=np.nan,ev=np.nan,pf=np.nan,totalR=0.0)
    return dict(n=len(a),wr=float((z.raw_R>0).mean()),ev=float(a.mean()),pf=float(pf(a)),totalR=float(a.sum()))

def new_state(day=None):
    return dict(bal=SIZE,peak=SIZE,floor=SIZE-TRAIL_USD,cycle_start=day,account_start=day,daily=defaultdict(float),paid_current=False,payouts=0,breaches=0,accounts=1,user=0.0)

def reset_after_breach(s,day):
    s.update(bal=SIZE,peak=SIZE,floor=SIZE-TRAIL_USD,cycle_start=day,account_start=day,daily=defaultdict(float),paid_current=False)
    s['accounts']+=1
    s['breaches']+=1

def update_floor_after_close(s):
    if s['bal']>s['peak']:
        s['peak']=s['bal']
    s['floor']=min(SIZE,s['peak']-TRAIL_USD)

def simulate_pair(slots,start,rcol):
    # Two accounts; at most one trade per account per trading day. All trades from the selected slots are eligible.
    z=tr[(tr.slot.isin(slots))&(tr.signal_dt>=start)&(tr.signal_dt<END)].copy().sort_values(['trade_day','entry_bar_dt','signal_dt','slot'])
    states=[new_state(),new_state()]
    ptr=0; paydates=[]; first_days=[]; resolved_total=0; resolved_success=0; trades_used=0
    slot_used=defaultdict(int)
    for day,dg in z.groupby('trade_day',sort=True):
        used=set()
        for _,r in dg.sort_values(['entry_bar_dt','signal_dt','slot']).iterrows():
            if len(used)>=2: break
            chosen=None
            for _ in range(2):
                idx=ptr%2; ptr=(ptr+1)%2
                if idx not in used:
                    chosen=idx; break
            if chosen is None: break
            used.add(chosen); s=states[chosen]
            if s['account_start'] is None: s['account_start']=day
            if s['cycle_start'] is None: s['cycle_start']=day
            trades_used+=1; slot_used[r.slot]+=1
            pnl=float(r[rcol])*RISK_USD
            s['bal']+=pnl; s['daily'][day]+=pnl
            # Official daily loss is equity based; here we can only check closed-event daily P&L.
            daily_breach=s['daily'][day] <= -DAILY_DD_USD + 1e-9
            trail_breach=s['bal'] <= s['floor'] + 1e-9
            if daily_breach or trail_breach:
                resolved_total+=1
                if s['paid_current']: resolved_success+=1
                reset_after_breach(s,day)
                continue
            update_floor_after_close(s)
            if (day-s['cycle_start']).days<WAIT_DAYS: continue
            profit=s['bal']-SIZE
            if profit<=0: continue
            pdays=sum(v>=PROF_DAY_MIN-1e-9 for v in s['daily'].values())
            best=max([0.0]+list(s['daily'].values()))
            cons=best/profit
            if pdays<3 or cons>CONS_MAX+1e-12: continue
            user=profit*SPLIT
            s['user']+=user; s['payouts']+=1; paydates.append(pd.Timestamp(day))
            if not s['paid_current']:
                s['paid_current']=True; first_days.append((day-s['account_start']).days)
            # payout removes profit; trailing DD locks at initial balance
            s['bal']=SIZE; s['peak']=SIZE+TRAIL_USD; s['floor']=SIZE; s['cycle_start']=day; s['daily']=defaultdict(float)
    for s in states:
        if s['paid_current']:
            resolved_total+=1; resolved_success+=1
    payouts=sum(s['payouts'] for s in states); breaches=sum(s['breaches'] for s in states); accounts=sum(s['accounts'] for s in states); user=sum(s['user'] for s in states)
    paydates=sorted(paydates); gaps=[(paydates[i]-paydates[i-1]).days for i in range(1,len(paydates))]
    return dict(trades_used=trades_used,payouts=payouts,breaches=breaches,accounts_used=accounts,user_paid=user,account_costs=accounts*COST_ACCOUNT,net_cash=user-accounts*COST_ACCOUNT,avg_payout=(user/payouts if payouts else np.nan),payout_before_breach_rate=(resolved_success/resolved_total if resolved_total else np.nan),resolved_accounts=resolved_total,successful_accounts=resolved_success,avg_first_payout_days=(float(np.mean(first_days)) if first_days else np.nan),median_first_payout_days=(float(np.median(first_days)) if first_days else np.nan),first_portfolio_payout_date=(str(paydates[0].date()) if paydates else ''),first_portfolio_payout_days=((paydates[0]-pd.Timestamp(start.date())).days if paydates else np.nan),median_gap_days=(float(np.median(gaps)) if gaps else np.nan),slot1_used=slot_used[slots[0]],slot2_used=(slot_used[slots[1]] if len(slots)>1 else 0))

# Individual slot edge metrics.
slot_rows=[]
for slot in SLOTS:
    for label,start in [('FULL_2012_2026',FULL_START),('YTD_2026',Y26_START)]:
        z=tr[(tr.slot==slot)&(tr.signal_dt>=start)&(tr.signal_dt<END)]
        raw=market_metrics(z,'raw_R'); comm=market_metrics(z,'R_comm'); stress=market_metrics(z,'R_stress')
        slot_rows.append(dict(slot=slot,period=label,trades=raw['n'],wr=raw['wr'],raw_ev_r=raw['ev'],raw_pf=raw['pf'],comm_ev_r=comm['ev'],comm_pf=comm['pf'],stress02_ev_r=stress['ev'],stress02_pf=stress['pf'],avg_stop_pips=float(z.stop_pips.mean()) if len(z) else np.nan,avg_lots=float(z.lots.mean()) if len(z) else np.nan,avg_commission=float(z.commission.mean()) if len(z) else np.nan,comm_ev_usd=comm['ev']*RISK_USD if len(z) else np.nan,stress02_ev_usd=stress['ev']*RISK_USD if len(z) else np.nan))
slot_df=pd.DataFrame(slot_rows)
slot_df.to_csv('negative_rr_six_slots_edge_results.csv',index=False)

# All 15 two-slot combos. Run both commission-only and +0.2pip stress.
combo_rows=[]
for a,b in combinations(SLOTS.keys(),2):
    for label,start in [('FULL_2012_2026',FULL_START),('YTD_2026',Y26_START)]:
        for cost_label,rcol in [('COMMISSION','R_comm'),('COMM_PLUS_0P2','R_stress')]:
            sm=simulate_pair((a,b),start,rcol)
            combo_rows.append(dict(slot1=a,slot2=b,period=label,cost_model=cost_label,**sm))
combo_df=pd.DataFrame(combo_rows)
combo_df.to_csv('negative_rr_six_slots_2account_combo_results.csv',index=False)

print('\nSLOT EDGE FULL SAMPLE - sorted by commission EV')
print(slot_df[slot_df.period=='FULL_2012_2026'].sort_values('comm_ev_r',ascending=False).to_string(index=False))
print('\nSLOT EDGE 2026 - sorted by commission EV')
print(slot_df[slot_df.period=='YTD_2026'].sort_values('comm_ev_r',ascending=False).to_string(index=False))
print('\nTOP TWO-SLOT COMBOS FULL - commission only')
print(combo_df[(combo_df.period=='FULL_2012_2026')&(combo_df.cost_model=='COMMISSION')].sort_values(['net_cash','payouts'],ascending=False).head(15).to_string(index=False))
print('\nTOP TWO-SLOT COMBOS 2026 - commission only')
print(combo_df[(combo_df.period=='YTD_2026')&(combo_df.cost_model=='COMMISSION')].sort_values(['net_cash','payouts'],ascending=False).head(15).to_string(index=False))
print('\nTOP TWO-SLOT COMBOS FULL - commission +0.2 pip')
print(combo_df[(combo_df.period=='FULL_2012_2026')&(combo_df.cost_model=='COMM_PLUS_0P2')].sort_values(['net_cash','payouts'],ascending=False).head(15).to_string(index=False))
