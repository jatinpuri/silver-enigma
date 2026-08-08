import json
from collections import defaultdict
import numpy as np
import pandas as pd

ATR_LEN=14; ADX_LEN=14; SL_ATR=2.25; HOLD=12
RRS=[0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.75,1.00,1.50]
RISK_PCT=0.027; N_ACCTS=2; SIZE=100000.0; SPLIT=0.80; COST=549.0
TRAIL=0.06*SIZE; PROF_DAY_MIN=0.005*SIZE; WAIT_DAYS=14; CONS_MAX=0.20
FULL_START=pd.Timestamp('2012-01-01',tz='UTC'); END=pd.Timestamp('2026-08-09',tz='UTC')
Y26_START=pd.Timestamp('2026-01-01',tz='UTC')
WINDOWS={'LON_OPEN':('Europe/London',7),'LON_H1':('Europe/London',8),'NY_H1':('America/New_York',8)}
STRATS=['CANDLE_ONLY','ADX_RISE','DI_ALIGN','ADX_DI','ADX_RANGE','ADX_DI_RANGE','ADX_DI_DIRECTION']

def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def pf(a):
    a=np.asarray(a,float); gp=a[a>0].sum(); gl=-a[a<0].sum()
    return gp/gl if gl>0 else (np.inf if gp>0 else np.nan)

def indicators(df):
    h,l,c=df.high,df.low,df.close; pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=rma(tr,ATR_LEN)
    up=h.diff(); down=-l.diff()
    plus=pd.Series(np.where((up>down)&(up>0),up,0.0),index=df.index)
    minus=pd.Series(np.where((down>up)&(down>0),down,0.0),index=df.index)
    atr_adx=rma(tr,ADX_LEN)
    pdi=100*rma(plus,ADX_LEN)/atr_adx.replace(0,np.nan)
    mdi=100*rma(minus,ADX_LEN)/atr_adx.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    adx=rma(dx,ADX_LEN)
    return atr,adx,pdi,mdi

arr=json.load(open('eurusd_h1_2011_2026.json'))
df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True)
df['atr'],df['adx'],df['pdi'],df['mdi']=indicators(df)
df['adx_prev']=df.adx.shift(1)
df['range']=df.high-df.low; df['range_prev']=df['range'].shift(1)
df['cdir']=np.where(df.close>df.open,1,np.where(df.close<df.open,-1,0))

locals_cache={}
for w,(tz,hour) in WINDOWS.items():
    loc=df.dt.dt.tz_convert(tz)
    locals_cache[w]=(loc.dt.hour.to_numpy(),loc.dt.weekday.to_numpy(),loc.dt.year.to_numpy())

def direction_and_ok(row,strat):
    cdir=int(row['cdir'])
    adxrise=np.isfinite(row['adx']) and np.isfinite(row['adx_prev']) and row['adx']>row['adx_prev']
    dialign=(np.isfinite(row['pdi']) and np.isfinite(row['mdi']) and cdir!=0 and ((cdir>0 and row['pdi']>row['mdi']) or (cdir<0 and row['mdi']>row['pdi'])))
    rangeexp=np.isfinite(row['range_prev']) and row['range']>row['range_prev']
    if strat=='CANDLE_ONLY': return cdir, cdir!=0
    if strat=='ADX_RISE': return cdir, cdir!=0 and adxrise
    if strat=='DI_ALIGN': return cdir, cdir!=0 and dialign
    if strat=='ADX_DI': return cdir, cdir!=0 and adxrise and dialign
    if strat=='ADX_RANGE': return cdir, cdir!=0 and adxrise and rangeexp
    if strat=='ADX_DI_RANGE': return cdir, cdir!=0 and adxrise and dialign and rangeexp
    if strat=='ADX_DI_DIRECTION':
        if not (adxrise and np.isfinite(row['pdi']) and np.isfinite(row['mdi']) and row['pdi']!=row['mdi']): return 0,False
        return (1 if row['pdi']>row['mdi'] else -1),True
    return 0,False

def build_trades(strat,rr):
    tp_atr=SL_ATR*rr; out=[]
    for w,(tz,sig_hour) in WINDOWS.items():
        lh,wd,yr=locals_cache[w]
        idxs=np.where((lh==sig_hour)&(wd<=4)&(yr>=2012)&(yr<=2026))[0]
        for i in idxs:
            row=df.iloc[i]
            if not np.isfinite(row['atr']): continue
            d,ok=direction_and_ok(row,strat)
            if not ok: continue
            j=i+1
            if j>=len(df) or (df.loc[j,'dt']-row['dt'])>pd.Timedelta(hours=1,minutes=1): continue
            entry=float(row['high'] if d>0 else row['low'])
            if d>0 and float(df.loc[j,'high'])<entry: continue
            if d<0 and float(df.loc[j,'low'])>entry: continue
            risk=SL_ATR*float(row['atr'])
            if risk<=0: continue
            stop=entry-d*risk; target=entry+d*tp_atr*float(row['atr'])
            end=min(j+HOLD-1,len(df)-1); rv=None; exit_idx=end; reason='TIME'
            for k in range(j,end+1):
                hi=float(df.loc[k,'high']); lo=float(df.loc[k,'low'])
                hit_sl=(lo<=stop) if d>0 else (hi>=stop)
                hit_tp=(hi>=target) if d>0 else (lo<=target)
                if hit_sl: rv=-1.0; exit_idx=k; reason='SL'; break
                if hit_tp: rv=rr; exit_idx=k; reason='TP'; break
            if rv is None:
                rv=((float(df.loc[end,'close'])-entry)*d)/risk
            out.append(dict(strategy=strat,rr=rr,window=w,signal_dt=row['dt'],entry_bar_dt=df.loc[j,'dt'],exit_dt=df.loc[exit_idx,'dt'],trade_day=row['dt'].date(),year=int(yr[i]),R=float(rv),reason=reason))
    if not out:
        return pd.DataFrame(columns=['strategy','rr','window','signal_dt','entry_bar_dt','exit_dt','trade_day','year','R','reason'])
    return pd.DataFrame(out).sort_values(['signal_dt','entry_bar_dt','window']).reset_index(drop=True)

def metrics(t):
    if len(t)==0: return dict(n=0,wr=np.nan,ev=np.nan,pf=np.nan,totalR=0)
    a=t.R.to_numpy(float)
    return dict(n=len(t),wr=float((a>0).mean()),ev=float(a.mean()),pf=float(pf(a)),totalR=float(a.sum()))

def simulate(t,start):
    z=t[(t.signal_dt>=start)&(t.signal_dt<END)].copy().sort_values(['trade_day','entry_bar_dt','signal_dt','window'])
    states=[]
    for _ in range(N_ACCTS):
        states.append(dict(bal=SIZE,floor=SIZE-TRAIL,cycle_start=None,account_start=None,daily=defaultdict(float),paid_current=False,first_day=None,payouts=0,breaches=0,accounts=1,user=0.0))
    ptr=0; paydates=[]; resolved_success=0; resolved_total=0; first_days=[]; trades_used=0
    for day,dg in z.groupby('trade_day',sort=True):
        used=set()
        for _,r in dg.sort_values(['entry_bar_dt','signal_dt','window']).iterrows():
            if len(used)>=N_ACCTS: break
            chosen=None
            for _ in range(N_ACCTS):
                idx=ptr%N_ACCTS; ptr=(ptr+1)%N_ACCTS
                if idx not in used: chosen=idx; break
            if chosen is None: break
            used.add(chosen); s=states[chosen]; trades_used+=1
            if s['account_start'] is None: s['account_start']=day
            if s['cycle_start'] is None: s['cycle_start']=day
            pnl=float(r.R)*SIZE*RISK_PCT; s['bal']+=pnl; s['daily'][day]+=pnl
            if s['bal']<s['floor']-1e-9:
                resolved_total+=1
                if s['paid_current']: resolved_success+=1
                s['breaches']+=1; s['accounts']+=1
                s.update(bal=SIZE,floor=SIZE-TRAIL,cycle_start=day,account_start=day,daily=defaultdict(float),paid_current=False,first_day=None)
                continue
            if (day-s['cycle_start']).days<WAIT_DAYS: continue
            profit=s['bal']-SIZE
            if profit<=0: continue
            pdays=sum(v>=PROF_DAY_MIN-1e-9 for v in s['daily'].values())
            best=max([0.0]+list(s['daily'].values()))
            cons=best/profit
            if pdays<3 or cons>CONS_MAX+1e-12: continue
            user=profit*SPLIT; s['user']+=user; s['payouts']+=1; paydates.append(pd.Timestamp(day))
            if not s['paid_current']:
                s['paid_current']=True; s['first_day']=(day-s['account_start']).days; first_days.append(s['first_day'])
            s['bal']=SIZE; s['floor']=SIZE; s['cycle_start']=day; s['daily']=defaultdict(float)
    for s in states:
        if s['paid_current']:
            resolved_total+=1; resolved_success+=1
    paydates=sorted(paydates)
    gaps=[(paydates[i]-paydates[i-1]).days for i in range(1,len(paydates))]
    payouts=sum(s['payouts'] for s in states); breaches=sum(s['breaches'] for s in states); accounts=sum(s['accounts'] for s in states); user=sum(s['user'] for s in states)
    return dict(trades_used=trades_used,payouts=payouts,breaches=breaches,accounts_used=accounts,user_paid=user,net_cash=user-accounts*COST,payout_before_breach_rate=(resolved_success/resolved_total if resolved_total else np.nan),resolved_accounts=resolved_total,successful_accounts=resolved_success,avg_first_payout_days=(np.mean(first_days) if first_days else np.nan),median_first_payout_days=(np.median(first_days) if first_days else np.nan),first_portfolio_payout_date=(paydates[0].date() if paydates else None),first_portfolio_payout_days=((paydates[0]-start.normalize()).days if paydates else np.nan),median_gap_days=(np.median(gaps) if gaps else np.nan))

rows=[]
for strat in STRATS:
    for rr in RRS:
        t=build_trades(strat,rr)
        fullm=metrics(t[(t.signal_dt>=FULL_START)&(t.signal_dt<END)])
        y26m=metrics(t[(t.signal_dt>=Y26_START)&(t.signal_dt<END)])
        sf=simulate(t,FULL_START); sy=simulate(t,Y26_START)
        row=dict(strategy=strat,rr=rr,max_tp_day_pct=rr*RISK_PCT*100,profitable_day_possible=(rr*RISK_PCT>=0.005))
        row.update({f'full_{k}':v for k,v in fullm.items()}); row.update({f'y26_{k}':v for k,v in y26m.items()})
        row.update({f'fullsim_{k}':v for k,v in sf.items()}); row.update({f'y26sim_{k}':v for k,v in sy.items()})
        rows.append(row)
        print(strat,rr,'PF',fullm['pf'],'EV',fullm['ev'],'success',sf['payout_before_breach_rate'],'med',sf['median_first_payout_days'],'2026 pays',sy['payouts'])

out=pd.DataFrame(rows)
out.to_csv('negative_rr_strategy_sweep_2x100k_2012_2026.csv',index=False)
neg=out[(out.rr<1)&out.profitable_day_possible].copy()
neg['score']=neg['fullsim_payout_before_breach_rate'].fillna(0)*100 + neg['fullsim_payouts']*0.5 + neg['full_pf'].clip(upper=3)*10 - neg['fullsim_median_first_payout_days'].fillna(999)/10 + neg['y26sim_payouts']*5
neg=neg.sort_values(['score','fullsim_payout_before_breach_rate','fullsim_payouts'],ascending=False)
neg.head(25).to_csv('negative_rr_strategy_sweep_top25.csv',index=False)
print('\nTOP PRACTICAL NEGATIVE RR')
print(neg.head(25)[['strategy','rr','full_n','full_wr','full_ev','full_pf','fullsim_payout_before_breach_rate','fullsim_median_first_payout_days','fullsim_payouts','fullsim_breaches','fullsim_net_cash','y26_n','y26_wr','y26_ev','y26_pf','y26sim_payouts','y26sim_breaches','y26sim_first_portfolio_payout_days']].to_string(index=False))
