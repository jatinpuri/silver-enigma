import json, heapq
import numpy as np
import pandas as pd
from optimizer import (SYMBOLS, ATR_PERIODS, FILTERS, DAY_POLICIES, ENTRY_MODES,
                       load_symbol, signal_indices, build_trades, simulate_rs,
                       metrics, rank_key)

ATR_GRID=[7,10,14,20,28]
SL_GRID=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0]
TP_GRID=[0.5,0.75,1.0,1.5,2.0,2.5,3.0,3.375,4.0,4.5,5.0]
HOLD_GRID=[4,6,8,10,12,16,24]
EXP_GRID=[1,2]
TOP_STRUCT=4
TOP_KEEP=30

DATA={}
for s in SYMBOLS:
    print('load',s,flush=True)
    df=load_symbol(s)
    DATA[s]=df


def prepare(symbol,hour,filt,dayp,ap,mode_name,expiry):
    df=DATA[symbol]
    high=df['high'].to_numpy(float); low=df['low'].to_numpy(float)
    open_=df['open'].to_numpy(float); close=df['close'].to_numpy(float)
    direction=df['dir'].to_numpy(np.int8); years=df['year'].to_numpy(np.int64)
    sig=signal_indices(df,hour,filt,dayp,ap)
    mode=0 if mode_name=='breakout' else 1
    ex=expiry if mode==0 else 1
    atr=df[f'atr{ap}'].to_numpy(float)
    trig,dirs,ent,at,yrs=build_trades(sig,direction,high,low,open_,atr,years,ex,mode)
    span=max((df['date'].iloc[-1]-df['date'].iloc[0]).total_seconds()/(365.25*86400),1.0)
    return (trig,dirs,ent,at,yrs,high,low,close,span)


def unit_metrics(pack,sl,tp,hold):
    trig,dirs,ent,at,yrs,high,low,close,span=pack
    if len(trig)<50: return None
    rs=simulate_rs(trig,dirs,ent,at,high,low,close,sl,tp,hold)
    return metrics(rs,yrs,span)


def portfolio_score(ms):
    if not ms: return (-9,)*8
    # common setup should not be rescued by a few great pairs
    minpf=min(m['pf'] for m in ms)
    minev=min(m['ev'] for m in ms)
    medpf=float(np.median([m['pf'] for m in ms]))
    medev=float(np.median([m['ev'] for m in ms]))
    robust_units=sum(1 for m in ms if m['pf']>1 and m['ev']>0)
    minsub=min(m['min_pf'] for m in ms)
    avgpf=float(np.mean([m['pf'] for m in ms]))
    avgev=float(np.mean([m['ev'] for m in ms]))
    return (robust_units,minpf,minev,minsub,medpf,medev,avgpf,avgev)


def evaluate(scope,mode_name,filt,dayp,ap,sl,tp,hold,expiry):
    units=[]; detail=[]
    hours=[1] if scope=='H1' else ([2] if scope=='H2' else [1,2])
    for s in SYMBOLS:
        for h in hours:
            pack=prepare(s,h,filt,dayp,ap,mode_name,expiry)
            m=unit_metrics(pack,sl,tp,hold)
            if m is None: return None
            units.append(m)
            detail.append((s,h,m))
    score=portfolio_score(units)
    return score,detail


def stage1(scope):
    rows=[]
    for mode in ENTRY_MODES:
        for filt in FILTERS:
            for dayp in DAY_POLICIES:
                exs=[1] if mode=='next_open' else [1,2]
                for ex in exs:
                    z=evaluate(scope,mode,filt,dayp,20,1.75,3.375,10,ex)
                    if z is None: continue
                    score,_=z
                    rows.append((score,mode,filt,dayp,ex))
    rows.sort(reverse=True)
    print('stage1',scope,rows[:TOP_STRUCT],flush=True)
    return rows[:TOP_STRUCT]


def optimise(scope):
    structs=stage1(scope)
    heap=[]
    for _,mode,filt,dayp,base_ex in structs:
        for ap in ATR_GRID:
            exs=[1] if mode=='next_open' else EXP_GRID
            for ex in exs:
                # pre-build once per symbol/hour for this structural config
                hours=[1] if scope=='H1' else ([2] if scope=='H2' else [1,2])
                packs={}
                valid=True
                for s in SYMBOLS:
                    for h in hours:
                        p=prepare(s,h,filt,dayp,ap,mode,ex)
                        if len(p[0])<50:
                            valid=False; break
                        packs[(s,h)]=p
                    if not valid: break
                if not valid: continue
                for sl in SL_GRID:
                    for tp in TP_GRID:
                        for hold in HOLD_GRID:
                            detail=[]; ms=[]
                            for key,p in packs.items():
                                m=unit_metrics(p,sl,tp,hold)
                                if m is None: valid=False; break
                                ms.append(m); detail.append((key[0],key[1],m))
                            if not valid: continue
                            score=portfolio_score(ms)
                            row={'scope':scope,'entry_mode':mode,'filter':filt,'day_policy':dayp,
                                 'atr_period':ap,'sl_atr':sl,'tp_atr':tp,'rr':tp/sl,
                                 'expiry_h':0 if mode=='next_open' else ex,'hold_h':hold,
                                 'positive_units':score[0],'min_pf':score[1],'min_ev':score[2],
                                 'min_subperiod_pf':score[3],'median_pf':score[4],'median_ev':score[5],
                                 'avg_pf':score[6],'avg_ev':score[7]}
                            for s,h,m in detail:
                                row[f'{s}_H{h}_pf']=m['pf']; row[f'{s}_H{h}_ev']=m['ev']; row[f'{s}_H{h}_n']=m['n']
                            item=(score,json.dumps(row,sort_keys=True),row)
                            if len(heap)<TOP_KEEP: heapq.heappush(heap,item)
                            elif item[0]>heap[0][0]: heapq.heapreplace(heap,item)
    rows=[x[2] for x in sorted(heap,reverse=True)]
    for i,r in enumerate(rows,1): r['rank']=i
    if rows: print('BEST',scope,json.dumps(rows[0],sort_keys=True),flush=True)
    return rows

allrows=[]
for scope in ['H1','H2','BOTH']:
    allrows.extend(optimise(scope))

out=pd.DataFrame(allrows)
out.to_csv('universal_optimisation_top.csv',index=False)
out[out['rank']==1].to_csv('universal_optimisation_best.csv',index=False)
print('\nFINAL')
print(out[out['rank']==1].to_string(index=False))
