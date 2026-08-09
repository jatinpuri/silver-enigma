import pandas as pd, numpy as np
x=pd.read_csv('setup_b_reversal_optimizer_fast_all.csv')
w=x.pivot_table(index=['filter','sl_atr','rr'],columns='period',values=['trades','wr','pf','ev','ev_comm','ev_stress','total','maxdd'],aggfunc='first')
w.columns=['__'.join(c) for c in w.columns]; w=w.reset_index()
req=(w['trades__TRAIN']>=100)&(w['trades__OOS']>=35)&(w['trades__YTD2026']>=5)&(w['ev_stress__TRAIN']>0)&(w['ev_stress__OOS']>0)&(w['ev_stress__YTD2026']>0)
s=w[req].copy()
if len(s):
 s['min_stress_ev']=s[['ev_stress__TRAIN','ev_stress__OOS','ev_stress__YTD2026']].min(axis=1)
 s['score']=s.min_stress_ev*np.sqrt(s['trades__OOS'])
 s=s.sort_values(['score','trades__OOS'],ascending=False)
s.to_csv('setup_b_reversal_strict_candidates.csv',index=False)
print(s.head(50).to_string(index=False) if len(s) else 'NO STRICT CANDIDATES')
