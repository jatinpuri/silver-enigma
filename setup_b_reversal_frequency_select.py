import pandas as pd, numpy as np
x=pd.read_csv('setup_b_reversal_optimizer_fast_all.csv')
w=x.pivot_table(index=['filter','sl_atr','rr'],columns='period',values=['trades','wr','pf','ev','ev_comm','ev_stress','total','maxdd'],aggfunc='first')
w.columns=['__'.join(c) for c in w.columns]; w=w.reset_index()
# Must be positive after actual commission in train, OOS and 2026. Stress is reported but not a hard requirement.
s=w[(w.trades__TRAIN>=100)&(w.trades__OOS>=35)&(w.trades__YTD2026>=5)&(w.ev_comm__TRAIN>0)&(w.ev_comm__OOS>0)&(w.ev_comm__YTD2026>0)].copy()
if len(s):
 s['min_comm_ev']=s[['ev_comm__TRAIN','ev_comm__OOS','ev_comm__YTD2026']].min(axis=1)
 s['freq_score']=s.trades__YTD2026 + .1*s.trades__OOS
 s=s.sort_values(['trades__YTD2026','trades__OOS','min_comm_ev'],ascending=False)
s.to_csv('setup_b_reversal_frequency_candidates.csv',index=False)
print(s.head(50).to_string(index=False) if len(s) else 'NO CANDIDATES')
