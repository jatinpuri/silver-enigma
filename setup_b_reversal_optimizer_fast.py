import json, itertools
import numpy as np
import pandas as pd
ATR_LEN=14; ADX_LEN=14; HOLD=12
START=pd.Timestamp('2012-01-01',tz='UTC'); TRAIN_END=pd.Timestamp('2022-01-01',tz='UTC'); END=pd.Timestamp('2026-08-09',tz='UTC')
RRS=np.array([0.20,0.25,0.30,0.40,0.50,0.60,0.75,1.00]); SLS=np.array([0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,3.00])
def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def pf(a):
 a=np.asarray(a,float); gp=a[a>0].sum(); gl=-a[a<0].sum(); return gp/gl if gl>0 else np.nan
def maxdd(a):
 a=np.asarray(a,float); e=np.r_[0,np.cumsum(a)]; p=np.maximum.accumulate(e); return float(np.max(p-e))
def ind(df):
 h,l,c=df.high,df.low,df.close; pc=c.shift(); tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); atr=rma(tr,14)
 up=h.diff(); dn=-l.diff(); pdm=pd.Series(np.where((up>dn)&(up>0),up,0.),index=df.index); mdm=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=df.index); base=rma(tr,14)
 pdi=100*rma(pdm,14)/base; mdi=100*rma(mdm,14)/base; dx=100*(pdi-mdi).abs()/(pdi+mdi); adx=rma(dx,14); return atr,adx,pdi,mdi
arr=json.load(open('eurusd_h1_2011_2026.json')); df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close']); df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True); df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True); df['atr'],df['adx'],df['pdi'],df['mdi']=ind(df)
df['cdir']=np.sign(df.close-df.open).fillna(0).astype(int); df['didir']=np.sign(df.pdi-df.mdi).fillna(0).astype(int); df['rng']=df.high-df.low; df['body']=(df.close-df.open).abs(); df['bodyfrac']=df.body/df.rng.replace(0,np.nan); df['range_ratio']=df.rng/df.rng.shift(); df['di_gap']=(df.pdi-df.mdi).abs(); df['adx_drop']=df.adx.shift()-df.adx
local=df.dt.dt.tz_convert('Europe/London'); idxs=np.where((local.dt.hour.to_numpy()==7)&(local.dt.weekday.to_numpy()<=3)&(local.dt.year.to_numpy()>=2012)&(local.dt.year.to_numpy()<=2026))[0]
base=[]
for i in idxs:
 if i<1: continue
 r=df.iloc[i]; p=df.iloc[i-1]; c=int(r.cdir)
 if c==0 or int(r.didir)==0 or not np.isfinite(r.atr) or not np.isfinite(r.adx) or not np.isfinite(p.adx) or not r.adx<p.adx: continue
 d=-c; j=i+1
 if j>=len(df) or (df.loc[j,'dt']-r['dt'])>pd.Timedelta(hours=1,minutes=1): continue
 entry=float(r.high if d>0 else r.low)
 if (d>0 and float(df.loc[j,'high'])<entry) or (d<0 and float(df.loc[j,'low'])>entry): continue
 base.append(dict(i=i,j=j,d=d,dt=r['dt'],c=c,di=int(r.didir),adx=float(r.adx),drop=float(r.adx_drop),gap=float(r.di_gap),rratio=float(r.range_ratio),bodyfrac=float(r.bodyfrac),atr=float(r.atr),entry=entry))
B=pd.DataFrame(base); n=len(B)
outcomes={}
high=df.high.to_numpy(float); low=df.low.to_numpy(float); close=df.close.to_numpy(float)
for sl in SLS:
 for rr in RRS:
  vals=np.empty(n); costs=np.empty(n); stress=np.empty(n)
  for z,row in enumerate(B.itertuples(index=False)):
   sd=sl*row.atr; st=row.entry-row.d*sd; tg=row.entry+row.d*(sl*rr)*row.atr; end=min(row.j+HOLD-1,len(df)-1); rv=None
   for k in range(row.j,end+1):
    hit_sl=(low[k]<=st) if row.d>0 else (high[k]>=st); hit_tp=(high[k]>=tg) if row.d>0 else (low[k]<=tg)
    if hit_sl: rv=-1.; break
    if hit_tp: rv=float(rr); break
   if rv is None: rv=((close[end]-row.entry)*row.d)/sd
   sp=sd/0.0001; vals[z]=rv; costs[z]=rv-0.4/sp; stress[z]=rv-0.6/sp
  outcomes[(float(sl),float(rr))]=(vals,costs,stress)

def mask(name):
 c=B.c.to_numpy(); di=B.di.to_numpy(); adx=B.adx.to_numpy(); drop=B['drop'].to_numpy(); gap=B.gap.to_numpy(); rr=B.rratio.to_numpy(); body=B.bodyfrac.to_numpy(); sup=c!=di
 D={'NONE':np.ones(n,bool),'DI_SUPPORT':sup,'DI_FADE':c==di,'DI_GAP3_SUPPORT':sup&(gap>=3),'DI_GAP5_SUPPORT':sup&(gap>=5),'DI_GAP10_SUPPORT':sup&(gap>=10),'ADX20_SUPPORT':sup&(adx>=20),'ADX25_SUPPORT':sup&(adx>=25),'ADX30_SUPPORT':sup&(adx>=30),'DROP0P5_SUPPORT':sup&(drop>=.5),'DROP1_SUPPORT':sup&(drop>=1),'RANGE_EXPAND_SUPPORT':sup&(rr>=1),'BODY50_SUPPORT':sup&(body>=.5),'SMALLBODY40_SUPPORT':sup&(body<=.4),'ADX20_GAP5_SUPPORT':sup&(adx>=20)&(gap>=5),'ADX20_GAP10_SUPPORT':sup&(adx>=20)&(gap>=10),'DROP0P5_GAP5_SUPPORT':sup&(drop>=.5)&(gap>=5),'RANGE_GAP5_SUPPORT':sup&(rr>=1)&(gap>=5)}
 return D[name]
FILTERS=['NONE','DI_SUPPORT','DI_FADE','DI_GAP3_SUPPORT','DI_GAP5_SUPPORT','DI_GAP10_SUPPORT','ADX20_SUPPORT','ADX25_SUPPORT','ADX30_SUPPORT','DROP0P5_SUPPORT','DROP1_SUPPORT','RANGE_EXPAND_SUPPORT','BODY50_SUPPORT','SMALLBODY40_SUPPORT','ADX20_GAP5_SUPPORT','ADX20_GAP10_SUPPORT','DROP0P5_GAP5_SUPPORT','RANGE_GAP5_SUPPORT']
dts=pd.to_datetime(B.dt,utc=True); rows=[]
for f in FILTERS:
 m=mask(f)
 for sl in SLS:
  for rr in RRS:
   R,Rc,Rs=outcomes[(float(sl),float(rr))]
   for pl,a,b in [('TRAIN',START,TRAIN_END),('OOS',TRAIN_END,END),('YTD2026',pd.Timestamp('2026-01-01',tz='UTC'),END),('FULL',START,END)]:
    q=m&(dts>=a)&(dts<b); x=R[q]
    if len(x)==0: continue
    yrs=pd.DataFrame({'dt':dts[q],'R':x}).groupby(dts[q].year).R.sum()
    rows.append([f,sl,rr,pl,len(x),(x>0).mean(),pf(x),x.mean(),Rc[q].mean(),Rs[q].mean(),x.sum(),maxdd(x),int((yrs>0).sum()),len(yrs)])
out=pd.DataFrame(rows,columns=['filter','sl_atr','rr','period','trades','wr','pf','ev','ev_comm','ev_stress','total','maxdd','positive_years','years']); out.to_csv('setup_b_reversal_optimizer_fast_all.csv',index=False)
w=out.pivot_table(index=['filter','sl_atr','rr'],columns='period',values=['trades','wr','pf','ev','ev_comm','ev_stress','total'],aggfunc='first'); w.columns=['__'.join(c) for c in w.columns]; w=w.reset_index()
sel=w[(w.trades__TRAIN>=120)&(w.ev_stress__TRAIN>0)&(w.trades__OOS>=40)&(w.ev_stress__OOS>0)].copy()
if len(sel):
 sel['score']=sel.ev_stress__OOS*np.sqrt(sel.trades__OOS)-.5*(sel.ev_stress__TRAIN-sel.ev_stress__OOS).abs(); sel=sel.sort_values('score',ascending=False)
sel.to_csv('setup_b_reversal_optimizer_fast_robust.csv',index=False); print('N signals',n); print(sel.head(40).to_string(index=False) if len(sel) else 'NO ROBUST CANDIDATES')
