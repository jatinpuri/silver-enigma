import json
import numpy as np
import pandas as pd
ATR_LEN=14; ADX_LEN=14; SL_ATR=2.25; TP_ATR=3.375; RR=1.5; HOLD=12
START=pd.Timestamp('2012-01-01',tz='UTC'); RECENT=pd.Timestamp('2022-01-01',tz='UTC'); Y26=pd.Timestamp('2026-01-01',tz='UTC'); END=pd.Timestamp('2026-08-09',tz='UTC')
def rma(s,n): return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
def pf(a):
 a=np.asarray(a,float); gp=a[a>0].sum(); gl=-a[a<0].sum(); return gp/gl if gl>0 else np.nan
def mdd(a):
 a=np.asarray(a,float); e=np.r_[0,np.cumsum(a)]; p=np.maximum.accumulate(e); return float(np.max(p-e))
arr=json.load(open('eurusd_h1_2011_2026.json')); df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True); df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True); df=df[(df.high!=df.low)|(df.open!=df.close)].reset_index(drop=True)
h,l,c=df.high,df.low,df.close; pc=c.shift(1); tr0=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); df['atr']=rma(tr0,14)
up=h.diff(); dn=-l.diff(); pdm=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=df.index); mdm=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=df.index); base=rma(tr0,14); df['pdi']=100*rma(pdm,14)/base; df['mdi']=100*rma(mdm,14)/base; dx=100*(df.pdi-df.mdi).abs()/(df.pdi+df.mdi); df['adx']=rma(dx,14)
df['cdir']=np.where(df.close>df.open,1,np.where(df.close<df.open,-1,0)); df['didir']=np.where(df.pdi>df.mdi,1,np.where(df.mdi>df.pdi,-1,0)); df['range']=df.high-df.low
loc=df.dt.dt.tz_convert('Europe/London'); idxs=np.where((loc.dt.hour.to_numpy()==7)&(loc.dt.weekday.to_numpy()<=3)&(loc.dt.year.to_numpy()>=2012)&(loc.dt.year.to_numpy()<=2026))[0]
variants=['RISE_BASE','FALL_ADX20_SAME','FALL_ADX25_SAME','FALL_ADX30_SAME','FALL_ADX35_SAME','FALL_AGREE_ADX20','FALL_AGREE_ADX25','FALL_AGREE_ADX30','FALL_AGREE_GAP3','FALL_AGREE_GAP5','FALL_AGREE_GAP10','FALL_RANGE_EXPAND','FALL_AGREE_RANGE_EXPAND','FALL_AGREE_ADX20_GAP5','FALL_AGREE_ADX25_GAP5']
def direction(v,i):
 if i<1:return 0
 r=df.iloc[i]; p=df.iloc[i-1]; c=int(r.cdir); di=int(r.didir); fall=r.adx<p.adx; agree=c==di; gap=abs(r.pdi-r.mdi); rex=r['range']>p['range']
 if c==0 or di==0:return 0
 if v=='RISE_BASE': return c if r.adx>p.adx else 0
 if not fall:return 0
 if v.startswith('FALL_ADX') and v.endswith('_SAME'):
  th=int(v.split('ADX')[1].split('_')[0]); return c if r.adx>=th else 0
 if v.startswith('FALL_AGREE_ADX') and '_GAP' not in v:
  th=int(v.split('ADX')[1]); return c if agree and r.adx>=th else 0
 if v.startswith('FALL_AGREE_GAP'):
  th=int(v.split('GAP')[1]); return c if agree and gap>=th else 0
 if v=='FALL_RANGE_EXPAND': return c if rex else 0
 if v=='FALL_AGREE_RANGE_EXPAND': return c if agree and rex else 0
 if v=='FALL_AGREE_ADX20_GAP5': return c if agree and r.adx>=20 and gap>=5 else 0
 if v=='FALL_AGREE_ADX25_GAP5': return c if agree and r.adx>=25 and gap>=5 else 0
 return 0
trades=[]
for v in variants:
 for i in idxs:
  d=direction(v,i); r=df.iloc[i]
  if d==0 or not np.isfinite(r.atr) or r.atr<=0:continue
  j=i+1
  if j>=len(df) or (df.loc[j,'dt']-r['dt'])>pd.Timedelta(hours=1,minutes=1):continue
  entry=float(r.high if d>0 else r.low)
  if d>0 and df.loc[j,'high']<entry:continue
  if d<0 and df.loc[j,'low']>entry:continue
  sd=2.25*float(r.atr); sp=sd/0.0001; stop=entry-d*sd; target=entry+d*3.375*float(r.atr); end=min(j+11,len(df)-1); rv=None
  for k in range(j,end+1):
   hi=float(df.loc[k,'high']); lo=float(df.loc[k,'low']); hs=(lo<=stop) if d>0 else (hi>=stop); ht=(hi>=target) if d>0 else (lo<=target)
   if hs:rv=-1.;break
   if ht:rv=1.5;break
  if rv is None:rv=((float(df.loc[end,'close'])-entry)*d)/sd
  trades.append([v,r['dt'],rv,rv-0.4/sp,rv-0.6/sp,sp])
t=pd.DataFrame(trades,columns=['variant','signal_dt','R','R_comm','R_stress','stop_pips']); rows=[]
for v in variants:
 z0=t[t.variant==v].sort_values('signal_dt')
 for lab,st in [('FULL',START),('RECENT',RECENT),('YTD2026',Y26)]:
  z=z0[(z0.signal_dt>=st)&(z0.signal_dt<END)]
  if not len(z):continue
  y=z.groupby(z.signal_dt.dt.year).R.sum(); rows.append([v,lab,len(z),(z.R>0).mean(),pf(z.R),z.R.mean(),z.R_comm.mean(),z.R_stress.mean(),z.R.sum(),mdd(z.R),int((y>0).sum()),len(y)])
out=pd.DataFrame(rows,columns=['variant','period','trades','win_rate','PF','EV_R','EV_after_comm_R','EV_after_comm_0p2_R','total_R','max_DD_R','positive_years','years']); out.to_csv('setup_b_adx_falling_quality_filters_results.csv',index=False); print(out.to_string(index=False))
