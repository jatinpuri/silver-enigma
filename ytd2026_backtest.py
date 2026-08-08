import json
import numpy as np
import pandas as pd

SYMBOLS=['EURUSD','GBPUSD','AUDUSD','USDCHF','USDJPY','EURGBP','XAUUSD']
SL=1.5; TP=5.0; HOLD=4; ATRP=28

def wilder_adx(df, length=14):
    h,l,c=df.high,df.low,df.close
    pc=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    up=h.diff(); down=-l.diff()
    plus=pd.Series(np.where((up>down)&(up>0),up,0.0),index=df.index)
    minus=pd.Series(np.where((down>up)&(down>0),down,0.0),index=df.index)
    atrw=tr.ewm(alpha=1/length,adjust=False,min_periods=length).mean()
    pdi=100*plus.ewm(alpha=1/length,adjust=False,min_periods=length).mean()/atrw.replace(0,np.nan)
    mdi=100*minus.ewm(alpha=1/length,adjust=False,min_periods=length).mean()/atrw.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    adx=dx.ewm(alpha=1/length,adjust=False,min_periods=length).mean()
    return tr,adx

def pf(rs):
    rs=np.array(rs,float); gp=rs[rs>0].sum(); gl=-rs[rs<0].sum()
    return gp/gl if gl>0 else float('inf')

def one(sym):
    arr=json.load(open(f'ytd_{sym}_h1.json'))
    df=pd.DataFrame(arr,columns=['timestamp','open','high','low','close'])
    df['dt']=pd.to_datetime(df.timestamp,unit='ms',utc=True)
    df=df.drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    # dukascopy-node can pad closed-market hours with flat candles when volume is omitted.
    # Remove these before ATR/ADX so weekends do not artificially suppress volatility.
    df=df[(df['high']>df['low']) | (df['open']!=df['close'])].copy().reset_index(drop=True)
    lon=df['dt'].dt.tz_convert('Europe/London')
    df['lh']=lon.dt.hour
    df['wd']=lon.dt.weekday
    df['year']=lon.dt.year
    df['range']=df.high-df.low
    df['dir']=np.where(df.close>df.open,1,np.where(df.close<df.open,-1,0))
    tr,adx=wilder_adx(df,14)
    df['atr']=tr.rolling(ATRP,min_periods=ATRP).mean()
    df['adx']=adx
    df['adx_prev']=adx.shift(1)
    trades=[]
    for i,row in df.iterrows():
        if row['year']!=2026 or row['lh']!=8 or row['wd']>4 or row['dir']==0: continue
        if not (np.isfinite(row['atr']) and np.isfinite(row['adx']) and np.isfinite(row['adx_prev'])): continue
        if not (row['adx']>row['adx_prev'] and row['range']>row['atr']): continue
        j=i+1
        if j>=len(df): continue
        # Next bar must actually be the 09:00-10:00 London bar; otherwise no valid 1h expiry window.
        lon_j=df.loc[j,'dt'].tz_convert('Europe/London')
        if lon_j.hour!=9 or lon_j.date()!=row['dt'].tz_convert('Europe/London').date(): continue
        d=int(row['dir']); entry=row['high'] if d>0 else row['low']
        if d>0 and df.loc[j,'high']<entry: continue
        if d<0 and df.loc[j,'low']>entry: continue
        risk=SL*row['atr']; target=entry+d*TP*row['atr']; stop=entry-d*risk
        end=min(j+HOLD-1,len(df)-1); rv=None; reason='time'
        for k in range(j,end+1):
            hi,lo=df.loc[k,'high'],df.loc[k,'low']
            hs=(lo<=stop) if d>0 else (hi>=stop)
            ht=(hi>=target) if d>0 else (lo<=target)
            if hs: rv=-1.0; reason='SL'; break
            if ht: rv=TP/SL; reason='TP'; break
        if rv is None:
            rv=((df.loc[end,'close']-entry)*d)/risk
        trades.append({'symbol':sym,'signal':str(row['dt']),'R':rv,'reason':reason})
    return trades

alltr=[]
for s in SYMBOLS:
    t=one(s); alltr+=t
    rs=[x['R'] for x in t]
    print(s,'n',len(rs),'EV',np.mean(rs) if rs else np.nan,'PF',pf(rs) if rs else np.nan,'WR',np.mean(np.array(rs)>0) if rs else np.nan,'sumR',np.sum(rs) if rs else 0)
rs=np.array([x['R'] for x in alltr],float)
print('ALL','n',len(rs),'EV',rs.mean(),'PF',pf(rs),'WR',(rs>0).mean(),'sumR',rs.sum())
pd.DataFrame(alltr).to_csv('ytd2026_trades.csv',index=False)
summary=[]
for s in SYMBOLS:
    z=np.array([x['R'] for x in alltr if x['symbol']==s],float)
    summary.append({'symbol':s,'trades':len(z),'ev':z.mean() if len(z) else np.nan,'pf':pf(z) if len(z) else np.nan,'wr':(z>0).mean() if len(z) else np.nan,'sumR':z.sum() if len(z) else 0})
summary.append({'symbol':'ALL','trades':len(rs),'ev':rs.mean(),'pf':pf(rs),'wr':(rs>0).mean(),'sumR':rs.sum()})
pd.DataFrame(summary).to_csv('ytd2026_summary.csv',index=False)
