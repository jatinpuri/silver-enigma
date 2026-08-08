import numpy as np
import pandas as pd

TARGET = ['LON_OPEN','LON_H1','NY_H1']
POLICIES = {'MonThu': {0,1,2,3}, 'MonFri': {0,1,2,3,4}}

def pf(x):
    a=np.asarray(x,float)
    gp=a[a>0].sum(); gl=-a[a<0].sum()
    return gp/gl if gl>0 else (np.inf if gp>0 else np.nan)

def maxdd(q):
    if len(q)==0: return np.nan
    q=q.sort_values('exit_dt')
    eq=np.r_[0.0,q['R'].cumsum().to_numpy(float)]
    peak=np.maximum.accumulate(eq)
    return float(np.max(peak-eq))

tr=pd.read_csv('eurusd_multihour_trades.csv',parse_dates=['signal_dt','entry_bar_dt','exit_dt'])
rows=[]
for policy,days in POLICIES.items():
    z=tr[tr['local_weekday'].isin(days) & tr['window'].isin(TARGET)].copy()
    for year in range(2012,2027):
        q=z[z['year']==year].copy()
        r=q['R'].to_numpy(float)
        rows.append({
            'day_policy':policy,'year':year,'trades':len(q),
            'totalR':float(r.sum()) if len(r) else 0.0,
            'ev':float(r.mean()) if len(r) else np.nan,
            'pf':float(pf(r)) if len(r) else np.nan,
            'wr':float((r>0).mean()) if len(r) else np.nan,
            'maxDD_R':maxdd(q)
        })

df=pd.DataFrame(rows)
df.to_csv('eurusd_target_annual_dd.csv',index=False)

summary=[]
for policy in POLICIES:
    q=df[(df.day_policy==policy) & (df.year<=2025)]
    summary.append({
        'day_policy':policy,
        'completed_years':len(q),
        'avg_annual_maxDD_R':q.maxDD_R.mean(),
        'median_annual_maxDD_R':q.maxDD_R.median(),
        'worst_annual_maxDD_R':q.maxDD_R.max(),
        'worst_DD_year':int(q.loc[q.maxDD_R.idxmax(),'year']),
        'best_annual_maxDD_R':q.maxDD_R.min(),
        'avg_annual_totalR':q.totalR.mean(),
        'median_annual_totalR':q.totalR.median(),
        'profitable_years':int((q.totalR>0).sum()),
    })
pd.DataFrame(summary).to_csv('eurusd_target_annual_dd_summary.csv',index=False)
print(df.to_string(index=False))
print('\nSUMMARY')
print(pd.DataFrame(summary).to_string(index=False))
