import pandas as pd

for buffer in [0.0,4000.0]:
    pays=pd.read_csv('turbo_compare_2012_2026_payouts.csv')
    br=pd.read_csv('turbo_compare_2012_2026_breaches.csv')
    pays=pays[pays['buffer']==buffer].copy()
    br=br[br['buffer']==buffer].copy()
    pays['date']=pd.to_datetime(pays['date'])
    br['date']=pd.to_datetime(br['date'])

    max_acct=int(max(pays['account'].max() if len(pays) else 0, br['account'].max() if len(br) else 0))
    # Account 1 begins 2012-01-01. Each subsequent account begins on the breach date of the prior account,
    # matching the simulation's replacement/reset convention.
    starts={1:pd.Timestamp('2012-01-01')}
    for _,r in br.sort_values('account').iterrows():
        starts[int(r['account'])+1]=r['date']

    rows=[]
    for acct in range(1,max_acct+1):
        start=starts.get(acct)
        p=pays[pays['account']==acct].sort_values('date')
        b=br[br['account']==acct].sort_values('date')
        first_payout=p.iloc[0]['date'] if len(p) else pd.NaT
        breach=b.iloc[0]['date'] if len(b) else pd.NaT
        got=not pd.isna(first_payout)
        rows.append({
            'buffer':buffer,'account':acct,'start_date':start.date() if start is not None else None,
            'first_payout_date':first_payout.date() if got else None,
            'days_to_first_payout':(first_payout-start).days if got and start is not None else None,
            'breach_date':breach.date() if not pd.isna(breach) else None,
            'got_payout_before_breach':got
        })
    df=pd.DataFrame(rows)
    succ=df[df.got_payout_before_breach & df.days_to_first_payout.notna()]
    out={
        'buffer':buffer,
        'accounts_observed':len(df),
        'accounts_with_first_payout':len(succ),
        'payout_before_breach_rate':len(succ)/len(df) if len(df) else None,
        'avg_days_to_first_payout_successful':succ.days_to_first_payout.mean(),
        'median_days_to_first_payout_successful':succ.days_to_first_payout.median(),
        'min_days_to_first_payout_successful':succ.days_to_first_payout.min(),
        'max_days_to_first_payout_successful':succ.days_to_first_payout.max(),
        'avg_weeks_to_first_payout_successful':succ.days_to_first_payout.mean()/7,
        'median_weeks_to_first_payout_successful':succ.days_to_first_payout.median()/7,
    }
    pd.DataFrame([out]).to_csv(f'turbo_first_payout_timing_buffer_{int(buffer)}.csv',index=False)
    df.to_csv(f'turbo_first_payout_accounts_buffer_{int(buffer)}.csv',index=False)
    print(pd.DataFrame([out]).to_string(index=False))
