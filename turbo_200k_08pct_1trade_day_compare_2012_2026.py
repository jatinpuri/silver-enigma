import pandas as pd
import numpy as np
from collections import defaultdict

START = 200000.0
RISK = 0.008 * START
TRAIL = 0.06 * START
PROFIT_DAY_MIN = 0.005 * START
CONSISTENCY_MAX = 0.20
WAIT_DAYS = 14
SPLIT = 0.80
WINDOWS = {'LON_OPEN','LON_H1','NY_H1'}
START_DATE = pd.Timestamp('2012-01-01', tz='UTC')
END_DATE = pd.Timestamp('2026-08-09', tz='UTC')  # includes trades through 2026-08-08

tr = pd.read_csv('eurusd_multihour_trades.csv')
for c in ['signal_dt','entry_bar_dt','exit_dt']:
    tr[c] = pd.to_datetime(tr[c], utc=True)
tr = tr[(tr.signal_dt >= START_DATE) & (tr.signal_dt < END_DATE) & tr.window.isin(WINDOWS) & (tr.local_weekday <= 4)].copy()

# Hard max one trade per signal/trading day: first triggered setup chronologically.
tr['trade_day'] = tr['signal_dt'].dt.date
tr = tr.sort_values(['trade_day','entry_bar_dt','signal_dt','window'])
tr = tr.groupby('trade_day', as_index=False, sort=True).first()
tr = tr.sort_values(['exit_dt','signal_dt','window']).reset_index(drop=True)


def simulate(buffer):
    post_payout_base = START + buffer
    bal = START
    floor = START - TRAIL
    acct = 1
    cycle_start = None
    cycle_base = START
    daily = defaultdict(float)
    payouts = []
    breaches = []
    yearly = defaultdict(lambda: {'trades':0,'payouts':0,'breaches':0,'gross_requested':0.0,'user_paid':0.0})

    for _, r in tr.iterrows():
        day = r.trade_day
        year = day.year
        yearly[year]['trades'] += 1
        if cycle_start is None:
            cycle_start = day

        pnl = float(r.R) * RISK
        bal += pnl
        daily[day] += pnl

        if bal < floor - 1e-9:
            breaches.append({
                'buffer':buffer,'account':acct,'date':str(day),'year':year,
                'balance':bal,'floor':floor,'pnl':pnl,'window':r.window
            })
            yearly[year]['breaches'] += 1
            acct += 1
            bal = START
            floor = START - TRAIL
            cycle_start = day
            cycle_base = START
            daily = defaultdict(float)
            continue

        if (day - cycle_start).days < WAIT_DAYS:
            continue

        cycle_profit = bal - cycle_base
        if cycle_profit <= 0:
            continue
        profdays = sum(v >= PROFIT_DAY_MIN - 1e-9 for v in daily.values())
        best = max([0.0] + list(daily.values()))
        consistency = best / cycle_profit if cycle_profit > 0 else np.inf
        if profdays < 3 or consistency > CONSISTENCY_MAX + 1e-12:
            continue

        desired_after = post_payout_base
        gross_request = bal - desired_after
        if gross_request <= 0:
            continue
        user_cash = gross_request * SPLIT

        payouts.append({
            'buffer':buffer,'account':acct,'date':str(day),'year':year,
            'gross_requested':gross_request,'user_payout':user_cash,
            'profitable_days':profdays,'best_day':best,'consistency':consistency,
            'balance_before':bal,'floor_before':floor,'balance_after':desired_after
        })
        yearly[year]['payouts'] += 1
        yearly[year]['gross_requested'] += gross_request
        yearly[year]['user_paid'] += user_cash

        bal = desired_after
        floor = START
        cycle_start = day
        cycle_base = desired_after
        daily = defaultdict(float)

    cycle_profit = bal - cycle_base
    best = max([0.0] + list(daily.values()))
    profdays = sum(v >= PROFIT_DAY_MIN - 1e-9 for v in daily.values())
    consistency = best / cycle_profit if cycle_profit > 0 else np.inf

    summary = {
        'buffer':buffer,
        'trades':len(tr),
        'payouts':len(payouts),
        'breaches':len(breaches),
        'accounts_used':acct,
        'gross_requested':sum(x['gross_requested'] for x in payouts),
        'user_paid':sum(x['user_payout'] for x in payouts),
        'ending_balance':bal,
        'ending_floor':floor,
        'ending_cycle_base':cycle_base,
        'ending_cycle_profit':cycle_profit,
        'ending_profitable_days':profdays,
        'ending_best_day':best,
        'ending_consistency':consistency,
    }

    yearly_rows = []
    for y in range(2012, 2027):
        d = yearly[y]
        yearly_rows.append({'buffer':buffer,'year':y,**d})
    return summary, payouts, breaches, yearly_rows

summaries=[]; all_payouts=[]; all_breaches=[]; all_yearly=[]
for buffer in [0.0, 4000.0]:
    s,p,b,y = simulate(buffer)
    summaries.append(s); all_payouts += p; all_breaches += b; all_yearly += y

pd.DataFrame(summaries).to_csv('turbo_compare_2012_2026_summary.csv', index=False)
pd.DataFrame(all_yearly).to_csv('turbo_compare_2012_2026_yearly.csv', index=False)
pd.DataFrame(all_payouts).to_csv('turbo_compare_2012_2026_payouts.csv', index=False)
pd.DataFrame(all_breaches).to_csv('turbo_compare_2012_2026_breaches.csv', index=False)

print('SUMMARY')
print(pd.DataFrame(summaries).to_string(index=False))
print('\nYEARLY')
print(pd.DataFrame(all_yearly).to_string(index=False))
print('\nPAYOUTS')
print(pd.DataFrame(all_payouts).to_string(index=False))
print('\nBREACHES')
print(pd.DataFrame(all_breaches).to_string(index=False))
