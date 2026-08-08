import io
import json
import math
import heapq
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from numba import njit

warnings.filterwarnings('ignore')

SYMBOLS = ['EURUSD','GBPUSD','AUDUSD','USDCHF','USDJPY','EURGBP','XAUUSD']
ATR_PERIODS = [7, 10, 14, 20, 28]
SL_GRID = np.round(np.arange(0.50, 3.001, 0.25), 3).tolist()
TP_GRID = [0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.375,3.75,4.00,4.50,5.00]
HOLD_GRID = [4,6,8,10,12,16,24]
EXPIRY_GRID = [1,2]
DAY_POLICIES = ['MonThu','MonFri','TueThu','TueFri']
FILTERS = [
    'none','adxrise','di','adx_di','range_prev','range2','range_atr1',
    'body60','body70','adxrise_range','adxdi_range','adxrise_atr','adxdi_atr'
]
ENTRY_MODES = ['breakout','next_open']
STAGE1_GEOMS = [(0.75,1.50,8),(1.50,2.25,10),(2.25,3.375,12)]
TOP_STRUCTURES_PER_HOUR = 4
TOP_ROWS_PER_HOUR = 30


def wilder_adx(df, length=14):
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    up = h.diff()
    down = -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_w = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    plus_sm = plus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    minus_sm = minus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    pdi = 100.0 * plus_sm / atr_w.replace(0, np.nan)
    mdi = 100.0 * minus_sm / atr_w.replace(0, np.nan)
    dx = 100.0 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    return tr, adx, pdi, mdi


def load_symbol(symbol):
    url = f'https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/{symbol}/{symbol}h1.csv'
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df.columns = [str(c).strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df = df.dropna(subset=['date','open','high','low','close']).copy()
    df = df.drop_duplicates('date').sort_values('date').reset_index(drop=True)
    for c in ['open','high','low','close']:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    df = df.dropna(subset=['open','high','low','close']).reset_index(drop=True)
    london = df['date'].dt.tz_convert('Europe/London')
    df['london_hour'] = london.dt.hour.astype(int)
    df['weekday'] = london.dt.weekday.astype(int)
    df['year'] = london.dt.year.astype(int)
    df['range'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['body_ratio'] = df['body'] / df['range'].replace(0, np.nan)
    df['dir'] = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0)).astype(np.int8)
    tr, adx, pdi, mdi = wilder_adx(df, 14)
    df['tr'] = tr
    df['adx14'] = adx
    df['pdi14'] = pdi
    df['mdi14'] = mdi
    for p in ATR_PERIODS:
        df[f'atr{p}'] = tr.rolling(p, min_periods=p).mean()
    return df


def day_mask(weekday, policy):
    if policy == 'MonThu': return (weekday >= 0) & (weekday <= 3)
    if policy == 'MonFri': return (weekday >= 0) & (weekday <= 4)
    if policy == 'TueThu': return (weekday >= 1) & (weekday <= 3)
    if policy == 'TueFri': return (weekday >= 1) & (weekday <= 4)
    raise ValueError(policy)


def filter_mask(df, name, atr_period):
    d = df['dir'].to_numpy()
    adx = df['adx14'].to_numpy(float)
    pdi = df['pdi14'].to_numpy(float)
    mdi = df['mdi14'].to_numpy(float)
    rng = df['range'].to_numpy(float)
    br = df['body_ratio'].to_numpy(float)
    atr = df[f'atr{atr_period}'].to_numpy(float)
    adxrise = np.r_[False, adx[1:] > adx[:-1]]
    di = ((d > 0) & (pdi > mdi)) | ((d < 0) & (mdi > pdi))
    rprev = np.r_[False, rng[1:] > rng[:-1]]
    r2 = np.zeros(len(df), dtype=bool)
    if len(df) > 2:
        r2[2:] = rng[2:] > np.maximum(rng[1:-1], rng[:-2])
    ratr = rng > atr
    masks = {
        'none': np.ones(len(df), dtype=bool),
        'adxrise': adxrise,
        'di': di,
        'adx_di': adxrise & di,
        'range_prev': rprev,
        'range2': r2,
        'range_atr1': ratr,
        'body60': br >= 0.60,
        'body70': br >= 0.70,
        'adxrise_range': adxrise & rprev,
        'adxdi_range': adxrise & di & rprev,
        'adxrise_atr': adxrise & ratr,
        'adxdi_atr': adxrise & di & ratr,
    }
    return masks[name] & np.isfinite(atr) & np.isfinite(adx)


@njit(cache=True)
def build_trades(sig_idx, direction, high, low, open_, atr, years, expiry, mode):
    nmax = len(sig_idx)
    trig = np.empty(nmax, np.int64)
    dirs = np.empty(nmax, np.int8)
    ent = np.empty(nmax, np.float64)
    at = np.empty(nmax, np.float64)
    yr = np.empty(nmax, np.int64)
    m = 0
    N = len(high)
    for z in range(nmax):
        s = sig_idx[z]
        d = direction[s]
        if d == 0 or not np.isfinite(atr[s]):
            continue
        if mode == 1:  # next open
            k = s + 1
            if k >= N:
                continue
            trig[m] = k
            dirs[m] = d
            ent[m] = open_[k]
            at[m] = atr[s]
            yr[m] = years[s]
            m += 1
        else:  # breakout stop
            level = high[s] if d > 0 else low[s]
            found = -1
            for j in range(1, expiry + 1):
                k = s + j
                if k >= N:
                    break
                if (d > 0 and high[k] >= level) or (d < 0 and low[k] <= level):
                    found = k
                    break
            if found >= 0:
                trig[m] = found
                dirs[m] = d
                ent[m] = level
                at[m] = atr[s]
                yr[m] = years[s]
                m += 1
    return trig[:m], dirs[:m], ent[:m], at[:m], yr[:m]


@njit(cache=True)
def simulate_rs(trig, dirs, ent, at, high, low, close, sl_mult, tp_mult, hold):
    n = len(trig)
    rs = np.empty(n, np.float64)
    N = len(high)
    for i in range(n):
        t = trig[i]
        d = dirs[i]
        e = ent[i]
        a = at[i]
        risk = sl_mult * a
        tpdist = tp_mult * a
        stop = e - risk if d > 0 else e + risk
        target = e + tpdist if d > 0 else e - tpdist
        end = t + hold - 1
        if end >= N:
            end = N - 1
        resolved = False
        rv = 0.0
        for k in range(t, end + 1):
            if d > 0:
                hs = low[k] <= stop
                ht = high[k] >= target
            else:
                hs = high[k] >= stop
                ht = low[k] <= target
            # Conservative H1 path rule: if both touched in one H1 bar, count SL first.
            if hs:
                rv = -1.0
                resolved = True
                break
            if ht:
                rv = tp_mult / sl_mult
                resolved = True
                break
        if not resolved:
            rv = ((close[end] - e) * d) / risk
        rs[i] = rv
    return rs


def pf_value(x):
    if len(x) == 0: return np.nan
    gp = x[x > 0].sum()
    gl = -x[x < 0].sum()
    if gl <= 1e-12:
        return 99.0 if gp > 0 else 0.0
    return float(gp / gl)


def max_dd(x):
    if len(x) == 0: return np.nan
    eq = np.cumsum(x)
    peak = np.maximum.accumulate(np.r_[0.0, eq])
    dd = peak[1:] - eq
    return float(np.max(dd)) if len(dd) else 0.0


def metrics(rs, yrs, span_years):
    if len(rs) < 1:
        return None
    full_pf = pf_value(rs)
    full_ev = float(np.mean(rs))
    full_wr = float(np.mean(rs > 0))
    dd = max_dd(rs)
    splits = [('old', yrs <= 2015), ('mid', (yrs >= 2016) & (yrs <= 2018)), ('recent', yrs >= 2019)]
    out = {
        'n': int(len(rs)), 'wr': full_wr, 'pf': full_pf, 'ev': full_ev,
        'sumR': float(np.sum(rs)), 'ddR': dd,
        'trades_year': float(len(rs) / span_years)
    }
    pfs, evs, ns = [], [], []
    for nm, mk in splits:
        z = rs[mk]
        out[nm+'_n'] = int(len(z))
        out[nm+'_pf'] = pf_value(z) if len(z) else np.nan
        out[nm+'_ev'] = float(np.mean(z)) if len(z) else np.nan
        pfs.append(out[nm+'_pf']); evs.append(out[nm+'_ev']); ns.append(len(z))
    finite_pf = [v for v in pfs if np.isfinite(v)]
    finite_ev = [v for v in evs if np.isfinite(v)]
    out['min_pf'] = float(min(finite_pf)) if finite_pf else -99.0
    out['min_ev'] = float(min(finite_ev)) if finite_ev else -99.0
    out['robust'] = bool(min(ns) >= 20 and all(v > 1.0 for v in finite_pf) and all(v > 0 for v in finite_ev))
    return out


def rank_key(m):
    # Robustness first, then worst-subperiod PF/EV, then whole-sample quality and lower DD.
    return (
        1 if m['robust'] else 0,
        round(m['min_pf'], 8),
        round(m['min_ev'], 8),
        round(m['pf'], 8),
        round(m['ev'], 8),
        round(-m['ddR'], 8),
        m['n']
    )


def push_top(heap, row, limit=TOP_ROWS_PER_HOUR):
    key = rank_key(row)
    item = (key, json.dumps(row, sort_keys=True), row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def signal_indices(df, hour_num, filt, day_policy, atr_period):
    # Hour 1 decision at 08:00 London uses completed 07:00-08:00 candle.
    # Hour 2 decision at 09:00 London uses completed 08:00-09:00 candle.
    sig_hour = 7 if hour_num == 1 else 8
    base = (df['london_hour'].to_numpy() == sig_hour)
    base &= day_mask(df['weekday'].to_numpy(), day_policy)
    base &= (df['dir'].to_numpy() != 0)
    base &= filter_mask(df, filt, atr_period)
    return np.flatnonzero(base).astype(np.int64)


def optimise_symbol(symbol):
    print(f'\n===== {symbol} =====', flush=True)
    df = load_symbol(symbol)
    high = df['high'].to_numpy(float)
    low = df['low'].to_numpy(float)
    open_ = df['open'].to_numpy(float)
    close = df['close'].to_numpy(float)
    direction = df['dir'].to_numpy(np.int8)
    years = df['year'].to_numpy(np.int64)
    start = df['date'].iloc[0]
    end = df['date'].iloc[-1]
    span_years = max((end-start).total_seconds() / (365.25*86400), 1.0)

    # Warm numba once.
    _ = build_trades(np.array([100],dtype=np.int64), direction, high, low, open_, df['atr14'].to_numpy(float), years, 1, 0)

    selected = {}
    for hour_num in [1,2]:
        structs = []
        for mode_name in ENTRY_MODES:
            mode = 0 if mode_name == 'breakout' else 1
            for filt in FILTERS:
                for dayp in DAY_POLICIES:
                    sig = signal_indices(df, hour_num, filt, dayp, 14)
                    if len(sig) < 50:
                        continue
                    atr = df['atr14'].to_numpy(float)
                    trig, dirs, ent, at, yrs = build_trades(sig, direction, high, low, open_, atr, years, 1, mode)
                    if len(trig) < 45:
                        continue
                    bestm = None
                    for sl,tp,hold in STAGE1_GEOMS:
                        rs = simulate_rs(trig, dirs, ent, at, high, low, close, sl, tp, hold)
                        m = metrics(rs, yrs, span_years)
                        if m is None: continue
                        if bestm is None or rank_key(m) > rank_key(bestm):
                            bestm = m
                    if bestm is not None:
                        structs.append((rank_key(bestm), mode_name, filt, dayp, bestm))
        structs.sort(key=lambda x:x[0], reverse=True)
        selected[hour_num] = structs[:TOP_STRUCTURES_PER_HOUR]
        print('Stage1', hour_num, [(x[1],x[2],x[3],round(x[4]['min_pf'],3),round(x[4]['ev'],3)) for x in selected[hour_num]], flush=True)

    heaps = {1:[],2:[]}
    for hour_num in [1,2]:
        for _, mode_name, filt, dayp, _ in selected[hour_num]:
            mode = 0 if mode_name == 'breakout' else 1
            for ap in ATR_PERIODS:
                sig = signal_indices(df, hour_num, filt, dayp, ap)
                if len(sig) < 45:
                    continue
                atr = df[f'atr{ap}'].to_numpy(float)
                expiries = EXPIRY_GRID if mode == 0 else [0]
                for expiry in expiries:
                    ex = expiry if mode == 0 else 1
                    trig, dirs, ent, at, yrs = build_trades(sig, direction, high, low, open_, atr, years, ex, mode)
                    if len(trig) < 45:
                        continue
                    for sl in SL_GRID:
                        for tp in TP_GRID:
                            for hold in HOLD_GRID:
                                rs = simulate_rs(trig, dirs, ent, at, high, low, close, sl, tp, hold)
                                m = metrics(rs, yrs, span_years)
                                if m is None or m['n'] < 60:
                                    continue
                                row = dict(m)
                                row.update({
                                    'symbol':symbol,'hour':hour_num,'entry_mode':mode_name,'filter':filt,
                                    'day_policy':dayp,'atr_period':ap,'sl_atr':sl,'tp_atr':tp,
                                    'rr':tp/sl,'expiry_h':expiry if mode==0 else 0,'hold_h':hold,
                                    'data_start':str(start.date()),'data_end':str(end.date())
                                })
                                push_top(heaps[hour_num], row)

    out = []
    for hour_num in [1,2]:
        rows = [x[2] for x in sorted(heaps[hour_num], reverse=True)]
        for rank,row in enumerate(rows,1):
            row['rank'] = rank
            out.append(row)
        if rows:
            b=rows[0]
            print('BEST',symbol,'H'+str(hour_num),json.dumps(b, sort_keys=True), flush=True)
    return out


def main():
    all_rows=[]
    for s in SYMBOLS:
        try:
            all_rows.extend(optimise_symbol(s))
        except Exception as e:
            print('ERROR',s,repr(e),flush=True)
    res=pd.DataFrame(all_rows)
    if len(res):
        cols=['symbol','hour','rank','entry_mode','filter','day_policy','atr_period','sl_atr','tp_atr','rr','expiry_h','hold_h','n','wr','pf','ev','sumR','ddR','trades_year','old_n','old_pf','old_ev','mid_n','mid_pf','mid_ev','recent_n','recent_pf','recent_ev','min_pf','min_ev','robust','data_start','data_end']
        res=res[[c for c in cols if c in res.columns]]
        res.to_csv('pair_optimisation_top.csv',index=False)
        best=res[res['rank']==1].copy()
        best.to_csv('pair_optimisation_best.csv',index=False)
        print('\n===== FINAL BEST =====')
        print(best.to_string(index=False))
    else:
        Path('pair_optimisation_top.csv').write_text('')
        Path('pair_optimisation_best.csv').write_text('')

if __name__=='__main__':
    main()
