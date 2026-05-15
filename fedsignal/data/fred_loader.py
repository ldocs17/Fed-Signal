import os
import pickle

import numpy as np
import pandas as pd

from fedsignal.config import FRED_KEY, FRED_CACHE, FORCE_REFRESH, CACHE_DIR


def load_fred_data():
    if not FORCE_REFRESH and os.path.exists(FRED_CACHE):
        print(f'Loading FRED data from cache: {FRED_CACHE}')
        with open(FRED_CACHE, 'rb') as f:
            cache = pickle.load(f)
        print(f'Loaded {len(cache)} series from cache')
        return cache

    print('Downloading from FRED API...')
    from fredapi import Fred
    fred = Fred(api_key=FRED_KEY)
    cache = {
        'gdp_growth':        fred.get_series('A191RL1Q225SBEA'),
        'cpi':               fred.get_series('CPIAUCSL'),
        'core_pce':          fred.get_series('PCEPILFE'),
        'unemployment':      fred.get_series('UNRATE'),
        'nonfarm_payrolls':  fred.get_series('PAYEMS'),
        'ten_year':          fred.get_series('DGS10'),
        'two_year':          fred.get_series('DGS2'),
        'three_month':       fred.get_series('DGS3MO'),
        'thirty_year':       fred.get_series('DGS30'),
        'fed_funds':         fred.get_series('DFF'),
        'one_year_daily':    fred.get_series('DGS1'),
        'ten_two_spread':    fred.get_series('T10Y2Y'),
        'breakeven_10y':     fred.get_series('T10YIE'),
        'umich_inflation_exp': fred.get_series('MICH'),
        'chicago_fed_nfci':  fred.get_series('NFCI'),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(FRED_CACHE, 'wb') as f:
        pickle.dump(cache, f)
    print(f'Saved {len(cache)} FRED series to {FRED_CACHE}')
    return cache


def classify_economic_regime(row):
    gdp = row['GDP Growth']
    inflation = row['core_pce_yoy']
    if gdp < 0 and inflation > 3.0:
        return 'stagflation'
    elif gdp < 0 and inflation < 2.0:
        return 'deflationary_recession'
    elif gdp > 3.0 and inflation > 3.0:
        return 'overheating'
    elif gdp > 1.5 and inflation < 2.5:
        return 'goldilocks'
    elif gdp < 1.5 and inflation < 2.0:
        return 'secular_stagnation'
    elif gdp > 1.5 and inflation > 2.5:
        return 'reflation'
    else:
        return 'transition'


def build_regime_data(fred_cache):
    two_year   = fred_cache['two_year']
    fed_funds  = fred_cache['fed_funds']
    core_pce   = fred_cache['core_pce']

    twoy_ff_spread = two_year - fed_funds
    core_pce_yoy   = core_pce.pct_change(periods=12) * 100

    regime_data = pd.DataFrame({
        'GDP Growth':    fred_cache['gdp_growth'],
        'unemployment':  fred_cache['unemployment'],
        'cpi':           fred_cache['cpi'],
        'core_pce':      core_pce,
        'fed_funds':     fed_funds,
        '10y_yield':     fred_cache['ten_year'],
        '2y_yield':      two_year,
        '10y_2y_spread': fred_cache['ten_two_spread'],
        'twoy_ff_spread': twoy_ff_spread,
        'breakeven_10y': fred_cache['breakeven_10y'],
        'nfci':          fred_cache['chicago_fed_nfci'],
        'core_pce_yoy':  core_pce_yoy,
    })
    regime_monthly = regime_data.resample('ME').last().ffill()

    regime_monthly['combined_regime'] = None
    for date, row in regime_monthly.dropna(subset=['GDP Growth', 'core_pce_yoy']).iterrows():
        try:
            regime_monthly.loc[date, 'combined_regime'] = classify_economic_regime(row)
        except Exception:
            pass

    print('Regime distribution:')
    print(regime_monthly['combined_regime'].value_counts())
    return regime_monthly
