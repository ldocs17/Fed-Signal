import json
import os
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from fedsignal.config import STATEMENTS_CACHE, CACHE_DIR, FORCE_REFRESH, HORIZON


def load_fomc_statements():
    if not FORCE_REFRESH and os.path.exists(STATEMENTS_CACHE):
        print(f'Loading FOMC statements from cache: {STATEMENTS_CACHE}')
        with open(STATEMENTS_CACHE, 'r', encoding='utf-8') as f:
            statements = json.load(f)
        print(f'Loaded {len(statements)} cached statements')
        return statements

    print('Scraping FOMC statements from federalreserve.gov...')
    base_url = 'https://www.federalreserve.gov'
    meeting_dates = [
        '20260128', '20251210', '20251029', '20250917',
        '20250730', '20250618', '20250507', '20250319',
        '20250129', '20241218', '20241107', '20240918',
        '20240731', '20240612', '20240501', '20240320',
        '20240131', '20231213', '20231101', '20230920',
        '20230726', '20230614', '20230503', '20230322',
        '20230201', '20221214', '20221102', '20220921',
        '20220727', '20220615', '20220504', '20220316',
        '20220126', '20211215', '20211103', '20210922',
        '20210728', '20210616', '20210428', '20210317',
        '20210127',
    ]
    for year in range(2020, 2014, -1):
        url = f'{base_url}/monetarypolicy/fomchistorical{year}.htm'
        resp = requests.get(url)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'pressreleases/monetary' in href and href.endswith('a.htm'):
                    date = href.split('monetary')[1].replace('a.htm', '')
                    if len(date) == 8 and date.isdigit() and date not in meeting_dates:
                        meeting_dates.append(date)
        time.sleep(1)

    meeting_dates.sort()
    print(f'Total meeting dates: {len(meeting_dates)}')

    statements = {}
    for date in meeting_dates:
        stmt_url = f'{base_url}/newsevents/pressreleases/monetary{date}a.htm'
        resp = requests.get(stmt_url)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            article = (
                soup.find('div', class_='col-xs-12 col-sm-8 col-md-8') or
                soup.find('div', class_='col-xs-12 col-sm-8 col-md-9') or
                soup.find('article')
            )
            if article:
                statements[date] = article.get_text(strip=True)
        time.sleep(1)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(STATEMENTS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(statements, f, ensure_ascii=False)
    print(f'Saved {len(statements)} statements to {STATEMENTS_CACHE}')
    return statements


def build_fomc_df(statements, regime_monthly, one_year_daily):
    fomc_data = []
    for date_str, text in statements.items():
        date = pd.to_datetime(date_str)
        regime_candidates = regime_monthly.index[regime_monthly.index < date]
        if len(regime_candidates) == 0:
            continue
        regime_row = regime_monthly.loc[regime_candidates[-1]]
        fomc_data.append({
            'date':           date,
            'statement':      text,
            'combined_regime': regime_row['combined_regime'],
            'GDP Growth':     regime_row['GDP Growth'],
            'core_pce_yoy':   regime_row['core_pce_yoy'],
            'unemployment':   regime_row['unemployment'],
            'fed_funds':      regime_row['fed_funds'],
            '10y_yield':      regime_row['10y_yield'],
            '2y_yield':       regime_row['2y_yield'],
            'twoy_ff_spread': regime_row['twoy_ff_spread'],
            '10y_2y_spread':  regime_row['10y_2y_spread'],
            'breakeven_10y':  regime_row['breakeven_10y'],
            'nfci':           regime_row['nfci'],
        })

    fomc_df = pd.DataFrame(fomc_data).sort_values('date').set_index('date')

    fomc_df['1y_60d_change'] = None
    fomc_df['1y_yield'] = None
    for date in fomc_df.index:
        try:
            loc = one_year_daily.index.get_indexer([date], method='ffill')[0]
            if loc + HORIZON < len(one_year_daily):
                fomc_df.loc[date, '1y_60d_change'] = (
                    one_year_daily.iloc[loc + HORIZON] - one_year_daily.iloc[loc]
                )
            fomc_df.loc[date, '1y_yield'] = one_year_daily.iloc[loc]
        except Exception:
            pass

    fomc_df['1y_60d_change'] = fomc_df['1y_60d_change'].astype(float)
    fomc_df['1y_yield'] = fomc_df['1y_yield'].astype(float)

    print(f'FOMC meetings: {len(fomc_df)}')
    print(f'Target available: {fomc_df["1y_60d_change"].notna().sum()}')
    print(f'Target horizon: {HORIZON} trading days from meeting-day close')
    print(f'\nRegime distribution:')
    print(fomc_df['combined_regime'].value_counts())
    return fomc_df
