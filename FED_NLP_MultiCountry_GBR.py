# %% [markdown]
# # newFedSignal — US GBR Model
# **Countries:** United States (Fed)
# 
# **US Model**: GradientBoostingRegressor with NLP + Regime Features (Walk-Forward CV)  

# **Target**: Short-term yield change, ~60 trading days after central bank meeting
# 
# **Key upgrades from original multi-country notebook:**
# - US: GBR replaces OLS, walk-forward replaces LOO, 7 features instead of 2
# - US: Regime classification + feature engineering (nlp_vs_regime, momentum, dispersion)
# - US: FRED data caching + FOMC statement caching
# - AU/UK: Kept as-is (LinearRegression + LOO) — too few observations for GBR
# %%
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import os, pickle, json

random.seed(24)
np.random.seed(24)
torch.manual_seed(24)

FRED_KEY = 'FRED_API'
device = torch.device('cpu')
MODEL_NAME = 'distilbert-base-uncased'

print("Imports complete.")
# %% [markdown]
# ---
# ## US — Federal Reserve (GBR + Walk-Forward)
# %%
# ══════════════════════════════════════════════════════════════════════════════
# US FRED DATA (cached) + REGIME CLASSIFICATION + FOMC SCRAPING (cached)
# ══════════════════════════════════════════════════════════════════════════════
import requests
from bs4 import BeautifulSoup
import time

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath('.')), 'data')
os.makedirs(CACHE_DIR, exist_ok=True)
FRED_CACHE = os.path.join(CACHE_DIR, 'fred_data.pkl')
STATEMENTS_CACHE = os.path.join(CACHE_DIR, 'fomc_statements.json')
FORCE_REFRESH = False

if not FORCE_REFRESH and os.path.exists(FRED_CACHE):
    print(f'Loading FRED data from cache: {FRED_CACHE}')
    with open(FRED_CACHE, 'rb') as f:
        fred_cache = pickle.load(f)
    gdp_growth = fred_cache['gdp_growth']
    cpi = fred_cache['cpi']
    core_pce = fred_cache['core_pce']
    unemployment = fred_cache['unemployment']
    nonfarm_payrolls = fred_cache['nonfarm_payrolls']
    ten_year = fred_cache['ten_year']
    two_year = fred_cache['two_year']
    three_month = fred_cache['three_month']
    thirty_year = fred_cache['thirty_year']
    fed_funds = fred_cache['fed_funds']
    one_year_daily = fred_cache['one_year_daily']
    ten_two_spread = fred_cache['ten_two_spread']
    breakeven_10y = fred_cache['breakeven_10y']
    umich_inflation_exp = fred_cache['umich_inflation_exp']
    chicago_fed_nfci = fred_cache['chicago_fed_nfci']
    print(f'Loaded {len(fred_cache)} series from cache')
else:
    print('Downloading from FRED API...')
    from fredapi import Fred
    fred = Fred(api_key=FRED_KEY)
    gdp_growth = fred.get_series('A191RL1Q225SBEA')
    cpi = fred.get_series('CPIAUCSL')
    core_pce = fred.get_series('PCEPILFE')
    unemployment = fred.get_series('UNRATE')
    nonfarm_payrolls = fred.get_series('PAYEMS')
    ten_year = fred.get_series('DGS10')
    two_year = fred.get_series('DGS2')
    three_month = fred.get_series('DGS3MO')
    thirty_year = fred.get_series('DGS30')
    fed_funds = fred.get_series('DFF')
    one_year_daily = fred.get_series('DGS1')
    ten_two_spread = fred.get_series('T10Y2Y')
    breakeven_10y = fred.get_series('T10YIE')
    umich_inflation_exp = fred.get_series('MICH')
    chicago_fed_nfci = fred.get_series('NFCI')
    fred_cache = {
        'gdp_growth': gdp_growth, 'cpi': cpi, 'core_pce': core_pce,
        'unemployment': unemployment, 'nonfarm_payrolls': nonfarm_payrolls,
        'ten_year': ten_year, 'two_year': two_year, 'three_month': three_month,
        'thirty_year': thirty_year, 'fed_funds': fed_funds,
        'one_year_daily': one_year_daily, 'ten_two_spread': ten_two_spread,
        'breakeven_10y': breakeven_10y, 'umich_inflation_exp': umich_inflation_exp,
        'chicago_fed_nfci': chicago_fed_nfci,
    }
    with open(FRED_CACHE, 'wb') as f:
        pickle.dump(fred_cache, f)
    print(f'Saved {len(fred_cache)} FRED series to {FRED_CACHE}')

twoy_ff_spread = two_year - fed_funds
core_pce_yoy = core_pce.pct_change(periods=12) * 100

regime_data = pd.DataFrame({
    'GDP Growth': gdp_growth, 'unemployment': unemployment, 'cpi': cpi,
    'core_pce': core_pce, 'fed_funds': fed_funds, '10y_yield': ten_year,
    '2y_yield': two_year, '10y_2y_spread': ten_two_spread,
    'twoy_ff_spread': twoy_ff_spread, 'breakeven_10y': breakeven_10y,
    'nfci': chicago_fed_nfci, 'core_pce_yoy': core_pce_yoy,
})
regime_monthly = regime_data.resample('ME').last().ffill()

def classify_economic_regime(row):
    gdp = row['GDP Growth']
    inflation = row['core_pce_yoy']
    if gdp < 0 and inflation > 3.0: return 'stagflation'
    elif gdp < 0 and inflation < 2.0: return 'deflationary_recession'
    elif gdp > 3.0 and inflation > 3.0: return 'overheating'
    elif gdp > 1.5 and inflation < 2.5: return 'goldilocks'
    elif gdp < 1.5 and inflation < 2.0: return 'secular_stagnation'
    elif gdp > 1.5 and inflation > 2.5: return 'reflation'
    else: return 'transition'

regime_monthly['combined_regime'] = None
for date, row in regime_monthly.dropna(subset=['GDP Growth', 'core_pce_yoy']).iterrows():
    try:
        regime_monthly.loc[date, 'combined_regime'] = classify_economic_regime(row)
    except:
        pass

print('Regime distribution:')
print(regime_monthly['combined_regime'].value_counts())

# FOMC Statement Scraping (cached)
if not FORCE_REFRESH and os.path.exists(STATEMENTS_CACHE):
    print(f'\nLoading FOMC statements from cache: {STATEMENTS_CACHE}')
    with open(STATEMENTS_CACHE, 'r', encoding='utf-8') as f:
        statements = json.load(f)
    print(f'Loaded {len(statements)} cached statements')
else:
    print('\nScraping FOMC statements from federalreserve.gov...')
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
        '20210127'
    ]
    historical_years = list(range(2020, 2014, -1))
    for year in historical_years:
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
    with open(STATEMENTS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(statements, f, ensure_ascii=False)
    print(f'Saved {len(statements)} statements to {STATEMENTS_CACHE}')

print(f'Statements available: {len(statements)}')
# %%
# ══════════════════════════════════════════════════════════════════════════════
# US BUILD fomc_df + TARGET
# ══════════════════════════════════════════════════════════════════════════════

fomc_data = []
for date_str, text in statements.items():
    date = pd.to_datetime(date_str)
    regime_candidates = regime_monthly.index[regime_monthly.index < date]
    if len(regime_candidates) == 0:
        continue
    regime_date = regime_candidates[-1]
    regime_row = regime_monthly.loc[regime_date]
    fomc_data.append({
        'date': date, 'statement': text,
        'combined_regime': regime_row['combined_regime'],
        'GDP Growth': regime_row['GDP Growth'],
        'core_pce_yoy': regime_row['core_pce_yoy'],
        'unemployment': regime_row['unemployment'],
        'fed_funds': regime_row['fed_funds'],
        '10y_yield': regime_row['10y_yield'],
        '2y_yield': regime_row['2y_yield'],
        'twoy_ff_spread': regime_row['twoy_ff_spread'],
        '10y_2y_spread': regime_row['10y_2y_spread'],
        'breakeven_10y': regime_row['breakeven_10y'],
        'nfci': regime_row['nfci'],
    })

fomc_df = pd.DataFrame(fomc_data).sort_values('date').set_index('date')

HORIZON = 60
fomc_df['1y_60d_change'] = None
for date in fomc_df.index:
    try:
        loc = one_year_daily.index.get_indexer([date], method='ffill')[0]
        if loc + HORIZON < len(one_year_daily):
            fomc_df.loc[date, '1y_60d_change'] = one_year_daily.iloc[loc + HORIZON] - one_year_daily.iloc[loc]
    except:
        pass
fomc_df['1y_60d_change'] = fomc_df['1y_60d_change'].astype(float)

fomc_df['1y_yield'] = None
for date in fomc_df.index:
    try:
        loc = one_year_daily.index.get_indexer([date], method='ffill')[0]
        fomc_df.loc[date, '1y_yield'] = one_year_daily.iloc[loc]
    except:
        pass
fomc_df['1y_yield'] = fomc_df['1y_yield'].astype(float)

print(f'FOMC meetings: {len(fomc_df)}')
print(f'Target available: {fomc_df["1y_60d_change"].notna().sum()}')
print(f'Target horizon: {HORIZON} trading days from meeting-day close')
print(f'\nRegime distribution:')
print(fomc_df['combined_regime'].value_counts())
# %% [markdown]
# ### US Training Sentences (88)
# %%
# ══════════════════════════════════════════════════════════════════════════════
# US NLP — Fine-Tune DistilBERT (30 epochs, 88 sentences) + Score Statements
# ══════════════════════════════════════════════════════════════════════════════

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sent_model_us = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
sent_model_us = sent_model_us.to(device)

sentence_training_US = [
    ("The Committee decided to raise the target range for the federal funds rate by 75 basis points", 0.95),
    ("The Committee decided to raise the target range for the federal funds rate by 50 basis points", 0.85),
    ("The Committee decided to raise the target range for the federal funds rate by 25 basis points", 0.75),
    ("The Committee decided to maintain the target range for the federal funds rate", 0.5),
    ("The Committee decided to lower the target range for the federal funds rate by 25 basis points", 0.25),
    ("The Committee decided to lower the target range for the federal funds rate by 50 basis points", 0.15),
    ("The Committee expects it will soon be appropriate to raise the target range for the federal funds rate", 0.85),
    ("It would be premature to begin lowering rates at this time", 0.8),
    ("The Committee anticipates that ongoing increases in the target range will be appropriate", 0.9),
    ("The Committee is strongly committed to returning inflation to its 2 percent objective", 0.8),
    ("The Committee is prepared to raise rates further if appropriate", 0.85),
    ("Additional policy firming may be appropriate", 0.8),
    ("The Committee will continue reducing its holdings of Treasury securities and agency debt", 0.75),
    ("The Committee decided to continue to reduce the monthly pace of its net asset purchases", 0.7),
    ("Ongoing increases in the target range will be appropriate to bring inflation down", 0.85),
    ("The Committee will take into account the cumulative tightening of monetary policy", 0.65),
    ("The Committee is prepared to adjust the stance of monetary policy as appropriate if risks emerge", 0.35),
    ("The Committee would be prepared to adjust the stance of monetary policy as appropriate", 0.4),
    ("The Committee judges that the risks have become more balanced", 0.3),
    ("The risks to achieving its employment and inflation goals are roughly in balance", 0.3),
    ("The Committee is attentive to the risks to both sides of its dual mandate", 0.35),
    ("In considering additional adjustments the Committee will carefully assess incoming data", 0.4),
    ("The Committee does not expect it will be appropriate to reduce the target range until it has gained greater confidence", 0.55),
    ("Inflation remains elevated", 0.8),
    ("Inflation remains well above the Committee's 2 percent objective", 0.85),
    ("Inflation remains somewhat elevated", 0.65),
    ("Inflation continues to run well above the longer-run goal", 0.85),
    ("Core inflation remains elevated and persistent", 0.85),
    ("Price pressures remain broad based across sectors", 0.8),
    ("Inflation has shown little progress toward the 2 percent objective", 0.85),
    ("Supply and demand imbalances have continued to contribute to elevated levels of inflation", 0.75),
    ("Wage growth remains elevated relative to what would be consistent with price stability", 0.8),
    ("Housing costs continue to put upward pressure on overall inflation", 0.75),
    ("Inflation expectations risk becoming unanchored", 0.9),
    ("Inflation has eased over the past year but remains above the Committee's target", 0.55),
    ("Inflation has made progress toward the Committee's 2 percent objective", 0.3),
    ("Inflation has made further progress toward the Committee's 2 percent objective", 0.25),
    ("Inflation has declined significantly from its peak", 0.2),
    ("The disinflationary process has continued", 0.2),
    ("Inflation has come down considerably over the past year", 0.25),
    ("Longer-term inflation expectations remain well anchored", 0.35),
    ("Inflation has moved closer to the Committee's 2 percent objective", 0.3),
    ("The labor market remains extremely tight", 0.8),
    ("Job gains have been robust in recent months", 0.75),
    ("Job gains have been strong", 0.7),
    ("The unemployment rate has remained low", 0.65),
    ("Demand continues to outstrip supply in the labor market", 0.8),
    ("Consumer spending has been particularly strong", 0.75),
    ("Consumer spending has been surprisingly resilient", 0.7),
    ("Job gains have been solid in recent months and the unemployment rate has declined substantially", 0.7),
    ("Job gains have moderated", 0.35),
    ("Job gains have slowed", 0.3),
    ("The unemployment rate has moved up but remains low", 0.35),
    ("The labor market is showing signs of softening", 0.25),
    ("Job gains have remained low and the unemployment rate has shown some signs of stabilization", 0.35),
    ("Employment gains have slowed significantly", 0.2),
    ("Economic activity has been expanding at a strong pace", 0.75),
    ("Economic activity has been expanding at a solid pace", 0.65),
    ("Available indicators suggest that economic activity has been expanding at a solid pace", 0.65),
    ("The economy has been stronger than expected", 0.75),
    ("Recent indicators suggest that economic activity has continued to expand at a solid pace", 0.65),
    ("Indicators of economic activity and employment have continued to strengthen", 0.7),
    ("Economic activity has slowed from its strong pace", 0.35),
    ("Economic activity has slowed considerably", 0.2),
    ("Growth has moderated from its earlier pace", 0.3),
    ("Recent indicators point to modest growth in spending and production", 0.35),
    ("The path of the economy continues to depend on the course of the virus", 0.3),
    ("Economic activity expanded at a moderate rate", 0.4),
    ("Overall financial conditions remain accommodative", 0.3),
    ("Financial conditions have tightened significantly", 0.25),
    ("Credit conditions have become more restrictive for households and businesses", 0.25),
    ("Tighter credit conditions for households and businesses are likely to weigh on economic activity", 0.2),
    ("The Federal Reserve's ongoing purchases will continue to foster smooth market functioning and accommodative financial conditions", 0.3),
    ("The Committee will continue reducing its holdings of Treasury securities", 0.7),
    ("The Committee decided to begin reducing its holdings at a pace of 95 billion per month", 0.75),
    ("The Committee decided to slow the pace of decline of its securities holdings", 0.35),
    ("Beginning in February the Committee will increase its holdings of Treasury securities", 0.2),
    ("The Committee will increase its holdings of agency mortgage-backed securities", 0.2),
    ("Risks to the economic outlook remain including from new variants of the virus", 0.25),
    ("Uncertainty about the economic outlook has increased", 0.3),
    ("Uncertainty about the economic outlook remains elevated", 0.35),
    ("The Committee sees the risks to achieving its goals as roughly in balance", 0.4),
    ("Risks to the economic outlook are weighted to the downside", 0.15),
    ("Upside risks to inflation remain", 0.8),
    ("The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run", 0.5),
    ("The Committee will continue to monitor the implications of incoming information for the economic outlook", 0.5),
    ("The Committee's assessments will take into account a wide range of information", 0.5),
    ("In assessing the appropriate stance of monetary policy the Committee will continue to monitor", 0.5),
]

texts = [s[0] for s in sentence_training_US]
scores = [s[1] for s in sentence_training_US]
print(f'US training sentences: {len(texts)}')

class SentDataset(Dataset):
    def __init__(self, texts, scores):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
        self.labels = torch.tensor(scores, dtype=torch.float32)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

dataset = SentDataset(texts, scores)
training_args = TrainingArguments(
    output_dir='./fed_sentence_model', num_train_epochs=30,
    per_device_train_batch_size=8, learning_rate=2e-5,
    logging_steps=50, save_strategy='no', use_cpu=True
)
trainer = Trainer(model=sent_model_us, args=training_args, train_dataset=dataset)
trainer.train()
print('US DistilBERT training complete')

skip_patterns_us = [
    'Voting for', 'Voting against', 'Vice Chair', 'Chair;',
    'media inquiries', 'email', 'Implementation Note',
    'For release', 'press@', 'call 202',
    'Brainard', 'Bullard', 'Bowman', 'Harker', 'Mester',
    'Kashkari', 'George', 'Kaplan', 'Clarida', 'Waller',
    'Jefferson', 'Cook', 'Hammack', 'Logan', 'Paulson',
    'Miran', 'Barr', 'Williams', 'Powell'
]

def score_statement(text, model=None, skip_patterns=None):
    if model is None:
        model = sent_model_us
    if skip_patterns is None:
        skip_patterns = skip_patterns_us
    sentences = text.split('.')
    sent_scores = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or any(skip in sent for skip in skip_patterns):
            continue
        inputs = tokenizer(sent, return_tensors='pt', truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        sent_scores.append(out.logits.cpu().numpy()[0][0])
    if not sent_scores:
        return 0.5, 0.0
    return np.mean(sent_scores), np.std(sent_scores)

print('\nScoring FOMC statements...')
for date, row in fomc_df.iterrows():
    if pd.notna(row.get('statement')):
        mean_s, disp_s = score_statement(row['statement'])
        fomc_df.loc[date, 'sent_level'] = mean_s
        fomc_df.loc[date, 'sent_dispersion'] = disp_s

fomc_df['sent_level'] = fomc_df['sent_level'].astype(float)
fomc_df['sent_dispersion'] = fomc_df['sent_dispersion'].astype(float)
fomc_df['sent_level_demeaned'] = fomc_df['sent_level'] - fomc_df['sent_level'].mean()

print(f'NLP scores range: [{fomc_df["sent_level"].min():.3f}, {fomc_df["sent_level"].max():.3f}]')
print(f'NLP dispersion range: [{fomc_df["sent_dispersion"].min():.3f}, {fomc_df["sent_dispersion"].max():.3f}]')
# %%
# ══════════════════════════════════════════════════════════════════════════════
# US FEATURE ENGINEERING + GBR WALK-FORWARD EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

# Regime ordinal encoding
REGIME_ORDINAL = {
    'deflationary_recession': 0, 'secular_stagnation': 1,
    'goldilocks': 2, 'transition': 3, 'reflation': 4,
    'overheating': 5, 'stagflation': 6,
}
fomc_df['regime_ordinal'] = fomc_df['combined_regime'].map(REGIME_ORDINAL)

# NLP vs regime expected score
REGIME_EXPECTED_SCORE = {
    'deflationary_recession': 0.20, 'secular_stagnation': 0.30,
    'goldilocks': 0.45, 'transition': 0.50, 'reflation': 0.62,
    'overheating': 0.78, 'stagflation': 0.72,
}
fomc_df['regime_expected_nlp'] = fomc_df['combined_regime'].map(REGIME_EXPECTED_SCORE)
fomc_df['nlp_vs_regime'] = fomc_df['sent_level'] - fomc_df['regime_expected_nlp']

# NLP momentum (EWMA)
fomc_df = fomc_df.sort_index()
fomc_df['nlp_ewma'] = fomc_df['sent_level'].ewm(span=4, adjust=False).mean()
fomc_df['nlp_momentum'] = fomc_df['sent_level'] - fomc_df['nlp_ewma']

# Final feature list
US_FEATURES = [
    'twoy_ff_spread',     # 2Y-FF spread (macro)
    'core_pce_yoy',       # Core PCE year-over-year (inflation)
    'sent_level_demeaned', # NLP hawkishness (demeaned)
    'nlp_vs_regime',      # NLP surprise vs regime expectation
    'nlp_momentum',       # NLP momentum (deviation from EWMA)
    'sent_dispersion',    # Sentence-level score dispersion
    'regime_ordinal',     # Economic regime (ordinal)
]

TARGET = '1y_60d_change'

# Prepare final dataset
model_df = fomc_df[US_FEATURES + [TARGET]].dropna().sort_index()
X_us = model_df[US_FEATURES]
y_us = model_df[TARGET]

print(f'US final dataset: {len(model_df)} observations')
print(f'Features: {US_FEATURES}')
print(f'\nFeature correlations with target:')
for feat in US_FEATURES:
    corr = model_df[feat].corr(model_df[TARGET])
    print(f'  {feat:<25} {corr:+.3f}')

# ── GBR Walk-Forward ──
us_model = Pipeline([
    ('scaler', StandardScaler()),
    ('gbr', GradientBoostingRegressor(
        n_estimators=80, max_depth=2, min_samples_leaf=5,
        learning_rate=0.05, subsample=0.8, random_state=42,
    ))
])

MIN_TRAIN = 20
X_arr, y_arr = X_us.values, y_us.values
us_preds, us_acts, train_sizes = [], [], []

for t in range(MIN_TRAIN, len(y_arr)):
    m = clone(us_model)
    m.fit(X_arr[:t], y_arr[:t])
    us_preds.append(m.predict(X_arr[t:t+1])[0])
    us_acts.append(y_arr[t])
    train_sizes.append(t)

us_preds = np.array(us_preds)
us_acts = np.array(us_acts)

# Metrics
ss_res = np.sum((us_acts - us_preds) ** 2)
ss_tot = np.sum((us_acts - us_acts.mean()) ** 2)
us_oos_r2 = 1 - ss_res / ss_tot
us_dir_acc = np.mean(np.sign(us_preds) == np.sign(us_acts)) * 100
us_mae = np.mean(np.abs(us_acts - us_preds))

# Spike metrics
spike_mask = np.abs(us_acts) > 0.5
n_spikes = spike_mask.sum()
spike_dir = np.mean(np.sign(us_preds[spike_mask]) == np.sign(us_acts[spike_mask])) * 100 if n_spikes >= 3 else float('nan')
spike_capture = np.mean(np.abs(us_preds[spike_mask]) / np.abs(us_acts[spike_mask])) * 100 if n_spikes >= 3 else float('nan')

print(f'\n{"="*60}')
print(f'US MODEL PERFORMANCE (GBR + Walk-Forward)')
print(f'{"="*60}')
print(f'Model:           GradientBoostingRegressor')
print(f'Features:        {len(US_FEATURES)}')
print(f'Observations:    {len(model_df)}')
print(f'Test predictions: {len(us_preds)} (walk-forward, min_train={MIN_TRAIN})')
print(f'')
print(f'OOS R2:          {us_oos_r2:+.3f}')
print(f'Dir Accuracy:    {us_dir_acc:.1f}%')
print(f'MAE:             {us_mae:.3f}')
print(f'')
print(f'Spikes (|change|>0.5%): {n_spikes}')
print(f'Spike Dir Acc:   {spike_dir:.1f}%')
print(f'Spike Capture:   {spike_capture:.0f}%')
print(f'{"="*60}')

# Feature importance
m_full = clone(us_model)
m_full.fit(X_arr, y_arr)
importances = m_full.named_steps['gbr'].feature_importances_

print(f'\nFeature Importance (Gini):')
for feat, imp in sorted(zip(US_FEATURES, importances), key=lambda x: -x[1]):
    bar = '|' * int(imp * 80)
    print(f'  {feat:<25} {imp:.3f} {bar}')

# Regime-stratified accuracy
test_idx = model_df.index[MIN_TRAIN:]
test_regimes = fomc_df.loc[test_idx, 'combined_regime'].values[:len(us_preds)]

print(f'\nRegime-Stratified Directional Accuracy:')
print(f'{"Regime":<25} {"N":>5} {"Dir Acc":>10}')
print('-' * 45)
for regime in sorted(set(test_regimes)):
    mask = np.array(test_regimes) == regime
    if mask.sum() >= 3:
        acc = np.mean(np.sign(us_preds[mask]) == np.sign(us_acts[mask])) * 100
        print(f'{regime:<25} {mask.sum():>5} {acc:>9.1f}%')

# Store US results for summary
us_results = {
    'GBR + NLP + Regime': {
        'preds': us_preds, 'acts': us_acts,
        'oos_r2': us_oos_r2, 'dir_acc': us_dir_acc / 100,
        'n': len(model_df),
    }
}
