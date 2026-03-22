# %% [markdown]
# # newFedSignal — Multi-Country GBR Model
# **Countries:** United States (Fed) · Australia (RBA) · United Kingdom (BoE)
# 
# **US Model**: GradientBoostingRegressor with NLP + Regime Features (Walk-Forward CV)  
# **AU/UK Models**: LinearRegression with NLP + Spread (LOO CV — limited data)  
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
# %% [markdown]
# ---
# ## Australia (RBA) & United Kingdom (BoE)
# %%
# =============================================================================
# RBA SCRAPER
# =============================================================================
def scrape_rba():
    print('\n' + '='*60)
    print('  SCRAPING: RESERVE BANK OF AUSTRALIA')
    print('='*60)

    base_url = 'https://www.rba.gov.au'
    stmts = {}

    for year in range(2015, 2027):
        index_url = f'{base_url}/media-releases/{year}/'
        print(f'  Scanning RBA {year}...')
        resp = requests.get(index_url)
        if resp.status_code != 200:
            continue

        soup = BeautifulSoup(resp.text, 'html.parser')
        links = soup.find_all('a', href=True)

        for link in links:
            href = link['href']
            text = link.get_text(strip=True).lower()
            is_mp = ('monetary policy decision' in text or 'monetary policy' in text and 'mr-' in href)
            if not is_mp:
                continue
            if not href.startswith('http'):
                href = base_url + href

            date_key = href.split('/')[-1].replace('.html', '')
            if date_key in stmts:
                continue

            try:
                resp2 = requests.get(href)
                if resp2.status_code == 200:
                    soup2 = BeautifulSoup(resp2.text, 'html.parser')
                    content = (
                        soup2.find('div', id='content') or
                        soup2.find('div', class_='rba-content') or
                        soup2.find('article') or soup2.find('main')
                    )
                    if content:
                        stmts[date_key] = content.get_text(separator=' ', strip=True)
                        print(f'    Statement: {date_key}')
            except Exception as e:
                print(f'    Error: {e}')
            time.sleep(1)
        time.sleep(1)

    print(f'\n  RBA statements: {len(stmts)}')
    return stmts


rba_statements = scrape_rba()
# %%
import requests
from bs4 import BeautifulSoup
import time
import json
import re

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-GB,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}

def extract_summary(soup):
    """Extract just the Monetary Policy Summary, not the full minutes."""
    summary_text = ''

    # METHOD 1: Find "Monetary Policy Summary" heading
    headings = soup.find_all(['h2', 'h3'])
    summary_start = None
    minutes_start = None

    for h in headings:
        h_text = h.get_text(strip=True).lower()
        if 'monetary policy summary' in h_text:
            summary_start = h
        if 'minutes of the monetary policy' in h_text:
            minutes_start = h

    if summary_start:
        parts = []
        for sibling in summary_start.find_next_siblings():
            if sibling == minutes_start:
                break
            sibling_text = sibling.get_text(strip=True)
            if sibling.name in ['h2', 'h3'] and 'minutes' in sibling_text.lower():
                break
            if re.match(r'^1[:\.]?\s', sibling_text):
                break
            if len(sibling_text) > 10:
                parts.append(sibling_text)
        summary_text = ' '.join(parts)

    # METHOD 2: No heading found, grab paragraphs before "1:"
    if not summary_text:
        content = soup.find(id='main-content') or soup.find('main')
        if content:
            paragraphs = content.find_all('p')
            parts = []
            for p in paragraphs:
                p_text = p.get_text(strip=True)
                if re.match(r'^1[:\.]?\s', p_text):
                    break
                if len(p_text) < 20:
                    continue
                if 'PDF' in p_text and 'MB' in p_text:
                    continue
                parts.append(p_text)
            summary_text = ' '.join(parts)

    return summary_text


def scrape_boe():
    print('\n' + '='*60)
    print('  SCRAPING: BANK OF ENGLAND (Summary only)')
    print('='*60)

    base_url = 'https://www.bankofengland.co.uk'
    session = requests.Session()
    session.headers.update(HEADERS)

    boe_meetings = []
    for year in range(2015, 2027):
        months = ['january','february','march','may','june',
                  'august','september','november','december']
        if year <= 2015:
            months = ['january','february','march','april','may','june',
                      'july','august','september','october','november','december']
        for month in months:
            boe_meetings.append((year, month))

    stmts = {}

    for year, month in boe_meetings:
        date_key = f'{year}-{month}'

        # Try multiple URL formats (BOE changed structure over the years)
        urls_to_try = [
            # Current format (2017+)
            f'{base_url}/monetary-policy-summary-and-minutes/{year}/{month}-{year}',
            # Old format (pre-2017): mpc-month-year
            f'{base_url}/monetary-policy-summary-and-minutes/{year}/mpc-{month}-{year}',
            # Another old variant
            f'{base_url}/monetary-policy-summary-and-minutes/{year}/monetary-policy-summary-and-minutes-{month}-{year}',
        ]

        got_it = False
        for url in urls_to_try:
            if got_it:
                break
            try:
                resp = session.get(url, allow_redirects=True, timeout=15)
                if resp.status_code == 403:
                    continue
                if resp.status_code != 200:
                    continue

                # Make sure we didn't get redirected to a generic page
                if 'access denied' in resp.text.lower()[:500]:
                    continue

                soup = BeautifulSoup(resp.text, 'html.parser')
                summary_text = extract_summary(soup)

                if summary_text and len(summary_text) > 200:
                    stmts[date_key] = summary_text
                    print(f'  OK: {date_key}  ({len(summary_text)} chars)')
                    got_it = True

            except Exception as e:
                continue

            time.sleep(1)

        if not got_it:
            print(f'  MISS: {date_key}')

        time.sleep(1)

    print(f'\n  BOE summaries scraped: {len(stmts)}')
    return stmts


if __name__ == '__main__':
    boe_statements = scrape_boe()

    with open('boe_statements.json', 'w') as f:
        json.dump(boe_statements, f, indent=2)
    print(f'Saved boe_statements.json ({len(boe_statements)} statements)')
# %%
import pandas as pd
import numpy as np
import requests
import io
import time

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

# =============================================================================
# POLICY RATES (hardcoded)
# =============================================================================

rba_changes = {
    '2015-02-03': 2.25, '2015-05-05': 2.00, '2016-05-03': 1.75,
    '2016-08-02': 1.50, '2019-06-04': 1.25, '2019-07-02': 1.00,
    '2019-10-01': 0.75, '2020-03-03': 0.50, '2020-03-19': 0.25,
    '2020-11-03': 0.10, '2022-05-03': 0.35, '2022-06-07': 0.85,
    '2022-07-05': 1.35, '2022-08-02': 1.85, '2022-09-06': 2.35,
    '2022-10-04': 2.60, '2022-11-01': 2.85, '2022-12-06': 3.10,
    '2023-02-07': 3.35, '2023-03-07': 3.60, '2023-05-02': 3.85,
    '2023-06-06': 4.10, '2023-11-07': 4.35, '2025-02-18': 4.10,
}

boe_changes = {
    '2016-08-04': 0.25, '2017-11-02': 0.50, '2018-08-02': 0.75,
    '2020-03-11': 0.25, '2020-03-19': 0.10, '2021-12-16': 0.25,
    '2022-02-03': 0.50, '2022-03-17': 0.75, '2022-05-05': 1.00,
    '2022-06-16': 1.25, '2022-08-04': 1.75, '2022-09-22': 2.25,
    '2022-11-03': 3.00, '2022-12-15': 3.50, '2023-02-02': 4.00,
    '2023-03-23': 4.25, '2023-05-11': 4.50, '2023-06-22': 5.00,
    '2023-08-03': 5.25, '2024-08-01': 5.00, '2024-11-07': 4.75,
    '2025-02-06': 4.50, '2025-05-08': 4.25, '2025-06-19': 4.00,
    '2025-08-07': 4.00, '2025-12-18': 3.75,
}

def build_policy_rate_monthly(changes_dict, start='2015-01-01'):
    idx = pd.date_range(start, pd.Timestamp.today(), freq='ME')
    rate = pd.Series(np.nan, index=idx)
    sorted_changes = sorted(changes_dict.items())
    initial = sorted_changes[0][1]
    for d, r in sorted_changes:
        if d <= start:
            initial = r
    rate.iloc[0] = initial
    for date_str, new_rate in sorted_changes:
        dt = pd.Timestamp(date_str)
        month_end = dt + pd.offsets.MonthEnd(0)
        if month_end in rate.index:
            rate.loc[month_end] = new_rate
    return rate.ffill().bfill()

print('Building policy rates...')
au_cash_rate = build_policy_rate_monthly(rba_changes)
uk_bank_rate = build_policy_rate_monthly(boe_changes)
print(f'  AU cash rate: {len(au_cash_rate)} months, current = {au_cash_rate.iloc[-1]}%')
print(f'  UK bank rate: {len(uk_bank_rate)} months, current = {uk_bank_rate.iloc[-1]}%')


# =============================================================================
# YIELD SCRAPERS - TRY MULTIPLE SOURCES
# =============================================================================

def try_investing_com(country, tenor='2-year'):
    """Scrape from investing.com historical data page."""
    slug_map = {
        ('australia', '2-year'): 'australia-2-year-bond-yield',
        ('australia', '1-year'): 'australia-1-year-bond-yield',
        ('australia', '10-year'): 'australia-10-year-bond-yield',
        ('uk', '2-year'): 'uk-2-year-bond-yield',
        ('uk', '1-year'): 'uk-1-year-bond-yield',
        ('uk', '10-year'): 'uk-10-year-bond-yield',
    }
    slug = slug_map.get((country, tenor))
    if not slug:
        return None
    url = f'https://www.investing.com/rates-bonds/{slug}-historical-data'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200 and 'curr_table' in resp.text:
            tables = pd.read_html(io.StringIO(resp.text))
            for t in tables:
                if 'Price' in t.columns or 'Yield' in t.columns:
                    print(f'    investing.com: got {country} {tenor} ({len(t)} rows)')
                    return t
    except Exception as e:
        pass
    return None


def try_wsj(country, tenor):
    """Try Wall Street Journal bond pages."""
    wsj_map = {
        ('australia', '2-year'): 'TMBMKAU-02Y',
        ('australia', '1-year'): 'TMBMKAU-01Y',
        ('australia', '10-year'): 'TMBMKAU-10Y',
        ('uk', '2-year'): 'TMBMKGB-02Y',
        ('uk', '1-year'): 'TMBMKGB-01Y',
        ('uk', '10-year'): 'TMBMKGB-10Y',
    }
    code = wsj_map.get((country, tenor))
    if not code:
        return None
    url = f'https://www.wsj.com/market-data/quotes/bond/{code}/historical-prices/download?num_rows=3000&range_days=3000&startDate=01/01/2015&endDate=12/31/2026'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200 and ',' in resp.text[:100]:
            df = pd.read_csv(io.StringIO(resp.text))
            print(f'    WSJ: got {country} {tenor} ({len(df)} rows)')
            return df
    except:
        pass
    return None


def try_rba_direct():
    """Pull AU yields directly from RBA statistical tables (CSV)."""
    # RBA Table F2: Capital Market Yields
    url = 'https://www.rba.gov.au/statistics/tables/csv/f2-data.csv'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text), skiprows=10)
            print(f'    RBA F2: got {len(df)} rows')
            return df
    except:
        pass
    # Alt: F2.1 daily
    url2 = 'https://www.rba.gov.au/statistics/tables/csv/f2.1-data.csv'
    try:
        resp = requests.get(url2, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text), skiprows=10)
            print(f'    RBA F2.1: got {len(df)} rows')
            return df
    except:
        pass
    return None


def try_boe_direct():
    """Pull UK yields from BOE yield curve data."""
    # BOE publishes daily yield curve estimates
    url = 'https://www.bankofengland.co.uk/statistics/yield-curves/traditional-curve-data'
    # This is harder to scrape, try their CSV endpoint
    # Nominal par yields
    csv_url = 'https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/latest-yield-curve-data.csv'
    try:
        resp = requests.get(csv_url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            print(f'    BOE yield curve: got {len(df)} rows')
            return df
    except:
        pass
    return None


def try_pandas_datareader(country, tenor):
    """Try pandas-datareader OECD source."""
    try:
        from pandas_datareader import wb, data as pdr
        # OECD MEI interest rates
        ticker_map = {
            ('australia', '10-year'): 'IRLTLT01AUM156N',
            ('uk', '10-year'): 'IRLTLT01GBM156N',
        }
        ticker = ticker_map.get((country, tenor))
        if ticker:
            df = pdr.DataReader(ticker, 'fred', '2015-01-01')
            if len(df) > 0:
                print(f'    pandas-datareader FRED: got {country} {tenor} ({len(df)} rows)')
                return df.squeeze()
    except:
        pass
    return None


# =============================================================================
# PULL YIELDS WITH FALLBACK CHAIN
# =============================================================================

def get_yield(country, tenors=['2-year', '1-year', '10-year']):
    """Try every source for each tenor until something works."""
    for tenor in tenors:
        print(f'  Trying {country} {tenor}...')

        # Source 1: RBA/BOE direct
        if country == 'australia':
            rba_data = try_rba_direct()
            if rba_data is not None:
                return rba_data, 'rba_direct'

        if country == 'uk':
            boe_data = try_boe_direct()
            if boe_data is not None:
                return boe_data, 'boe_direct'

        # Source 2: Investing.com
        result = try_investing_com(country, tenor)
        if result is not None:
            return result, f'investing_{tenor}'

        # Source 3: WSJ
        result = try_wsj(country, tenor)
        if result is not None:
            return result, f'wsj_{tenor}'

        # Source 4: pandas-datareader
        result = try_pandas_datareader(country, tenor)
        if result is not None:
            return result, f'pdr_{tenor}'

        time.sleep(1)

    return None, 'none'


print('\n--- Pulling AU yields ---')
au_yield_data, au_source = get_yield('australia', ['2-year', '1-year', '10-year'])
print(f'  AU source: {au_source}')

print('\n--- Pulling UK yields ---')
uk_yield_data, uk_source = get_yield('uk', ['2-year', '1-year', '10-year'])
print(f'  UK source: {uk_source}')


# =============================================================================
# PARSE RBA DIRECT DATA (if that's what worked)
# =============================================================================

def parse_rba_f2(df):
    """Parse RBA F2 CSV into clean yield series."""
    if df is None:
        return None

    print(f'    RBA columns: {df.columns.tolist()[:15]}')
    print(f'    RBA shape: {df.shape}')
    print(f'    First row: {df.iloc[0].tolist()[:10]}')

    # Find the date column (first column that parses as dates)
    date_col = None
    for col in df.columns:
        try:
            test = pd.to_datetime(df[col].iloc[:5], errors='coerce')
            if test.notna().sum() >= 3:
                date_col = col
                break
        except:
            continue
    if date_col is None:
        date_col = df.columns[0]

    # Find best yield column: prefer 2Y, then 1Y, then 3Y, then anything with 'year'
    yield_col = None
    priorities = ['2 year', '2-year', '2year',
                  '1 year', '1-year', '1year',
                  '3 year', '3-year', '3year',
                  'year', 'yield', 'bond']
    for keyword in priorities:
        for col in df.columns:
            if keyword in col.lower() and col != date_col:
                # Make sure it has numeric data
                test = pd.to_numeric(df[col], errors='coerce')
                if test.notna().sum() > 10:
                    yield_col = col
                    break
        if yield_col:
            break

    # Last resort: use the second column
    if yield_col is None:
        for col in df.columns:
            if col != date_col:
                test = pd.to_numeric(df[col], errors='coerce')
                if test.notna().sum() > 10:
                    yield_col = col
                    break

    if yield_col is None:
        print('    ERROR: could not find any yield column')
        return None

    print(f'    Using date col: {date_col}')
    print(f'    Using yield col: {yield_col}')

    dates = pd.to_datetime(df[date_col], errors='coerce')
    yields = pd.to_numeric(df[yield_col], errors='coerce')

    result = pd.Series(yields.values, index=dates, name='yield')
    result = result.dropna().sort_index()
    result = result[result.index.notna()]

    print(f'    Parsed: {len(result)} obs from {result.index[0]} to {result.index[-1]}')
    return result


# =============================================================================
# BUILD FINAL DATAFRAMES
# =============================================================================
print('\nBuilding macro dataframes...')

# Parse whatever we got
au_yield = None
uk_yield = None

if au_source == 'rba_direct' and au_yield_data is not None:
    au_yield = parse_rba_f2(au_yield_data)
elif isinstance(au_yield_data, pd.Series):
    au_yield = au_yield_data
elif isinstance(au_yield_data, pd.DataFrame) and au_yield_data is not None:
    # From investing.com or WSJ - find the yield/price column
    for col in ['Yield', 'Price', 'Close']:
        if col in au_yield_data.columns:
            au_yield_data['date'] = pd.to_datetime(au_yield_data.get('Date', au_yield_data.index), errors='coerce')
            au_yield = au_yield_data.set_index('date')[col].sort_index()
            au_yield = pd.to_numeric(au_yield, errors='coerce')
            break

if uk_source == 'boe_direct' and uk_yield_data is not None:
    # BOE data needs its own parser
    uk_yield = None  # will need custom parsing
elif isinstance(uk_yield_data, pd.Series):
    uk_yield = uk_yield_data
elif isinstance(uk_yield_data, pd.DataFrame) and uk_yield_data is not None:
    for col in ['Yield', 'Price', 'Close']:
        if col in uk_yield_data.columns:
            uk_yield_data['date'] = pd.to_datetime(uk_yield_data.get('Date', uk_yield_data.index), errors='coerce')
            uk_yield = uk_yield_data.set_index('date')[col].sort_index()
            uk_yield = pd.to_numeric(uk_yield, errors='coerce')
            break

# Build AU macro
au_dict = {'cash_rate': au_cash_rate}
if au_yield is not None:
    au_dict['short_yield'] = au_yield
au_macro = pd.DataFrame(au_dict)
au_macro.index = pd.DatetimeIndex(au_macro.index).tz_localize(None)
au_macro = au_macro.resample('ME').last().ffill()
if 'short_yield' in au_macro.columns:
    au_macro['short_spread'] = au_macro['short_yield'] - au_macro['cash_rate']
print(f'\n  AU macro: {au_macro.shape}')
print(au_macro.tail())

# Build UK macro
uk_dict = {'bank_rate': uk_bank_rate}
if uk_yield is not None:
    uk_dict['short_yield'] = uk_yield
uk_macro = pd.DataFrame(uk_dict)
uk_macro.index = pd.DatetimeIndex(uk_macro.index).tz_localize(None)
uk_macro = uk_macro.resample('ME').last().ffill()
if 'short_yield' in uk_macro.columns:
    uk_macro['short_spread'] = uk_macro['short_yield'] - uk_macro['bank_rate']
print(f'\n  UK macro: {uk_macro.shape}')
print(uk_macro.tail())


# =============================================================================
# IF EVERYTHING FAILS: manual fallback
# =============================================================================
if au_yield is None:
    print('\n  !! AU yields failed from all sources.')
    print('  Manual option: download from https://www.rba.gov.au/statistics/tables/')
    print('  Table F2 "Capital Market Yields" -> download CSV -> load with pd.read_csv()')

if uk_yield is None:
    print('\n  !! UK yields failed from all sources.')
    print('  Manual option: download from https://www.bankofengland.co.uk/statistics/yield-curves')
    print('  Or: https://www.investing.com/rates-bonds/uk-2-year-bond-yield-historical-data')
    print('  Download CSV -> load with pd.read_csv()')
# %%
import pandas as pd
import numpy as np

# =============================================================================
# BUILD COUNTRY DATAFRAMES
# Uses same target construction as FedNotebook: yield change over ~60 trading days
# =============================================================================

def build_country_df(statements_raw, macro_df, yield_series, country_code):
    """
    statements_raw: dict {date_key: statement_text}
    macro_df: DataFrame with 'cash_rate'/'bank_rate', 'short_yield', 'short_spread'
    yield_series: pd.Series of the yield used for target variable (daily or monthly)
    country_code: 'AU' or 'UK'
    """
    rows = []

    for date_str, text in statements_raw.items():
        # Parse the date key
        date = None

        # Try standard formats first
        for fmt in [None, '%Y%m%d', '%Y-%B', '%d-%b-%Y']:
            try:
                if fmt:
                    date = pd.to_datetime(date_str, format=fmt)
                else:
                    date = pd.to_datetime(date_str)
                break
            except:
                continue

        # Handle RBA keys like 'mr-25-03' -- extract date from the statement text
        if date is None and text:
            import re
            # Look for patterns like "18 February 2025" or "7 November 2023" in the text
            match = re.search(r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', text)
            if match:
                try:
                    date = pd.to_datetime(match.group(1))
                except:
                    pass

            # Also try "February 2025" without day
            if date is None:
                match = re.search(r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', text)
                if match:
                    try:
                        date = pd.to_datetime(match.group(1))
                    except:
                        pass

            # Last resort: use the mr-YY-NN to get approximate year
            if date is None:
                try:
                    parts = date_str.replace('mr-', '').split('-')
                    if len(parts) == 2:
                        yr = int(parts[0])
                        if yr < 50:
                            yr += 2000
                        else:
                            yr += 1900
                        release_num = int(parts[1])
                        # Rough approximation: release number maps ~monthly
                        month = min(release_num, 12)
                        date = pd.Timestamp(year=yr, month=month, day=15)
                except:
                    pass

        if date is None:
            continue

        # Make date tz-naive
        if hasattr(date, 'tz') and date.tz is not None:
            date = date.tz_localize(None)

        # Match to nearest macro data
        try:
            macro_dates = macro_df.index[macro_df.index <= date]
            if len(macro_dates) == 0:
                continue
            macro_date = macro_dates[-1]
            row = macro_df.loc[macro_date]

            rows.append({
                'date': date,
                'statement': text,
                'short_yield': row.get('short_yield', np.nan),
                'short_spread': row.get('short_spread', np.nan),
            })
        except Exception as e:
            continue

    if not rows:
        print(f'  {country_code}: No rows built!')
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('date').set_index('date')
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)

    # -- TARGET: yield change ~60 trading days after statement --
    # Same logic as FedNotebook: iloc[loc + 44]
    yield_clean = yield_series.dropna().copy()
    yield_clean.index = pd.DatetimeIndex(yield_clean.index).tz_localize(None)

    df['1y_60d_change'] = np.nan
    for date in df.index:
        try:
            loc = yield_clean.index.get_indexer([date], method='ffill')[0]
            if loc >= 0 and loc + 44 < len(yield_clean):
                df.loc[date, '1y_60d_change'] = float(yield_clean.iloc[loc + 44]) - float(yield_clean.iloc[loc])
        except:
            pass

    df['1y_60d_change'] = df['1y_60d_change'].astype(float)

    print(f'  {country_code} df: {df.shape}')
    print(f'    Statements: {df["statement"].notna().sum()}')
    print(f'    Target obs:  {df["1y_60d_change"].notna().sum()}')
    print(f'    Date range:  {df.index[0].strftime("%Y-%m-%d")} to {df.index[-1].strftime("%Y-%m-%d")}')

    return df


# =============================================================================
# BUILD
# =============================================================================
# au_yield_data came from RBA F2 parse - this is your yield series for the target
# For AU: use the parsed au_yield (2Y ACGB from RBA)
# For UK: use uk_yield_data (10Y gilt from FRED, or whatever pulled)

# If au_yield came from parse_rba_f2, it's already a clean pd.Series
# If uk_yield_data came from pandas-datareader, squeeze it

# Make sure yield series are clean
au_yield_for_target = au_yield.copy() if au_yield is not None else pd.Series(dtype=float)
uk_yield_for_target = uk_yield.copy() if uk_yield is not None else pd.Series(dtype=float)

print('Building AU dataframe...')
df_au = build_country_df(rba_statements, au_macro, au_yield_for_target, 'AU')

print('\nBuilding UK dataframe...')
df_uk = build_country_df(boe_statements, uk_macro, uk_yield_for_target, 'UK')

# Quick sanity check
print('\n' + '='*60)
print('  SANITY CHECK')
print('='*60)
for name, df in [('AU', df_au), ('UK', df_uk)]:
    if len(df) > 0:
        valid = df['1y_60d_change'].notna()
        print(f'\n  {name}:')
        print(f'    Total meetings:  {len(df)}')
        print(f'    With target:     {valid.sum()}')
        print(f'    Mean change:     {df.loc[valid, "1y_60d_change"].mean():.4f}')
        print(f'    Std change:      {df.loc[valid, "1y_60d_change"].std():.4f}')
        if 'short_spread' in df.columns:
            print(f'    Spread obs:      {df["short_spread"].notna().sum()}')
# %% [markdown]
# ### AU & UK Training Sentences
# %%
sentence_training_AU = [
    # =============================================================================
    # RATE DECISIONS - EXPLICIT ACTIONS
    # =============================================================================
    # RBA uses "cash rate target" not "federal funds rate"
    # RBA target band is 2-3% (not 2% like Fed)
    
    # Hikes
    ("The Board decided to increase the cash rate target by 50 basis points", 0.92),
    ("The Board decided to increase the cash rate target by 25 basis points", 0.80),
    ("At its meeting today the Board decided to increase the cash rate target by 25 basis points to 4.35 per cent", 0.80),
    ("The Board judged an increase in interest rates was warranted to be more assured that inflation would return to target in a reasonable timeframe", 0.82),
    ("This further increase in interest rates is to provide greater confidence that inflation will return to target within a reasonable timeframe", 0.82),
    
    # Holds
    ("The Board decided to leave the cash rate target unchanged", 0.52),
    ("The Board decided to leave the cash rate target unchanged at 4.35 per cent", 0.55),
    ("The Board took the decision to hold interest rates steady this month to provide additional time to assess the impact of the increase in interest rates to date", 0.55),
    ("The Board decided that it was appropriate to hold rates steady to provide time to assess the impact of the increase in interest rates so far", 0.55),
    
    # Cuts
    ("The Board decided to lower the cash rate target by 25 basis points", 0.25),
    ("At its meeting today the Board decided to lower the cash rate target to 4.10 per cent", 0.25),
    ("In removing a little of the policy restrictiveness in its decision today the Board acknowledges that progress has been made", 0.28),
    ("The Board decided to reduce the cash rate target by 50 basis points", 0.15),
    ("The Board decided to lower the cash rate target to 0.25 per cent", 0.05),
    ("The Board decided to reduce the cash rate to 0.10 per cent", 0.03),
    
    # =============================================================================
    # FORWARD GUIDANCE - HAWKISH
    # =============================================================================
    
    # Strong hawkish
    ("The Board expects that further increases in interest rates will be needed over the months ahead to ensure that inflation returns to target", 0.90),
    ("The Board expects that further tightening of monetary policy will be needed to ensure that inflation returns to target and that this period of high inflation is only temporary", 0.88),
    ("Some further tightening of monetary policy may well be needed to ensure that inflation returns to target", 0.82),
    ("Some further tightening of monetary policy may be required to ensure that inflation returns to target in a reasonable timeframe", 0.80),
    ("Whether further tightening of monetary policy is required will depend upon the data and the evolving assessment of risks", 0.70),
    ("The Board is not ruling anything in or out", 0.58),
    ("The Board remains vigilant to the risk of continued high inflation", 0.75),
    ("The Board is not ruling anything out", 0.60),
    
    # Moderate hawkish
    ("Keeping the cash rate at the current level is important to reduce inflationary pressures", 0.62),
    ("The Board remains cautious on prospects for further policy easing", 0.62),
    ("The forecasts suggest that if monetary policy is eased too much too soon disinflation could stall", 0.68),
    ("If monetary policy is eased too much too soon disinflation could stall and inflation would settle above the midpoint of the target range", 0.70),
    ("The Board expects it will be some time before inflation is sustainably low and stable", 0.65),
    ("The Board does not see inflation returning sustainably to target until 2026", 0.65),
    ("The process of returning inflation to target has been slow and bumpy", 0.68),
    ("The Board acknowledges that progress has been made but is cautious about the outlook", 0.58),
    
    # =============================================================================
    # FORWARD GUIDANCE - DOVISH
    # =============================================================================
    
    # Strong dovish
    ("The Board is prepared to use its full range of tools to support the economy", 0.10),
    ("The Board will do what is necessary to support the Australian economy", 0.12),
    ("The Board considers it unlikely that the cash rate target will be increased for at least three years", 0.08),
    ("The Board will not increase the cash rate target until actual inflation is sustainably within the 2 to 3 per cent target range", 0.10),
    ("The Board will maintain highly accommodative policy settings as long as is required", 0.08),
    
    # Moderate dovish
    ("Some of the upside risks to inflation appear to have eased and there are signs that disinflation might be occurring a little more quickly than earlier expected", 0.30),
    ("These factors give the Board more confidence that inflation is moving sustainably towards the midpoint of the 2 to 3 per cent target range", 0.28),
    ("Inflationary pressures are easing a little more quickly than expected", 0.28),
    ("There has also been continued subdued growth in private demand and wage pressures have eased", 0.30),
    ("The Board has gained confidence that inflation is moving sustainably towards the midpoint of the target range", 0.25),
    ("Progress on inflation and the evolving balance of risks give the Board confidence to begin easing policy", 0.22),
    
    # =============================================================================
    # FORWARD GUIDANCE - PACE / CONDITIONALITY
    # =============================================================================
    
    ("The Board will continue to rely upon the data and the evolving assessment of risks to guide its decisions", 0.50),
    ("The Board will continue to pay close attention to developments in the global economy trends in domestic demand and the outlook for inflation and the labour market", 0.50),
    ("The path of interest rates that will best ensure that inflation returns to target in a reasonable timeframe will depend upon the data and the evolving assessment of risks", 0.50),
    ("The Board will be paying close attention to developments in the global economy trends in household spending and the outlook for inflation and the labour market", 0.50),
    ("How interest rates evolve from here is uncertain and the Board is watching developments closely", 0.52),
    
    # =============================================================================
    # INFLATION ASSESSMENT - HAWKISH
    # =============================================================================
    
    # Very hawkish
    ("Inflation in Australia is too high", 0.88),
    ("Inflation at 7 per cent is still too high and it will be some time yet before it is back in the target range", 0.90),
    ("Inflation is still too high and is proving more persistent than expected", 0.88),
    ("CPI inflation was 7.8 per cent the highest since 1990", 0.92),
    ("In underlying terms inflation was 6.9 per cent which was higher than expected", 0.90),
    ("Inflation expectations risk becoming unanchored", 0.92),
    ("If high inflation were to become entrenched in peoples expectations it would be much more costly to reduce later", 0.85),
    ("There is a risk that expectations of ongoing high inflation contribute to larger increases in both prices and wages", 0.85),
    
    # Moderately hawkish
    ("Inflation is still some way above the midpoint of the 2 to 3 per cent target range", 0.72),
    ("Inflation is proving persistent", 0.75),
    ("Underlying inflation has been above the midpoint of the target for 11 consecutive quarters", 0.78),
    ("Quarterly underlying CPI inflation has fallen very little over the past year", 0.75),
    ("Services price inflation remains high with strong demand for some services", 0.72),
    ("Rents are increasing at the fastest rate in some years with vacancy rates low in many parts of the country", 0.70),
    ("The prices of many services are continuing to rise briskly", 0.72),
    ("Inflation has been higher than expected and labour market conditions have proven stronger than anticipated", 0.70),
    ("The recent data have demonstrated that the process of returning inflation to target has been slow and bumpy", 0.68),
    ("Services price inflation remains elevated in many economies", 0.65),
    ("Goods price inflation has eased but services price inflation remains high", 0.62),
    
    # =============================================================================
    # INFLATION ASSESSMENT - DOVISH
    # =============================================================================
    
    # Strong dovish
    ("Inflation has fallen substantially since the peak in 2022", 0.25),
    ("Inflation has come down considerably from its peak", 0.22),
    ("The disinflationary process is continuing", 0.22),
    ("Underlying inflation was 3.2 per cent which suggests inflationary pressures are easing a little more quickly than expected", 0.28),
    
    # Moderate dovish
    ("Inflation has passed its peak but is still too high", 0.45),
    ("Inflation continues to moderate", 0.32),
    ("A range of information suggests that inflation has peaked in Australia", 0.35),
    ("The monthly CPI indicator suggests that inflation has peaked in Australia", 0.35),
    ("Goods price inflation has declined but services price inflation remains high", 0.42),
    ("Goods price inflation is expected to moderate over the months ahead", 0.35),
    ("Medium-term inflation expectations remain well anchored and it is important that this remains the case", 0.40),
    ("Longer term inflation expectations have been consistent with the inflation target", 0.38),
    ("Overall measures of inflation expectations remain consistent with the inflation target", 0.40),
    ("Inflation has eased substantially while the labor market has remained strong", 0.32),
    ("The decline in inflation reflects the easing of supply-demand imbalances", 0.30),
    
    # =============================================================================
    # LABOUR MARKET - HAWKISH
    # =============================================================================
    # RBA uses "labour" not "labor"
    
    ("The labour market remains very tight", 0.80),
    ("The labour market remains extremely tight with the unemployment rate at historically low levels", 0.82),
    ("Conditions in the labour market remain tight", 0.72),
    ("The unemployment rate is at a near 50-year low", 0.78),
    ("Labour market conditions remain tight and in fact tightened a little further", 0.75),
    ("Many firms continue to experience difficulty hiring workers", 0.72),
    ("Job vacancies and job ads are both at very high levels", 0.75),
    ("Demand has remained above the economys capacity to supply goods and services", 0.78),
    ("The availability of labour is still a constraint for a range of employers", 0.70),
    ("Measures of labour underutilisation have declined", 0.72),
    ("Labour force participation has remained below pre-pandemic levels", 0.70),
    ("Productivity growth has not picked up which implies that growth in unit labour costs remains high", 0.75),
    ("Unit labour costs are rising briskly with productivity growth remaining subdued", 0.78),
    ("The Board remains alert to the risk of a prices-wages spiral given the limited spare capacity in the economy", 0.82),
    
    # =============================================================================
    # LABOUR MARKET - DOVISH
    # =============================================================================
    
    ("Labour market conditions are easing but are still tight relative to full employment", 0.42),
    ("Conditions in the labour market have eased but remain tight", 0.40),
    ("Labour market conditions have eased", 0.32),
    ("The unemployment rate has increased", 0.25),
    ("The unemployment rate is expected to increase", 0.30),
    ("Firms report that labour shortages have eased", 0.35),
    ("The number of vacancies has declined", 0.35),
    ("Wage pressures have eased a little more than expected", 0.30),
    ("Wage pressures have eased", 0.32),
    ("Wages growth is not expected to increase much further", 0.38),
    ("Wages growth is still consistent with the inflation target provided that productivity growth picks up", 0.42),
    ("The gap between jobs and workers has narrowed", 0.32),
    ("There is a risk of a sharper deterioration in the labour market than currently projected", 0.22),
    
    # =============================================================================
    # ECONOMIC ACTIVITY - HAWKISH
    # =============================================================================
    
    ("The economy has been stronger than expected", 0.72),
    ("The Australian economy grew strongly", 0.70),
    ("GDP growth has been above trend", 0.68),
    ("Economic activity has been surprisingly resilient", 0.68),
    ("Household spending remains strong supported by a solid labour market", 0.68),
    ("Consumer spending has been surprisingly resilient", 0.68),
    ("The economy is operating at a high level of capacity utilisation", 0.65),
    ("Demand continues to exceed supply in the economy", 0.75),
    
    # =============================================================================
    # ECONOMIC ACTIVITY - DOVISH
    # =============================================================================
    
    ("Growth in output has been weak", 0.25),
    ("Private domestic demand is recovering a little more slowly than earlier expected", 0.28),
    ("Economic growth remains slow", 0.25),
    ("Higher interest rates have led people to cut back on spending", 0.30),
    ("The economy has been experiencing a period of below-trend growth", 0.28),
    ("Household consumption growth is weak as is dwelling investment", 0.22),
    ("High inflation is weighing on peoples real incomes and household consumption growth is weak", 0.25),
    ("The combination of higher interest rates and cost-of-living pressures is leading to a substantial slowing in household spending", 0.25),
    ("GDP growth is expected to slow", 0.30),
    ("Growth is expected to remain low over the next year", 0.28),
    ("The pace of economic growth has slowed significantly", 0.20),
    ("Consumer spending has weakened reflecting the squeeze on household budgets", 0.22),
    ("The economy contracted modestly", 0.15),
    
    # =============================================================================
    # MONETARY POLICY STANCE
    # =============================================================================
    
    # Restrictive stance (neutral to mildly dovish - acknowledging tightening is biting)
    ("Monetary policy has been restrictive and will remain so", 0.48),
    ("The Boards assessment is that monetary policy has been restrictive and will remain so after this reduction in the cash rate", 0.35),
    ("Higher interest rates are working to establish a more sustainable balance between supply and demand in the economy", 0.48),
    ("Higher interest rates are working to bring aggregate demand and supply closer towards balance", 0.45),
    ("The current stance of monetary policy is restrictive", 0.50),
    ("The full effect of the cumulative increase in interest rates is yet to be felt", 0.55),
    ("The Board recognises that monetary policy operates with a lag and that the full effect of the cumulative increase in interest rates is yet to be felt", 0.55),
    
    # =============================================================================
    # HOUSING / PROPERTY
    # =============================================================================
    
    ("Housing prices are rising again", 0.62),
    ("Housing prices are continuing to rise across the country", 0.62),
    ("Housing cost inflation is abating", 0.35),
    ("Dwelling investment remains weak", 0.30),
    ("The share of household incomes used to meet mortgage payments is high by recent historical standards and still rising", 0.42),
    ("Rents are increasing at the fastest rate in some years", 0.70),
    
    # =============================================================================
    # FINANCIAL CONDITIONS
    # =============================================================================
    
    ("A large share of the increase in the cash rate has been passed on to borrowers", 0.50),
    ("Housing credit growth has stabilised at a lower level", 0.38),
    ("Tighter monetary policy has contributed to a noticeable slowing in the growth of demand", 0.42),
    ("Financial conditions are restrictive", 0.48),
    ("Household balance sheets are being affected by the decline in housing prices", 0.30),
    
    # =============================================================================
    # GLOBAL / EXTERNAL
    # =============================================================================
    
    ("The outlook for the Chinese economy has softened", 0.32),
    ("Uncertainty about the outlook abroad remains significant", 0.38),
    ("Geopolitical and policy uncertainties are pronounced", 0.38),
    ("Global inflation remains very high", 0.65),
    ("Most central banks have been easing monetary policy as they become more confident that inflation is moving sustainably back towards their targets", 0.35),
    ("The outlook for the global economy remains subdued with below average growth expected", 0.30),
    ("Global growth is subdued and inflation is above target in many countries", 0.55),
    ("There also remains a high level of uncertainty about the overseas outlook", 0.40),
    
    # =============================================================================
    # RISKS / UNCERTAINTY
    # =============================================================================
    
    # Hawkish risks
    ("Upside risks to inflation remain", 0.78),
    ("Services price inflation has been surprisingly persistent overseas and the same could occur in Australia", 0.72),
    ("There is a risk that high inflation could become entrenched in peoples expectations", 0.85),
    ("The Board remains alert to the risk of a prices-wages spiral", 0.80),
    
    # Dovish risks
    ("There is a risk that any pick-up in consumption is slower than expected resulting in continued subdued output growth", 0.25),
    ("There is a risk of continued subdued output growth and a sharper deterioration in the labour market", 0.22),
    ("Downside risks to the global outlook have increased", 0.25),
    
    # Balanced
    ("There are risks on both sides", 0.45),
    ("There are notable uncertainties about the outlook for domestic economic activity and inflation", 0.45),
    ("The economic outlook is uncertain", 0.45),
    ("The path to achieving a soft landing remains a narrow one", 0.52),
    ("There are uncertainties regarding the lags in the effect of monetary policy", 0.48),
    
    # =============================================================================
    # CORE MANDATE / PRIORITY LANGUAGE
    # =============================================================================
    
    ("Sustainably returning inflation to target within a reasonable timeframe remains the Boards highest priority", 0.60),
    ("Returning inflation to target within a reasonable timeframe remains the Boards priority", 0.60),
    ("The Boards priority is to return inflation to target", 0.62),
    ("Bringing inflation down to the 2 to 3 per cent target range is the Boards highest priority", 0.62),
    ("The Board remains resolute in its determination to return inflation to target and will do what is necessary to achieve that outcome", 0.65),
    ("High inflation makes life difficult for everyone and damages the functioning of the economy", 0.65),
    ("It erodes the value of savings hurts household budgets and makes it harder for businesses to plan and invest", 0.65),
    
    # =============================================================================
    # WAGES / PRODUCTIVITY (RBA-specific focus)
    # =============================================================================
    
    ("Wages growth is continuing to pick up in response to the tight labour market and higher inflation", 0.72),
    ("Wages growth has picked up but at the aggregate level remains consistent with the inflation target provided productivity growth picks up", 0.55),
    ("At the aggregate level wages growth is still consistent with the inflation target provided that productivity growth picks up", 0.50),
    ("Growth in public sector wages is expected to pick up further", 0.62),
    ("The annual increase in award wages was higher than it was last year", 0.65),
    ("Wages growth remains consistent with the inflation target", 0.45),
    ("Wage growth has shown signs of moderating", 0.35),
    ("Wage pressures have eased more than expected", 0.28),
    
    # =============================================================================
    # PANDEMIC ERA
    # =============================================================================
    
    ("The pandemic continues to cause significant disruption to the Australian economy", 0.15),
    ("The recovery from the pandemic is well under way", 0.40),
    ("The pandemic-related supply and demand imbalances have contributed to considerable inflation", 0.62),
    ("The Board is committed to doing what is necessary to ensure the recovery is sustained", 0.15),
    
    # =============================================================================
    # BOILERPLATE / NEUTRAL
    # =============================================================================
    
    ("This is consistent with the RBAs mandate for price stability and full employment", 0.50),
    ("The Board will continue to monitor the implications of incoming data for the economic outlook", 0.50),
    ("The Board is seeking to return inflation to the 2 to 3 per cent target range while keeping the economy on an even keel", 0.50),
    ("The Board will make its decisions based on the data and the evolving assessment of risks", 0.50),
    ("Information received since the last meeting is broadly in line with expectations", 0.50),
    
    # =============================================================================
    # TRANSITION / AMBIGUOUS (filling 0.4-0.6 range)
    # =============================================================================
    
    ("Inflation has passed its peak but remains above target", 0.48),
    ("The labour market remains tight though conditions have eased somewhat", 0.48),
    ("Economic growth has slowed but remains positive", 0.42),
    ("Higher interest rates are working as intended but inflation remains above target", 0.52),
    ("The economy is adjusting to higher interest rates with demand slowing and inflation moderating", 0.45),
    ("Progress on inflation has been welcome but the job is not yet done", 0.55),
    ("Labour market conditions are easing gradually but remain tight", 0.45),
    ("The economy is growing below trend as intended to bring inflation down", 0.48),
    ("Growth is expected to pick up gradually as real incomes recover", 0.42),
    ("While today's decision recognises the welcome progress on inflation the Board remains cautious", 0.45),
]
# %%
sentence_training_boe = [
    # =============================================================================
    # RATE DECISIONS - EXPLICIT ACTIONS
    # =============================================================================
    # BOE uses "Bank Rate" not "federal funds rate" or "cash rate"
    # BOE target is 2% CPI
    # MPC = Monetary Policy Committee (9 members)
    
    # Hikes
    ("The MPC voted to increase Bank Rate by 0.75 percentage points", 0.95),
    ("The MPC voted by a majority to increase Bank Rate by 0.5 percentage points", 0.88),
    ("The MPC voted by a majority to increase Bank Rate by 0.25 percentage points", 0.78),
    ("The MPC voted unanimously to increase Bank Rate by 0.25 percentage points", 0.80),
    
    # Holds
    ("The MPC voted by a majority to maintain Bank Rate at 5.25 per cent", 0.55),
    ("The MPC voted to maintain Bank Rate", 0.52),
    ("The MPC voted by a majority of 6-3 to maintain Bank Rate at 4.75 per cent", 0.55),
    
    # Cuts
    ("The MPC voted by a majority to reduce Bank Rate by 0.25 percentage points", 0.25),
    ("The MPC voted to reduce Bank Rate by 0.5 percentage points", 0.15),
    ("The MPC voted to reduce Bank Rate to 0.1 per cent", 0.05),
    ("At its meeting the MPC voted by a majority of 5-4 to reduce Bank Rate by 0.25 percentage points", 0.28),
    
    # =============================================================================
    # VOTE SPLITS (BOE-specific - very important signal)
    # =============================================================================
    # BOE publishes exact vote counts which are a key signal
    
    # Hawkish splits
    ("Four members preferred to increase Bank Rate by 0.25 percentage points", 0.58),
    ("Two members preferred to increase Bank Rate by 0.5 percentage points", 0.60),
    ("Three members voted to maintain Bank Rate", 0.42),
    ("The MPC voted by a majority of 5-4 to maintain Bank Rate with four members preferring to increase", 0.58),
    
    # Dovish splits
    ("Two members preferred to maintain Bank Rate", 0.45),
    ("Three members preferred to reduce Bank Rate by 0.25 percentage points", 0.42),
    ("Two members preferred to reduce Bank Rate by 0.25 percentage points", 0.45),
    ("Four members voted to maintain Bank Rate at 4 per cent", 0.55),
    ("Two members voted to reduce Bank Rate by 0.25 percentage points", 0.45),
    
    # =============================================================================
    # FORWARD GUIDANCE - HAWKISH
    # =============================================================================
    
    # Strong hawkish
    ("If there were to be evidence of more persistent pressures then further tightening in monetary policy would be required", 0.85),
    ("The MPC will adjust Bank Rate as necessary to return inflation to the 2 per cent target sustainably in the medium term", 0.60),
    ("Monetary policy will need to be sufficiently restrictive for sufficiently long to return inflation to the 2 per cent target sustainably", 0.78),
    ("The MPC will ensure that Bank Rate is sufficiently restrictive for sufficiently long to return inflation to the 2 per cent target", 0.75),
    ("Interest rates need to remain high enough to be confident that inflation will fall all the way to target and stay there", 0.72),
    
    # Moderate hawkish
    ("The MPC will continue to monitor closely indications of persistent inflationary pressures", 0.68),
    ("The Committee remains focused on squeezing out any existing or emerging persistent inflationary pressures", 0.72),
    ("Domestic inflationary pressures are resolving more slowly", 0.68),
    ("Remaining domestic inflationary pressures are resolving more slowly than expected", 0.72),
    ("The risks from greater inflation persistence remain", 0.70),
    ("Judgements around further policy easing will become a closer call", 0.58),
    ("The MPC will need to see further evidence that inflationary pressures are easing", 0.65),
    ("The Committee judges that monetary policy needs to remain restrictive", 0.68),
    
    # =============================================================================
    # FORWARD GUIDANCE - DOVISH
    # =============================================================================
    
    # Strong dovish
    ("The Bank of England stands ready to take whatever additional action is necessary", 0.10),
    ("The MPC will take whatever action is needed to achieve its remit", 0.12),
    ("The Committee continues to judge that some degree of monetary policy easing is warranted", 0.22),
    ("The Committee voted to maintain the stock of sterling non-financial investment-grade corporate bond purchases and government bond purchases", 0.15),
    
    # Moderate dovish
    ("There has been substantial disinflation over the past two and a half years supported by the restrictive stance of monetary policy", 0.30),
    ("That progress has allowed for reductions in Bank Rate over the past year", 0.28),
    ("Bank Rate is likely to continue on a gradual downward path", 0.25),
    ("On the basis of the current evidence Bank Rate is likely to continue on a gradual downward path", 0.28),
    ("The restrictiveness of policy has fallen as Bank Rate has been reduced", 0.30),
    ("The risk from greater inflation persistence has become somewhat less pronounced", 0.32),
    ("The MPC's approach is to reduce the degree of monetary policy restrictiveness gradually", 0.30),
    ("The Committee judges that some reduction in the degree of policy restrictiveness is now appropriate", 0.25),
    ("Monetary policy had helped to reduce inflationary pressures over the past three years and that had allowed the MPC to make policy less restrictive", 0.28),
    
    # =============================================================================
    # FORWARD GUIDANCE - PACE / CONDITIONALITY
    # =============================================================================
    
    ("A gradual approach to removing monetary policy restrictiveness remains appropriate", 0.42),
    ("A gradual approach allows the MPC to assess carefully the balance of risks to inflation as the evidence evolves", 0.45),
    ("The extent of further easing in monetary policy will depend on the evolution of the outlook for inflation", 0.45),
    ("The extent of any further rate cuts will depend on how the evidence on inflation persistence and weakening demand plays out", 0.42),
    ("The MPC will continue to decide the appropriate level of Bank Rate at each meeting", 0.50),
    ("As Bank Rate falls how much further to lower it will inevitably become a closer call", 0.48),
    ("Monetary policy has been guided by the need to squeeze remaining inflationary pressures out of the economy", 0.60),
    
    # =============================================================================
    # INFLATION ASSESSMENT - HAWKISH
    # =============================================================================
    
    # Very hawkish
    ("CPI inflation remains well above the 2 per cent target", 0.85),
    ("Inflation is still too high", 0.85),
    ("CPI inflation was 10.5 per cent its highest level in over 40 years", 0.95),
    ("Global consumer price inflation remains high", 0.75),
    ("UK domestic inflationary pressures have been firmer than expected", 0.80),
    ("Both private sector regular pay growth and services CPI inflation have been notably higher than forecast", 0.82),
    
    # Moderately hawkish
    ("Services CPI inflation remains elevated", 0.72),
    ("Services consumer price inflation remains persistent", 0.72),
    ("Services price inflation has remained persistent and the same could occur in the United Kingdom", 0.70),
    ("Core CPI inflation remains elevated", 0.72),
    ("Pay growth remains elevated and above rates consistent with the 2 per cent target", 0.75),
    ("Annual private sector regular average weekly earnings growth remains above target-consistent levels", 0.72),
    ("Second-round effects in domestic prices and wages have taken longer to unwind than they did to emerge", 0.70),
    ("Indicators of inflation persistence remain elevated", 0.70),
    ("Services price inflation is expected to remain broadly unchanged in the near term", 0.62),
    ("Forward-looking wage indicators have remained elevated", 0.68),
    ("Some indicators of wage and price-setting appear to have plateaued", 0.65),
    
    # =============================================================================
    # INFLATION ASSESSMENT - DOVISH
    # =============================================================================
    
    # Strong dovish
    ("CPI inflation has fallen back sharply", 0.22),
    ("There has been substantial disinflation", 0.22),
    ("CPI inflation has fallen to the 2 per cent target", 0.20),
    ("Inflation has returned to around the 2 per cent target", 0.20),
    ("Underlying disinflation has generally continued", 0.28),
    
    # Moderate dovish
    ("Twelve-month CPI inflation has fallen", 0.32),
    ("CPI inflation is expected to fall significantly", 0.28),
    ("CPI inflation is projected to return to the 2 per cent target", 0.30),
    ("There has been progress in disinflation particularly as previous external shocks have abated", 0.32),
    ("Reflecting restrictive monetary policy pay growth and services price inflation have continued to ease", 0.30),
    ("Measures of underlying services price inflation pointed to the disinflation process continuing", 0.30),
    ("Pay growth has eased", 0.32),
    ("Private sector regular AWE pay growth has fallen", 0.32),
    ("A range of pay settlements data suggest that pay growth is moderating", 0.35),
    ("Agents intelligence suggests that pay settlements are expected to be lower next year", 0.32),
    ("Headline CPI inflation is expected to decline significantly", 0.28),
    ("The decline in headline inflation largely reflects the easing of external price pressures", 0.30),
    
    # =============================================================================
    # LABOUR MARKET - HAWKISH
    # =============================================================================
    
    ("The labour market remains tight", 0.72),
    ("The labour market has remained tight relative to historical standards", 0.70),
    ("Wage growth has been stronger than can be explained by economic fundamentals", 0.78),
    ("Pay growth has been above rates consistent with meeting the inflation target", 0.75),
    ("There has been upside news on wages and the labour market", 0.72),
    ("The labour market appeared still to be relatively tight", 0.65),
    ("Employment growth has been stronger than expected", 0.68),
    
    # =============================================================================
    # LABOUR MARKET - DOVISH
    # =============================================================================
    
    ("The Committee now judges that the labour market is broadly in balance", 0.38),
    ("The labour market has continued to loosen", 0.30),
    ("The labour market had loosened further", 0.28),
    ("The unemployment rate has continued to move higher", 0.25),
    ("The LFS unemployment rate has risen", 0.28),
    ("The redundancy rate has risen to its highest level since 2013", 0.22),
    ("Employment growth has remained subdued", 0.30),
    ("Labour market tightness has eased", 0.32),
    ("The level of vacancies has been broadly stable suggesting the labour market is cooling", 0.35),
    ("There is evidence of subdued economic growth and building slack in the labour market", 0.25),
    ("Most labour market data have not suggested a rapid opening up of slack even though the unemployment rate has continued to move higher", 0.35),
    
    # =============================================================================
    # ECONOMIC ACTIVITY - HAWKISH
    # =============================================================================
    
    ("UK GDP growth has been stronger than expected", 0.68),
    ("GDP growth has surprised to the upside", 0.68),
    ("Near-term growth outturns have tended to be stronger than expected", 0.65),
    ("Consumer spending has been more resilient than anticipated", 0.65),
    ("The economy appears to have been somewhat more resilient", 0.62),
    ("GDP is expected to increase in the second quarter", 0.58),
    
    # =============================================================================
    # ECONOMIC ACTIVITY - DOVISH
    # =============================================================================
    
    ("GDP growth has been subdued", 0.28),
    ("Most indicators of UK near-term activity have declined", 0.25),
    ("GDP growth is expected to have been weaker than projected", 0.25),
    ("UK GDP is expected to grow below its potential rate", 0.30),
    ("A margin of economic slack is projected to emerge", 0.25),
    ("Aggregate demand and supply are judged to be broadly in balance", 0.40),
    ("The economy has been experiencing a period of below-trend growth", 0.28),
    ("Business surveys have remained subdued", 0.28),
    ("Monthly GDP declined", 0.22),
    ("The economy is in or close to recession", 0.15),
    ("GDP growth has weakened and the economy has stagnated", 0.20),
    ("Consumer spending has weakened reflecting the squeeze on real incomes", 0.25),
    
    # =============================================================================
    # QUANTITATIVE TIGHTENING (QT) - BOE specific
    # =============================================================================
    
    # Hawkish (active QT)
    ("The Committee voted unanimously to reduce the stock of UK government bond purchases by 100 billion over the next twelve months", 0.68),
    ("The Bank should reduce the stock of UK government bond purchases held for monetary policy purposes", 0.65),
    ("The Committee voted to reduce the stock of purchased gilts", 0.65),
    ("The MPC reaffirmed that there would be a high bar for amending the planned reduction in the stock of purchased gilts", 0.62),
    
    # Dovish (slowing QT / QE)
    ("The Committee voted to reduce the stock by 70 billion over the next 12 months", 0.55),
    ("The MPC stands ready to make further asset purchases if needed", 0.15),
    ("The Committee voted to increase the stock of UK government bond purchases", 0.10),
    ("The Committee voted to maintain the stock of sterling non-financial investment-grade corporate bond purchases", 0.18),
    
    # =============================================================================
    # FINANCIAL CONDITIONS
    # =============================================================================
    
    ("Mortgage rates have risen notably", 0.55),
    ("Gilt yields have risen materially particularly at shorter maturities", 0.58),
    ("Financial conditions have tightened", 0.48),
    ("The significant tightening in monetary policy since the end of 2021 is having an impact", 0.48),
    ("Credit conditions have tightened and are weighing on economic activity", 0.35),
    ("The sterling effective exchange rate has appreciated further", 0.45),
    ("There have been large and volatile moves in global financial markets", 0.40),
    ("Net lending has contributed towards stronger broad money growth over recent quarters", 0.58),
    
    # =============================================================================
    # GLOBAL / EXTERNAL
    # =============================================================================
    
    ("Global activity indicators have been more resilient than expected", 0.58),
    ("There has been a further increase in geopolitical and global trade policy uncertainty", 0.38),
    ("The US administration has imposed tariffs on some goods imports", 0.40),
    ("Trade policy uncertainty may weigh on activity in a number of advanced economies including the United Kingdom", 0.32),
    ("The most acute risks from recent global banking sector stresses appeared to have faded", 0.48),
    ("Growth momentum in China appears to have slowed", 0.35),
    ("Recent weakness in Chinese tradable goods prices could put downward pressure on world export price inflation", 0.30),
    ("The euro-area economy has been more resilient than expected", 0.55),
    
    # =============================================================================
    # RISKS / UNCERTAINTY
    # =============================================================================
    
    # Hawkish risks
    ("There are upside risks to the inflation outlook from domestic price and wage pressures", 0.78),
    ("The risks to the inflation outlook are skewed to the upside", 0.80),
    ("There remains a risk that inflation persistence could be more enduring", 0.75),
    ("Second-round effects in wages and prices pose upside risks to inflation", 0.75),
    ("The risk that tariff-driven price increases could become embedded in broader expectations", 0.72),
    
    # Dovish risks
    ("The risk to medium-term inflation from weaker demand remains", 0.28),
    ("Downside risks to the growth outlook have increased", 0.22),
    ("There is a risk that demand weakness could lead to inflation falling below target", 0.20),
    ("Both of these could lead to inflation falling below target in the medium term", 0.22),
    
    # Balanced
    ("The risks around achieving the inflation target are more balanced", 0.38),
    ("The risks from greater inflation persistence and weaker demand are more balanced", 0.40),
    ("Monetary policy is being set to ensure CPI inflation settles sustainably at 2 per cent in the medium term which involves balancing the risks", 0.45),
    ("Uncertainties around the global financial and economic outlook remain elevated", 0.42),
    ("There is considerable uncertainty around statistics derived from the ONS Labour Force Survey", 0.45),
    
    # =============================================================================
    # ENERGY / EXTERNAL SHOCKS (BOE-specific focus)
    # =============================================================================
    
    ("Wholesale gas futures and oil prices have fallen materially", 0.30),
    ("Energy prices have risen sharply contributing to higher inflation", 0.78),
    ("The direct contribution of energy has remained the largest component of the overshoot in CPI inflation", 0.65),
    ("As the effects of the energy price shock unwound headline CPI inflation has fallen sharply", 0.25),
    ("Wholesale energy prices have fallen significantly", 0.28),
    ("Past increases in energy and other goods prices are falling out of the annual rate calculation", 0.28),
    
    # =============================================================================
    # HOUSING / PROPERTY
    # =============================================================================
    
    ("The clearest signs of weakness continue to be in the housing sector", 0.30),
    ("Housing investment and most measures of house prices have fallen", 0.28),
    ("The housing market has continued to weaken reflecting higher mortgage rates", 0.28),
    ("House prices have stabilised and may be rising again", 0.52),
    ("Housing activity remains weak", 0.30),
    
    # =============================================================================
    # FISCAL POLICY (BOE more explicit about fiscal)
    # =============================================================================
    
    ("Announced increases in the National Living Wage and employer National Insurance contributions", 0.62),
    ("Fiscal measures could increase the level of GDP", 0.55),
    ("Government spending has contributed to aggregate demand", 0.58),
    ("The impact of fiscal policy on inflation will depend on how businesses absorb higher costs", 0.55),
    ("Firms have indicated that they expect to pass on some of the increase in costs from higher employer NICs", 0.65),
    
    # =============================================================================
    # BOILERPLATE / NEUTRAL
    # =============================================================================
    
    ("The MPC sets monetary policy to meet the 2 per cent inflation target and in a way that helps to sustain growth and employment", 0.50),
    ("The MPC adopts a medium-term and forward-looking approach to determine the monetary stance", 0.50),
    ("The Bank of England Act 1998 gives the Bank of England operational responsibility for setting monetary policy", 0.50),
    ("The MPC's remit is clear that the inflation target applies at all times reflecting the primacy of price stability", 0.50),
    ("The Committee discussed the international economy monetary and financial conditions demand and output and supply costs and prices", 0.50),
    
    # =============================================================================
    # PERSISTENCE LANGUAGE (BOE-specific analytical framework)
    # =============================================================================
    
    ("There are signs that the persistence of inflation may be fading", 0.32),
    ("Key indicators of inflation persistence have remained elevated", 0.72),
    ("The pace at which domestic inflationary pressures ease will depend on the evolution of the economy", 0.55),
    ("Second-round effects in domestic prices and wages are taking longer to unwind", 0.70),
    ("The MPC judges that the risks to medium-term inflation from greater inflation persistence and weaker demand are more balanced", 0.40),
    ("Underlying inflation persistence has eased but remains above target-consistent levels", 0.55),
    ("The Committee will continue to assess whether the rate of disinflation is consistent with inflation returning sustainably to target", 0.55),
    
    # =============================================================================
    # TRANSITION / AMBIGUOUS (filling 0.4-0.6 range)
    # =============================================================================
    
    ("CPI inflation has fallen but remains above the 2 per cent target", 0.48),
    ("Pay growth has eased but remains elevated relative to target-consistent levels", 0.52),
    ("The labour market has loosened but conditions remain relatively tight by historical standards", 0.45),
    ("Growth has been subdued but the economy has avoided recession", 0.42),
    ("Inflation has fallen significantly from its peak but domestic price pressures remain", 0.48),
    ("There are encouraging signs of disinflation but services inflation remains sticky", 0.52),
    ("The economy is adjusting to the higher level of interest rates with demand slowing and inflation moderating", 0.45),
    ("GDP growth has picked up somewhat but remains below its potential rate", 0.42),
    ("The recent data on inflation have been mixed with goods prices easing but services prices remaining firm", 0.52),
    ("Labour market indicators present a mixed picture with some signs of loosening alongside continued tightness in certain sectors", 0.48),
]
# %%
# =============================================================================
# TRAIN AU + UK MODELS (10 epochs each, fresh DistilBERT)
# =============================================================================

def train_country_model(training_data, country_code, epochs=10):
    print(f'\n{"="*60}')
    print(f'  Training: {country_code} ({epochs} epochs)')
    print(f'{"="*60}')

    t = [s[0] for s in training_data]
    sc = [s[1] for s in training_data]
    print(f'  Sentences: {len(t)}')

    ds = SentDataset(t, sc)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model = model.to(device)

    args = TrainingArguments(
        output_dir=f'./models/{country_code.lower()}_model',
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        logging_steps=50,
        save_strategy='no',
        use_cpu=True,
    )

    Trainer(model=model, args=args, train_dataset=ds).train()
    model.eval()
    print(f'  {country_code} training complete.')
    return model


model_au = train_country_model(sentence_training_AU, 'AU', epochs=10)
model_uk = train_country_model(sentence_training_boe, 'UK', epochs=10)
# %%
# =============================================================================
# SCORE + REGRESSION FOR AU & UK
# Uses model_au and model_uk trained in the cell above (in memory)
# Same structure as US: sent_level_demeaned + short_spread -> 1y_60d_change
# =============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut


def score_country(text, model, skip_patterns):
    sentences = text.split('.')
    scores = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        if any(skip in sent for skip in skip_patterns):
            continue
        inputs = tokenizer(sent, return_tensors='pt', truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        scores.append(outputs.logits.cpu().numpy()[0][0])
    return np.mean(scores) if scores else 0.5


SKIP_AU = [
    'Voting for', 'Voting against',
    'media inquiries', 'email', 'Communications Department',
    'For release', 'Reserve Bank of Australia SYDNEY',
    'Bullock', 'Lowe', 'Kent', 'Ellis', 'Debelle',
    'Hunter', 'Kohler', 'Hauser', 'phone', 'rbainfo',
]

SKIP_UK = [
    'Voting for', 'Voting against',
    'media inquiries', 'email', 'Bank of England Act',
    'For release', 'Treasury representative',
    'Bailey', 'Broadbent', 'Cunliffe', 'Haskel', 'Mann',
    'Pill', 'Ramsden', 'Dhingra', 'Tenreyro', 'Breeden',
    'Lombardelli', 'Taylor', 'Greene',
    'Sam Beckett', 'David Roberts', 'Jonathan Bewes',
]


def run_country_regression(df, model, skip_patterns, country_code):
    if df.empty:
        print(f'{country_code}: no data')
        return None

    print(f'\nScoring {country_code}...')
    for date, row in df.iterrows():
        if pd.notna(row.get('statement')):
            df.loc[date, 'sent_level'] = score_country(row['statement'], model, skip_patterns)

    df['sent_level'] = df['sent_level'].astype(float)
    df['sent_level_demeaned'] = df['sent_level'] - df['sent_level'].mean()

    target = '1y_60d_change'
    feature_sets = {
        'Sentence-level + spread': ['sent_level_demeaned', 'short_spread'],
        'Spread only': ['short_spread'],
        'Sentence-level only': ['sent_level_demeaned'],
    }

    loo = LeaveOneOut()

    print(f'\n{"="*70}')
    print(f'  {country_code} RESULTS')
    print(f'{"="*70}')
    print(f'  {"Feature Set":<40} {"In R2":>8} {"OOS R2":>8} {"Dir %":>8} {"Obs":>6}')
    print(f'  ' + '-' * 66)

    results = {}
    for name, features in feature_sets.items():
        try:
            md = df[[target] + features].dropna()
            X = md[features]
            y = md[target]
            if len(y) < 5:
                print(f'  {name:<40} SKIP (n={len(y)})')
                continue

            reg_full = LinearRegression()
            reg_full.fit(X, y)
            in_r2 = reg_full.score(X, y)

            preds, acts = [], []
            for train_idx, test_idx in loo.split(X):
                reg = LinearRegression()
                reg.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds.append(reg.predict(X.iloc[test_idx])[0])
                acts.append(y.iloc[test_idx].values[0])

            preds = np.array(preds)
            acts = np.array(acts)
            ss_res = np.sum((acts - preds) ** 2)
            ss_tot = np.sum((acts - acts.mean()) ** 2)
            oos_r2 = 1 - ss_res / ss_tot
            direction = np.mean(np.sign(preds) == np.sign(acts))

            print(f'  {name:<40} {in_r2:>8.3f} {oos_r2:>8.3f} {direction:>7.1%} {len(y):>6}')
            results[name] = {
                'preds': preds, 'acts': acts,
                'oos_r2': oos_r2, 'dir_acc': direction,
                'in_r2': in_r2, 'n': len(y),
            }
        except Exception as e:
            print(f'  {name:<40} ERROR: {e}')

    print(f'{"="*70}')
    return results


# model_au and model_uk come from Cell 20 (training cell above)
au_results = run_country_regression(df_au, model_au, SKIP_AU, 'AU')
uk_results = run_country_regression(df_uk, model_uk, SKIP_UK, 'UK')
# %% [markdown]
# ---
# ## Cross-Country Summary
# %%
def best_result(r):
    if not r:
        return None, None
    best = max(r.values(), key=lambda x: x.get('oos_r2', -99))
    return best['oos_r2'], best['dir_acc']

summary = [
    ('US (GBR)', us_oos_r2, us_dir_acc / 100, fomc_df['1y_60d_change'].notna().sum()),
    ('AU (OLS)', *best_result(au_results), df_au['1y_60d_change'].notna().sum() if len(df_au) > 0 else 0),
    ('UK (OLS)', *best_result(uk_results), df_uk['1y_60d_change'].notna().sum() if len(df_uk) > 0 else 0),
]

print(f'\n{"="*50}')
print(f'CROSS-COUNTRY SUMMARY')
print(f'{"="*50}')
print(f'{"Country":<12} {"Model":<8} {"OOS R2":>10} {"Dir %":>8} {"Obs":>6}')
print('-' * 48)
for cc, r2, da, n in summary:
    r2s = f'{r2:+.3f}' if r2 is not None else '  N/A '
    das = f'{da:.1%}' if da is not None else '  N/A'
    print(f'{cc:<20} {r2s:>10} {das:>8} {n:>6}')
print(f'{"="*50}')
print(f'\nUS uses GBR + walk-forward CV with 7 features (NLP + regime + macro)')
print(f'AU/UK use OLS + LOO CV with 2 features (NLP demeaned + spread)')
# %% [markdown]
# ## Visualization
# %%
regime_colors = {
    'goldilocks': '#2ecc71', 'reflation': '#e67e22', 'overheating': '#e74c3c',
    'stagflation': '#8e44ad', 'transition': '#f1c40f', 'secular_stagnation': '#3498db',
    'deflationary_recession': '#2c3e50',
}

fig, axes = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle('newFedSignal Multi-Country: US (GBR) + AU/UK (OLS)', fontsize=15, fontweight='bold')

# --- US ---
ax0, ax1 = axes[0, 0], axes[0, 1]

# US NLP sentiment bar chart
valid_us = fomc_df['sent_level_demeaned'].dropna()
ax0.bar(valid_us.index, valid_us.values, width=25, color='#3498db', edgecolor='black', linewidth=0.3, alpha=0.8)
ax0.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax0.set_title('US -- Sentence-Level Sentiment (Demeaned)', fontsize=11)
ax0.set_ylabel('Score')

# US predicted vs actual (regime colored)
test_dates = model_df.index[MIN_TRAIN:][:len(us_preds)]
test_reg = fomc_df.loc[test_dates, 'combined_regime'].values

ax1.plot(test_dates, us_acts, 'k-', linewidth=1.8, label='Actual', zorder=3)
ax1.plot(test_dates, us_preds, 'b-', linewidth=1.5, alpha=0.8, label='Predicted')
for i in range(len(test_dates) - 1):
    ax1.axvspan(test_dates[i], test_dates[i+1], alpha=0.15,
                color=regime_colors.get(test_reg[i], '#bdc3c7'))
ax1.axhline(0, color='red', linestyle='--', linewidth=0.8)
ax1.set_title(f'US -- GBR Pred vs Actual (OOS R2={us_oos_r2:+.3f}, Dir={us_dir_acc:.1f}%)', fontsize=11)
ax1.set_ylabel('1Y Yield 60d Change')
ax1.legend(fontsize=9)

# --- AU ---
ax2, ax3 = axes[1, 0], axes[1, 1]
if len(df_au) > 0 and 'sent_level_demeaned' in df_au.columns:
    valid_au = df_au['sent_level_demeaned'].dropna()
    ax2.bar(valid_au.index, valid_au.values, width=25, color='#f39c12', edgecolor='black', linewidth=0.3, alpha=0.8)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_title('AU -- Sentence-Level Sentiment (Demeaned)', fontsize=11)
    ax2.set_ylabel('Score')

    if au_results and 'Sentence-level + spread' in au_results:
        r = au_results['Sentence-level + spread']
        md_au = df_au[['1y_60d_change', 'sent_level_demeaned', 'short_spread']].dropna()
        ax3.plot(md_au.index, r['acts'], label='Actual', color='black', linewidth=1.5)
        ax3.plot(md_au.index, r['preds'], label='Predicted', color='#f39c12', linewidth=1.5, alpha=0.7)
        ax3.axhline(0, color='red', linestyle='--', linewidth=0.8)
        ax3.set_title(f'AU -- Pred vs Actual (OOS R2={r["oos_r2"]:.3f}, Dir={r["dir_acc"]:.1%})', fontsize=11)
        ax3.set_ylabel('Yield 60d Change')
        ax3.legend(fontsize=9)
    elif au_results and 'Sentence-level only' in au_results:
        r = au_results['Sentence-level only']
        md_au = df_au[['1y_60d_change', 'sent_level_demeaned']].dropna()
        ax3.plot(md_au.index, r['acts'], label='Actual', color='black', linewidth=1.5)
        ax3.plot(md_au.index, r['preds'], label='Predicted', color='#f39c12', linewidth=1.5, alpha=0.7)
        ax3.axhline(0, color='red', linestyle='--', linewidth=0.8)
        ax3.set_title(f'AU -- Pred vs Actual (OOS R2={r["oos_r2"]:.3f}, Dir={r["dir_acc"]:.1%})', fontsize=11)
        ax3.set_ylabel('Yield 60d Change')
        ax3.legend(fontsize=9)
else:
    ax2.text(0.5, 0.5, 'AU: No data', ha='center', va='center', transform=ax2.transAxes, color='gray')
    ax3.text(0.5, 0.5, 'AU: No data', ha='center', va='center', transform=ax3.transAxes, color='gray')

# --- UK ---
ax4, ax5 = axes[2, 0], axes[2, 1]
if len(df_uk) > 0 and 'sent_level_demeaned' in df_uk.columns:
    valid_uk = df_uk['sent_level_demeaned'].dropna()
    ax4.bar(valid_uk.index, valid_uk.values, width=25, color='#e74c3c', edgecolor='black', linewidth=0.3, alpha=0.8)
    ax4.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax4.set_title('UK -- Sentence-Level Sentiment (Demeaned)', fontsize=11)
    ax4.set_ylabel('Score')

    if uk_results and 'Sentence-level + spread' in uk_results:
        r = uk_results['Sentence-level + spread']
        md_uk = df_uk[['1y_60d_change', 'sent_level_demeaned', 'short_spread']].dropna()
        ax5.plot(md_uk.index, r['acts'], label='Actual', color='black', linewidth=1.5)
        ax5.plot(md_uk.index, r['preds'], label='Predicted', color='#e74c3c', linewidth=1.5, alpha=0.7)
        ax5.axhline(0, color='red', linestyle='--', linewidth=0.8)
        ax5.set_title(f'UK -- Pred vs Actual (OOS R2={r["oos_r2"]:.3f}, Dir={r["dir_acc"]:.1%})', fontsize=11)
        ax5.set_ylabel('Yield 60d Change')
        ax5.legend(fontsize=9)
    elif uk_results and 'Sentence-level only' in uk_results:
        r = uk_results['Sentence-level only']
        md_uk = df_uk[['1y_60d_change', 'sent_level_demeaned']].dropna()
        ax5.plot(md_uk.index, r['acts'], label='Actual', color='black', linewidth=1.5)
        ax5.plot(md_uk.index, r['preds'], label='Predicted', color='#e74c3c', linewidth=1.5, alpha=0.7)
        ax5.axhline(0, color='red', linestyle='--', linewidth=0.8)
        ax5.set_title(f'UK -- Pred vs Actual (OOS R2={r["oos_r2"]:.3f}, Dir={r["dir_acc"]:.1%})', fontsize=11)
        ax5.set_ylabel('Yield 60d Change')
        ax5.legend(fontsize=9)
else:
    ax4.text(0.5, 0.5, 'UK: No data', ha='center', va='center', transform=ax4.transAxes, color='gray')
    ax5.text(0.5, 0.5, 'UK: No data', ha='center', va='center', transform=ax5.transAxes, color='gray')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()