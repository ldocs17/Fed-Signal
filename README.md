# FedSignal BackEnd

A machine learning pipeline that predicts short-term US Treasury yield changes following Federal Reserve (FOMC) meetings. It combines NLP sentiment analysis of FOMC statements with macroeconomic regime classification and a Gradient Boosting Regressor to forecast the 1-year yield change ~60 trading days after each meeting.

---

## How It Works

The pipeline runs in six stages:

1. **FRED Data** — Downloads 15 macroeconomic series from the Federal Reserve Economic Data API (GDP growth, CPI, core PCE, unemployment, yield curves, Fed Funds rate, etc.) and caches them locally.

2. **Regime Classification** — Classifies each month into one of seven economic regimes (goldilocks, reflation, overheating, stagflation, transition, secular stagnation, deflationary recession) based on real GDP growth and core PCE inflation.

3. **FOMC Statement Scraping** — Scrapes monetary policy press releases from federalreserve.gov for meetings from 2015 to present and caches them locally.

4. **NLP Scoring** — Fine-tunes a DistilBERT model on 88 hand-labeled hawkish/dovish sentences from FOMC language, then scores each statement sentence-by-sentence to produce a hawkishness score (0 = very dovish, 1 = very hawkish).

5. **Feature Engineering** — Builds seven model features from the NLP scores and macro data:

   | Feature | Description |
   |---|---|
   | `twoy_ff_spread` | 2-year yield minus Fed Funds rate |
   | `core_pce_yoy` | Core PCE year-over-year inflation |
   | `sent_level_demeaned` | NLP hawkishness score (demeaned) |
   | `nlp_vs_regime` | NLP score minus regime-expected score (surprise) |
   | `nlp_momentum` | NLP score deviation from 4-meeting EWMA |
   | `sent_dispersion` | Standard deviation of sentence-level scores |
   | `regime_ordinal` | Economic regime encoded as ordinal (0–6) |

6. **GBR Walk-Forward** — Trains a `GradientBoostingRegressor` using walk-forward cross-validation (minimum 20 meetings of training data) and evaluates out-of-sample predictions on every subsequent meeting.

---

## Project Structure

```
FedSignal_BackEnd/
├── fedsignal/
│   ├── config.py               # API keys, model constants, cache paths, skip patterns
│   ├── data/
│   │   ├── fred_loader.py      # FRED download + caching, regime classification
│   │   ├── fomc_scraper.py     # federalreserve.gov scraping + caching, fomc_df builder
│   │   └── sentences_us.json   # 88 labeled training sentences (hawkishness scores)
│   ├── models/
│   │   ├── nlp.py              # DistilBERT fine-tuning and statement scoring
│   │   └── gbr.py              # Feature engineering, walk-forward CV, metrics
│   └── visualize.py            # Matplotlib chart (sentiment bar + pred vs actual)
├── run.py                      # End-to-end pipeline orchestrator
├── tests/
│   ├── test_regime.py          # Unit tests for regime classification logic
│   ├── test_features.py        # Unit tests for feature engineering and walk-forward
│   └── test_nlp.py             # Unit tests for SentDataset, scoring, JSON schema
└── data/                       # Auto-created cache directory (fred_data.pkl, fomc_statements.json)
```

---

## Setup

**Requirements:** Python 3.8+

Install dependencies:

```bash
pip install pandas numpy torch transformers scikit-learn matplotlib requests beautifulsoup4 fredapi
```

Set your FRED API key in `fedsignal/config.py`:

```python
FRED_KEY = 'your_fred_api_key_here'
```

> Free FRED API keys are available at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

---

## Running

```bash
python run.py
```

On first run the pipeline downloads FRED data and scrapes FOMC statements (~5–10 minutes with rate limiting). Both are cached to the `data/` directory — subsequent runs load from cache in seconds.

To force a re-download, set `FORCE_REFRESH = True` in `fedsignal/config.py`.

---

## Output

**Console metrics:**
```
============================================================
US MODEL PERFORMANCE (GBR + Walk-Forward)
============================================================
Model:            GradientBoostingRegressor
Features:         7
OOS R2:           +0.XXX
Dir Accuracy:     XX.X%
MAE:              0.XXX

Feature Importance (Gini):
  twoy_ff_spread            0.XXX ||||||||||||
  sent_level_demeaned       0.XXX ||||||||
  ...

Regime-Stratified Directional Accuracy:
  goldilocks                   N     XX.X%
  reflation                    N     XX.X%
  ...
```

**Chart:** A 1×2 matplotlib figure showing FOMC NLP sentiment over time (left) and walk-forward predicted vs actual yield changes with regime-colored background (right).

---

## Testing

```bash
python -m pytest tests/ -v
```

The test suite covers regime classification boundary conditions, feature engineering correctness, walk-forward output shapes, `SentDataset` behavior, statement scoring with mocked models, and the training sentence JSON schema. No API keys or model training required — all tests run with synthetic or mocked data.

---

## Caching Behavior

| File | Contents | Refreshed when |
|---|---|---|
| `data/fred_data.pkl` | 15 FRED series | `FORCE_REFRESH = True` |
| `data/fomc_statements.json` | FOMC press release text | `FORCE_REFRESH = True` |

The DistilBERT model is re-trained on every run (30 epochs, ~2–5 min on CPU). To persist it across runs, save and reload the model from `./fed_sentence_model/` after training.

---

## Key Configuration (`fedsignal/config.py`)

| Setting | Default | Description |
|---|---|---|
| `FRED_KEY` | `'FRED_API'` | Your FRED API key |
| `MODEL_NAME` | `'distilbert-base-uncased'` | HuggingFace model for NLP |
| `HORIZON` | `60` | Trading days after meeting for target yield change |
| `MIN_TRAIN` | `20` | Minimum meetings before walk-forward predictions begin |
| `FORCE_REFRESH` | `False` | Set to `True` to re-download all cached data |
