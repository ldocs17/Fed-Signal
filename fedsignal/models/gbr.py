import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fedsignal.config import MIN_TRAIN

REGIME_ORDINAL = {
    'deflationary_recession': 0,
    'secular_stagnation':     1,
    'goldilocks':             2,
    'transition':             3,
    'reflation':              4,
    'overheating':            5,
    'stagflation':            6,
}

REGIME_EXPECTED_SCORE = {
    'deflationary_recession': 0.20,
    'secular_stagnation':     0.30,
    'goldilocks':             0.45,
    'transition':             0.50,
    'reflation':              0.62,
    'overheating':            0.78,
    'stagflation':            0.72,
}

US_FEATURES = [
    'twoy_ff_spread',
    'core_pce_yoy',
    'sent_level_demeaned',
    'nlp_vs_regime',
    'nlp_momentum',
    'sent_dispersion',
    'regime_ordinal',
]

TARGET = '1y_60d_change'


def engineer_features(fomc_df):
    fomc_df = fomc_df.copy()
    fomc_df['regime_ordinal']    = fomc_df['combined_regime'].map(REGIME_ORDINAL)
    fomc_df['regime_expected_nlp'] = fomc_df['combined_regime'].map(REGIME_EXPECTED_SCORE)
    fomc_df['nlp_vs_regime']     = fomc_df['sent_level'] - fomc_df['regime_expected_nlp']

    fomc_df = fomc_df.sort_index()
    fomc_df['nlp_ewma']      = fomc_df['sent_level'].ewm(span=4, adjust=False).mean()
    fomc_df['nlp_momentum']  = fomc_df['sent_level'] - fomc_df['nlp_ewma']
    return fomc_df


def _build_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('gbr', GradientBoostingRegressor(
            n_estimators=80, max_depth=2, min_samples_leaf=5,
            learning_rate=0.05, subsample=0.8, random_state=42,
        )),
    ])


def run_walk_forward(fomc_df):
    model_df = fomc_df[US_FEATURES + [TARGET]].dropna().sort_index()
    X = model_df[US_FEATURES].values
    y = model_df[TARGET].values

    base_model = _build_model()
    preds, acts = [], []
    for t in range(MIN_TRAIN, len(y)):
        m = clone(base_model)
        m.fit(X[:t], y[:t])
        preds.append(m.predict(X[t:t+1])[0])
        acts.append(y[t])

    preds = np.array(preds)
    acts  = np.array(acts)

    m_full = clone(base_model)
    m_full.fit(X, y)
    importances = m_full.named_steps['gbr'].feature_importances_

    ss_res  = np.sum((acts - preds) ** 2)
    ss_tot  = np.sum((acts - acts.mean()) ** 2)
    oos_r2  = 1 - ss_res / ss_tot
    dir_acc = float(np.mean(np.sign(preds) == np.sign(acts)) * 100)
    mae     = float(np.mean(np.abs(acts - preds)))

    spike_mask    = np.abs(acts) > 0.5
    n_spikes      = int(spike_mask.sum())
    spike_dir     = float(np.mean(np.sign(preds[spike_mask]) == np.sign(acts[spike_mask])) * 100) if n_spikes >= 3 else float('nan')
    spike_capture = float(np.mean(np.abs(preds[spike_mask]) / np.abs(acts[spike_mask])) * 100)     if n_spikes >= 3 else float('nan')

    test_idx     = model_df.index[MIN_TRAIN:]
    test_dates   = test_idx[:len(preds)]
    test_regimes = fomc_df.loc[test_idx, 'combined_regime'].values[:len(preds)]

    return {
        'model_df':     model_df,
        'preds':        preds,
        'acts':         acts,
        'importances':  importances,
        'oos_r2':       oos_r2,
        'dir_acc':      dir_acc,
        'mae':          mae,
        'n_spikes':     n_spikes,
        'spike_dir':    spike_dir,
        'spike_capture': spike_capture,
        'test_dates':   test_dates,
        'test_regimes': test_regimes,
    }


def print_metrics(results):
    preds        = results['preds']
    acts         = results['acts']
    model_df     = results['model_df']
    importances  = results['importances']
    oos_r2       = results['oos_r2']
    dir_acc      = results['dir_acc']
    mae          = results['mae']
    n_spikes     = results['n_spikes']
    spike_dir    = results['spike_dir']
    spike_capture = results['spike_capture']
    test_regimes = results['test_regimes']

    print(f'\n{"="*60}')
    print(f'US MODEL PERFORMANCE (GBR + Walk-Forward)')
    print(f'{"="*60}')
    print(f'Model:            GradientBoostingRegressor')
    print(f'Features:         {len(US_FEATURES)}')
    print(f'Observations:     {len(model_df)}')
    print(f'Test predictions: {len(preds)} (walk-forward, min_train={MIN_TRAIN})')
    print(f'')
    print(f'OOS R2:           {oos_r2:+.3f}')
    print(f'Dir Accuracy:     {dir_acc:.1f}%')
    print(f'MAE:              {mae:.3f}')
    print(f'')
    print(f'Spikes (|change|>0.5%): {n_spikes}')
    print(f'Spike Dir Acc:    {spike_dir:.1f}%')
    print(f'Spike Capture:    {spike_capture:.0f}%')
    print(f'{"="*60}')

    print(f'\nFeature Importance (Gini):')
    for feat, imp in sorted(zip(US_FEATURES, importances), key=lambda x: -x[1]):
        bar = '|' * int(imp * 80)
        print(f'  {feat:<25} {imp:.3f} {bar}')

    print(f'\nRegime-Stratified Directional Accuracy:')
    print(f'{"Regime":<25} {"N":>5} {"Dir Acc":>10}')
    print('-' * 45)
    for regime in sorted(set(test_regimes)):
        mask = np.array(test_regimes) == regime
        if mask.sum() >= 3:
            acc = np.mean(np.sign(preds[mask]) == np.sign(acts[mask])) * 100
            print(f'{regime:<25} {mask.sum():>5} {acc:>9.1f}%')

    print(f'\nFeature correlations with target:')
    for feat in US_FEATURES:
        corr = model_df[feat].corr(model_df[TARGET])
        print(f'  {feat:<25} {corr:+.3f}')
