import numpy as np
import pandas as pd
import pytest
from fedsignal.models.gbr import (
    engineer_features,
    REGIME_ORDINAL,
    REGIME_EXPECTED_SCORE,
    US_FEATURES,
    TARGET,
)


def make_fomc_df(n=15, regime='goldilocks'):
    dates = pd.date_range('2018-01-01', periods=n, freq='6W')
    rng = np.random.default_rng(42)
    sent = rng.uniform(0.3, 0.8, n)
    return pd.DataFrame({
        'combined_regime':    [regime] * n,
        'sent_level':         sent,
        'sent_level_demeaned': sent - sent.mean(),  # normally added by score_fomc_df
        'sent_dispersion':    rng.uniform(0.0, 0.15, n),
        'twoy_ff_spread':     rng.normal(0, 0.5, n),
        'core_pce_yoy':       rng.normal(2.0, 0.5, n),
        '1y_60d_change':      rng.normal(0, 0.4, n),
    }, index=dates)


def test_engineer_features_adds_required_columns():
    # sent_level_demeaned comes from score_fomc_df; engineer_features adds the rest
    df = engineer_features(make_fomc_df())
    for col in ['regime_ordinal', 'nlp_vs_regime', 'nlp_momentum']:
        assert col in df.columns, f'Missing: {col}'


def test_regime_ordinal_maps_correctly():
    for regime, expected_ordinal in REGIME_ORDINAL.items():
        df = engineer_features(make_fomc_df(regime=regime))
        assert (df['regime_ordinal'] == expected_ordinal).all()


def test_nlp_vs_regime_value():
    sent = 0.6
    df = make_fomc_df()
    df['sent_level'] = sent
    df = engineer_features(df)
    expected = sent - REGIME_EXPECTED_SCORE['goldilocks']
    assert np.allclose(df['nlp_vs_regime'], expected)


def test_all_us_features_present_after_engineering():
    df = engineer_features(make_fomc_df())
    missing = [f for f in US_FEATURES if f not in df.columns]
    assert not missing, f'Missing US features: {missing}'


def test_engineer_features_does_not_mutate_input():
    df_orig = make_fomc_df()
    cols_before = set(df_orig.columns)
    engineer_features(df_orig)
    assert set(df_orig.columns) == cols_before


def test_run_walk_forward_shapes():
    from fedsignal.models.gbr import run_walk_forward
    from fedsignal.config import MIN_TRAIN

    df = engineer_features(make_fomc_df(n=30))
    # Add sent_level_demeaned (normally done by score_fomc_df)
    df['sent_level_demeaned'] = df['sent_level'] - df['sent_level'].mean()
    results = run_walk_forward(df)

    n_test = len(results['model_df']) - MIN_TRAIN
    assert len(results['preds']) == n_test
    assert len(results['acts'])  == n_test
    assert len(results['test_dates'])   == n_test
    assert len(results['test_regimes']) == n_test
    assert len(results['importances'])  == len(US_FEATURES)


def test_run_walk_forward_metrics_in_results():
    from fedsignal.models.gbr import run_walk_forward

    df = engineer_features(make_fomc_df(n=30))
    df['sent_level_demeaned'] = df['sent_level'] - df['sent_level'].mean()
    results = run_walk_forward(df)

    for key in ('oos_r2', 'dir_acc', 'mae', 'n_spikes', 'spike_dir', 'spike_capture'):
        assert key in results, f'Missing key in results: {key}'
