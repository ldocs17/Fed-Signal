import pandas as pd
import pytest
from fedsignal.data.fred_loader import classify_economic_regime


def row(gdp, inflation):
    return pd.Series({'GDP Growth': gdp, 'core_pce_yoy': inflation})


def test_stagflation():
    assert classify_economic_regime(row(-1.0, 4.0)) == 'stagflation'


def test_deflationary_recession():
    assert classify_economic_regime(row(-1.0, 1.0)) == 'deflationary_recession'


def test_overheating():
    assert classify_economic_regime(row(4.0, 4.0)) == 'overheating'


def test_goldilocks():
    assert classify_economic_regime(row(2.5, 2.0)) == 'goldilocks'


def test_secular_stagnation():
    assert classify_economic_regime(row(1.0, 1.5)) == 'secular_stagnation'


def test_reflation():
    assert classify_economic_regime(row(2.0, 3.0)) == 'reflation'


def test_transition():
    # gdp < 1.5 and inflation >= 2.0, none of the other branches
    assert classify_economic_regime(row(1.0, 2.5)) == 'transition'


def test_all_regimes_covered():
    from fedsignal.models.gbr import REGIME_ORDINAL
    regimes = {
        classify_economic_regime(row(-1.0, 4.0)),
        classify_economic_regime(row(-1.0, 1.0)),
        classify_economic_regime(row(4.0, 4.0)),
        classify_economic_regime(row(2.5, 2.0)),
        classify_economic_regime(row(1.0, 1.5)),
        classify_economic_regime(row(2.0, 3.0)),
        classify_economic_regime(row(1.0, 2.5)),
    }
    assert regimes == set(REGIME_ORDINAL.keys())
