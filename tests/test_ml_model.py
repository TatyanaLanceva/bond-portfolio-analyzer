"""
Unit-тесты для src/ml_model.py (ML-прогноз доходности).
Внешние файлы модели (.pkl) не требуются — используется мок.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src.ml_model import (
    FEATURES,
    _ensure_lastprice,
    _add_engineered_features,
    predict_ytm,
    enrich_with_ml_scores,
    load_model,
)


# ---------------------------------------------------------------------------
# Фикстура: синтетические данные для ML
# ---------------------------------------------------------------------------

@pytest.fixture
def ml_sample_df():
    return pd.DataFrame([
        {
            "ISIN": "RU000A1017J5", "SHORTNAME": "ОФЗ-ПД", "RATING": "AAA",
            "YEARS_TO_MATURITY": 2.0, "COUPONPERCENT": 7.5, "LASTPRICE": 98.0,
            "RATING_SCORE": 0, "FACEVALUE": 1000.0, "IS_GOVERNMENT": 1,
            "MOD_DURATION": 1.9, "COUPONS_REMAINING": 4, "YTM_PCT": 8.0,
            "DIRTY_PRICE_RUB": 980.0, "MATDATE": pd.Timestamp("2028-06-15"),
        },
        {
            "ISIN": "RU000A0JQAM6", "SHORTNAME": "Корп Б", "RATING": "A",
            "YEARS_TO_MATURITY": 5.0, "COUPONPERCENT": 9.5, "LASTPRICE": 95.0,
            "RATING_SCORE": 5, "FACEVALUE": 1000.0, "IS_GOVERNMENT": 0,
            "MOD_DURATION": 4.5, "COUPONS_REMAINING": 10, "YTM_PCT": 11.0,
            "DIRTY_PRICE_RUB": 950.0, "MATDATE": pd.Timestamp("2031-09-20"),
        },
        {
            "ISIN": "RU000A0JT6B2", "SHORTNAME": "Корп В", "RATING": "BB-",
            "YEARS_TO_MATURITY": 6.0, "COUPONPERCENT": 10.0, "LASTPRICE": 88.0,
            "RATING_SCORE": 12, "FACEVALUE": 1000.0, "IS_GOVERNMENT": 0,
            "MOD_DURATION": 5.2, "COUPONS_REMAINING": 12, "YTM_PCT": 14.0,
            "DIRTY_PRICE_RUB": 880.0, "MATDATE": pd.Timestamp("2032-12-01"),
        },
    ])

# ---------------------------------------------------------------------------
# Тесты _ensure_lastprice
# ---------------------------------------------------------------------------

def test_ensure_lastprice_existing(ml_sample_df):
    out = _ensure_lastprice(ml_sample_df)
    assert pd.api.types.is_numeric_dtype(out["LASTPRICE"])
    assert out.loc[0, "LASTPRICE"] == 98.0


def test_ensure_lastprice_missing(ml_sample_df):
    df = ml_sample_df.copy().drop(columns=["LASTPRICE"])
    out = _ensure_lastprice(df)
    expected = 980.0 / 1000.0 * 100.0
    assert abs(out.loc[0, "LASTPRICE"] - expected) < 0.01


# ---------------------------------------------------------------------------
# Тесты _add_engineered_features
# ---------------------------------------------------------------------------

def test_add_engineered_features_columns(ml_sample_df):
    out = _add_engineered_features(ml_sample_df)
    for col in ["IS_GOVERNMENT", "MOD_DURATION", "COUPONS_REMAINING"]:
        assert col in out.columns


def test_add_engineered_features_is_government(ml_sample_df):
    out = _add_engineered_features(ml_sample_df)
    ofz = out[out["SHORTNAME"].str.startswith("ОФЗ")]
    assert (ofz["IS_GOVERNMENT"] == 1).all()
    corp = out[~out["SHORTNAME"].str.startswith("ОФЗ")]
    assert (corp["IS_GOVERNMENT"] == 0).all()


def test_add_engineered_features_mod_duration(ml_sample_df):
    out = _add_engineered_features(ml_sample_df)
    assert out["MOD_DURATION"].notna().all()
    assert (out["MOD_DURATION"] >= 0).all()
    assert np.isfinite(out["MOD_DURATION"]).all()


# ---------------------------------------------------------------------------
# Тесты predict_ytm с моком модели
# ---------------------------------------------------------------------------

def test_predict_ytm_returns_array(ml_sample_df):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([8.5, 10.5, 13.5])
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.ones((3, len(FEATURES)))

    result = predict_ytm(ml_sample_df, mock_model, mock_scaler, FEATURES)
    assert len(result) == 3
    mock_model.predict.assert_called_once()


def test_predict_ytm_missing_features_filled(ml_sample_df):
    df = ml_sample_df.copy().drop(columns=["MOD_DURATION"])
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([8.0, 9.0, 10.0])
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.ones((3, len(FEATURES)))

    result = predict_ytm(df, mock_model, mock_scaler, FEATURES)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Тесты enrich_with_ml_scores
# ---------------------------------------------------------------------------

def test_enrich_with_ml_scores_adds_columns(ml_sample_df):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([8.0, 10.0, 13.0])
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.ones((3, len(FEATURES)))

    result = enrich_with_ml_scores(ml_sample_df, mock_model, mock_scaler, FEATURES)
    for col in ["PREDICTED_YTM", "EXCESS_YTM", "ML_GROUP", "ML_RANK_IN_GROUP"]:
        assert col in result.columns
    assert len(result) == 3


def test_enrich_with_ml_scores_excess_calculation(ml_sample_df):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([8.0, 10.0, 13.0])
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.ones((3, len(FEATURES)))

    result = enrich_with_ml_scores(ml_sample_df, mock_model, mock_scaler, FEATURES)
    expected_excess = ml_sample_df["YTM_PCT"].values - np.array([8.0, 10.0, 13.0])
    np.testing.assert_array_almost_equal(result["EXCESS_YTM"].values, expected_excess)


# ---------------------------------------------------------------------------
# Тест load_model (файлы .pkl могут не существовать — проверяем обработку)
# ---------------------------------------------------------------------------

@patch("src.ml_model.MODEL_PATH", Path("nonexistent_model.pkl"))
@patch("src.ml_model.SCALER_PATH", Path("nonexistent_scaler.pkl"))
def test_load_model_missing_files_raises():
    with pytest.raises(FileNotFoundError):
        load_model()
