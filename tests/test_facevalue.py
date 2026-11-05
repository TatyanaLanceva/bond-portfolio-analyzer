"""Тесты для модуля facevalue."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.facevalue import enrich_facevalue, fetch_facevalue_from_moex_securities


@pytest.fixture
def csv_df():
    return pd.DataFrame({
        "ISIN": ["RU000A100001", "RU000A100002", "RU000A100003"],
        "SHORTNAME": ["A", "B", "C"],
        "MATDATE": ["2030-01-01"] * 3,
        "YTM_PCT": [10.0, 11.0, 12.0],
        "FACEVALUE": [1000.0, None, None],
    })


def test_csv_facevalue_kept(csv_df, tmp_path):
    out = enrich_facevalue(csv_df, cache_path=tmp_path / "c.json")
    assert out.loc[out["ISIN"] == "RU000A100001", "FACEVALUE_SOURCE"].iloc[0] == "csv"


@patch("src.facevalue.requests.get")
def test_moex_success(mock_get, csv_df, tmp_path):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"securities": {
            "columns": ["FACEVALUE", "FACEUNIT"], "data": [[5000.0, "SUR"]]}},
    )
    mock_get.return_value.raise_for_status = lambda: None
    out = enrich_facevalue(csv_df, cache_path=tmp_path / "c.json", use_cache=False)
    b = out.loc[out["ISIN"] == "RU000A100002"].iloc[0]
    assert b["FACEVALUE"] == 5000.0
    assert b["FACEVALUE_SOURCE"] == "moex_iss"


@patch("src.facevalue.requests.get")
def test_moex_error_excludes_bond_not_fallback_to_1000(mock_get, csv_df, tmp_path):
    mock_get.side_effect = __import__("requests").ConnectionError("boom")
    out = enrich_facevalue(csv_df, cache_path=tmp_path / "c.json", use_cache=False)
    # бумаги без номинала исключены, 1000.0 нигде нет
    assert "RU000A100002" not in out["ISIN"].values
    assert "RU000A100003" not in out["ISIN"].values
    assert 1000.0 not in out["FACEVALUE"].values


@patch("src.facevalue.requests.get")
def test_non_rub_excluded(mock_get, csv_df, tmp_path):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"securities": {
            "columns": ["FACEVALUE", "FACEUNIT"], "data": [[1000.0, "USD"]]}},
    )
    mock_get.return_value.raise_for_status = lambda: None
    out = enrich_facevalue(csv_df, cache_path=tmp_path / "c.json", use_cache=False)
    assert "RU000A100002" not in out["ISIN"].values