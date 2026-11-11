"""Тесты для модуля moex_amort."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.moex_amort import check_amortization_bulk, drop_amortizing_bonds, AmortizationInfo


@patch("src.moex_amort.requests.get")
def test_amortizing_detected(mock_get):
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"amortizations": {
            "columns": ["maturity_date", "value_rub"], "data": [["2026-01-01", 100.0]]}},
    )
    mock_get.return_value.raise_for_status = lambda: None
    info = check_amortization_bulk(["RU000A100001"])["RU000A100001"]
    assert info.status == "amortizing"


@patch("src.moex_amort.requests.get")
def test_network_error_is_unknown_not_false(mock_get):
    mock_get.side_effect = __import__("requests").Timeout("boom")
    info = check_amortization_bulk(["RU000A100001"])["RU000A100001"]
    assert info.status == "unknown"  # НЕ 'not_amortizing'


def test_filter_raises_when_enabled_but_no_data():
    df = pd.DataFrame({"ISIN": ["A", "B"]})
    with pytest.raises(ValueError):
        drop_amortizing_bonds(df, enabled=True, amortizing_isins=None)


def test_filter_warns_when_soft():
    import pandas as pd
    df = pd.DataFrame({"ISIN": ["A", "B"]})
    out = drop_amortizing_bonds(
        df, enabled=True, amortizing_isins=None, fail_on_missing_data=False
    )
    assert len(out) == len(df)  # ничего не потеряли, но warning в логе
