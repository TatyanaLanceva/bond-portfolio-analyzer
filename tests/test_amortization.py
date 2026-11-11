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
def test_coupon_before_amortization_on_same_date():
    from datetime import date
    from src.cashflow import calculate_full_cashflow

    flows = calculate_full_cashflow(
        purchase_date=date(2026, 1, 1),
        maturity_date=date(2027, 1, 1),
        coupon_rate_pct=12,
        coupon_period_months=6,
        initial_facevalue=1000,
        purchase_price_pct=100,
        amortization_schedule=[(date(2026, 7, 1), 500)],
        tax_rate=0,
    )

    first_coupon = flows[
        (flows["date"] == date(2026, 7, 1))
        & (flows["type"] == "coupon")
    ]

    assert len(first_coupon) == 1
    assert first_coupon.iloc[0]["coupon"] == 60.0
    assert flows["net"].sum() == 90.0