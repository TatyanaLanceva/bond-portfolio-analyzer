"""
Unit-тесты для src/core.py (бизнес-логика портфеля).
Внешние API не используются — все данные синтетические.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import pytest

from src.core import (
    load_bonds_data,
    build_portfolio,
    calculate_metrics,
    calculate_full_cashflow,
    drop_amortizing_bonds,
    portfolio_table_view,
    buy_list_export_frame,
    _csv_marked_as_amortizing,
)


# ---------------------------------------------------------------------------
# Фикстуры: синтетические данные облигаций
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bonds():
    data = [
        ("RU000A1017J5", "ОФЗ-ПД 26222", "AAA", 2.0, 980.0, 8.0, 7.5, 1000, "2028-06-15"),
        ("RU000A0JX5W4", "Корп А", "BBB", 1.0, 1020.0, 9.0, 8.0, 1000, "2027-03-10"),
        ("RU000A0JQAM6", "Корп Б", "A", 5.0, 950.0, 11.0, 9.5, 1000, "2031-09-20"),
        ("RU000A0JT6B2", "Корп В", "BB-", 6.0, 880.0, 14.0, 10.0, 1000, "2032-12-01"),
        ("RU000A0JUH3", "Корп Г", "AA", 0.5, 1005.0, 7.0, 6.0, 1000, "2027-01-15"),
        ("RU000A0JXYZ1", "ОФЗ-ПК 26223", "AAA", 3.5, 960.0, 8.5, 7.0, 1000, "2029-08-10"),
        ("RU000A0JABC1", "Корп Д", "B", 10.0, 800.0, 18.0, 12.0, 1000, "2036-05-05"),
    ]
    df = pd.DataFrame(
        data,
        columns=["ISIN", "SHORTNAME", "RATING", "YEARS_TO_MATURITY",
                 "DIRTY_PRICE_RUB", "YTM_PCT", "COUPONPERCENT", "FACEVALUE", "MATDATE"],
    )
    df["MATDATE"] = pd.to_datetime(df["MATDATE"]).dt.normalize()
    return df


@pytest.fixture
def sample_bonds_with_amortizing(sample_bonds):
    df = sample_bonds.copy()
    df["AMORTIZING"] = 0
    df.loc[df["ISIN"] == "RU000A0JX5W4", "AMORTIZING"] = 1
    df.loc[df["ISIN"] == "RU000A0JT6B2", "AMORTIZING"] = 1
    return df
# ---------------------------------------------------------------------------
# Тесты calculate_metrics
# ---------------------------------------------------------------------------

def test_calculate_metrics_empty():
    m = calculate_metrics(pd.DataFrame())
    assert m["Count"] == 0
    assert m["Cost"] == 0.0


def test_calculate_metrics_nonempty(sample_bonds):
    pf = sample_bonds.iloc[:3].copy()
    pf["QUANTITY"] = 10
    pf["INVESTED"] = pf["DIRTY_PRICE_RUB"] * pf["QUANTITY"]
    m = calculate_metrics(pf)
    assert m["Count"] == 3
    assert m["Cost"] > 0
    assert 0 <= m["Yield_Pct"] <= 100


# ---------------------------------------------------------------------------
# Тесты calculate_full_cashflow
# ---------------------------------------------------------------------------

def test_calculate_full_cashflow_empty():
    cf = calculate_full_cashflow(pd.DataFrame())
    assert cf.empty


def test_calculate_full_cashflow_has_columns(sample_bonds):
    pf = sample_bonds.iloc[:2].copy()
    pf["QUANTITY"] = 5
    pf["INVESTED"] = pf["DIRTY_PRICE_RUB"] * pf["QUANTITY"]
    cf = calculate_full_cashflow(pf, tax_rate=0.13)
    expected_cols = {"YEAR", "COUPON", "PRINCIPAL", "TOTAL", "CUMULATIVE"}
    assert expected_cols.issubset(set(cf.columns))


# ---------------------------------------------------------------------------
# Тесты drop_amortizing_bonds
# ---------------------------------------------------------------------------

def test_drop_amortizing_bonds_excludes_marked(sample_bonds_with_amortizing):
    result = drop_amortizing_bonds(sample_bonds_with_amortizing, enabled=True)
    assert "RU000A0JX5W4" not in result["ISIN"].values
    assert "RU000A0JT6B2" not in result["ISIN"].values
    assert "RU000A1017J5" in result["ISIN"].values


def test_drop_amortizing_bonds_disabled_keeps_all(sample_bonds_with_amortizing):
    result = drop_amortizing_bonds(sample_bonds_with_amortizing, enabled=False)
    assert len(result) == len(sample_bonds_with_amortizing)


def test_csv_marked_as_amortizing_values():
    assert _csv_marked_as_amortizing(1) is True
    assert _csv_marked_as_amortizing(0) is False
    assert _csv_marked_as_amortizing("1") is True
    assert _csv_marked_as_amortizing("YES") is True
    assert _csv_marked_as_amortizing("ДА") is True
    assert _csv_marked_as_amortizing(np.nan) is False
    assert _csv_marked_as_amortizing(None) is False


# ---------------------------------------------------------------------------
# Тесты отображения и экспорта
# ---------------------------------------------------------------------------

def test_portfolio_table_view_rounds(sample_bonds):
    pf = sample_bonds.iloc[:2].copy()
    pf["QUANTITY"] = 10
    pf["INVESTED"] = pf["DIRTY_PRICE_RUB"] * pf["QUANTITY"]
    v = portfolio_table_view(pf)
    assert pd.api.types.is_numeric_dtype(v["DIRTY_PRICE_RUB"])


def test_buy_list_export_frame_columns(sample_bonds):
    pf = sample_bonds.iloc[:2].copy()
    pf["QUANTITY"] = 10
    pf["INVESTED"] = pf["DIRTY_PRICE_RUB"] * pf["QUANTITY"]
    ex = buy_list_export_frame(pf)
    expected = {"ISIN", "SHORTNAME", "RATING", "MATDATE", "FACEVALUE", "COUPONPERCENT",
                "YTM_PCT", "YEARS_TO_MATURITY", "DIRTY_PRICE_RUB", "QUANTITY", "INVESTED"}
    assert expected.issubset(set(ex.columns))


# ---------------------------------------------------------------------------
# Тест журнала
# ---------------------------------------------------------------------------

def test_load_journal_missing_file(tmp_path, monkeypatch):
    import src.core as core_mod
    monkeypatch.setattr(core_mod, "JOURNAL_PATH", str(tmp_path / "journal.csv"))
    j = core_mod.load_journal()
    assert j.empty
    assert list(j.columns) == ["DATE", "ISIN", "SHORTNAME", "RATING", "QUANTITY", "PRICE", "INVESTED"]
