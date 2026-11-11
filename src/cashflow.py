"""Денежный поток облигации с учётом амортизации номинала и НДФЛ."""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd


def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return date(y, m, day)


def calculate_full_cashflow(
    purchase_date: date,
    maturity_date: date,
    coupon_rate_pct: float,
    coupon_period_months: int,
    initial_facevalue: float,
    purchase_price_pct: float,
    amortization_schedule: Optional[list[tuple[date, float]]] = None,
    tax_rate: float = 0.13,
) -> pd.DataFrame:
    """
    Возвращает DataFrame с денежными потоками:
        date, type, coupon, principal, outstanding_facevalue, tax, net

    amortization_schedule: [(payment_date, amount_rub), ...] — суммы погашения номинала.
    Купон после амортизации начисляется на остаточный номинал.
    """
    if coupon_rate_pct < 0 or coupon_period_months <= 0:
        raise ValueError("coupon_rate_pct >= 0 и coupon_period_months > 0 обязательны")
    if initial_facevalue <= 0:
        raise ValueError("initial_facevalue должен быть > 0")

    amort_map = {d: amt for d, amt in (amortization_schedule or [])}

    # график купонов
    coupon_dates: list[date] = []
    d = _add_months(purchase_date, coupon_period_months)
    while d <= maturity_date:
        coupon_dates.append(d)
        d = _add_months(d, coupon_period_months)

    # общий таймлайн
    events: list[tuple[date, str]] = [(dt, "coupon") for dt in coupon_dates]
    events += [(dt, "amort") for dt in amort_map]
    events.sort(key=lambda event: (event[0], 0 if event[1] == "coupon" else 1))

    outstanding = initial_facevalue
    rows: list[dict] = []

    # покупка
    purchase_cost = initial_facevalue * purchase_price_pct / 100
    rows.append({
        "date": purchase_date, "type": "purchase",
        "coupon": 0.0, "principal": -purchase_cost,
        "outstanding_facevalue": outstanding,
        "tax": 0.0, "net": -purchase_cost,
    })

    payments_per_year = 12 / coupon_period_months
    for dt, kind in events:
        if kind == "amort":
            amt = min(amort_map[dt], outstanding)
            outstanding -= amt
            rows.append({
                "date": dt, "type": "amortization",
                "coupon": 0.0, "principal": amt,
                "outstanding_facevalue": outstanding,
                "tax": 0.0, "net": amt,
            })
        else:  # coupon
            coupon = outstanding * coupon_rate_pct / 100 / payments_per_year
            tax = coupon * tax_rate
            rows.append({
                "date": dt, "type": "coupon",
                "coupon": coupon, "principal": 0.0,
                "outstanding_facevalue": outstanding,
                "tax": tax, "net": coupon - tax,
            })

    # финальное погашение остатка
    if outstanding > 0:
        rows.append({
            "date": maturity_date, "type": "redemption",
            "coupon": 0.0, "principal": outstanding,
            "outstanding_facevalue": 0.0,
            "tax": 0.0, "net": outstanding,
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def calculate_dirty_price(
    facevalue: float,
    price_pct: float,
    accrued_interest: float = 0.0,
) -> float:
    '''Dirty price of bond (% of nominal + accrued interest).'''
    return round(facevalue * price_pct / 100.0 + accrued_interest, 2)


def calculate_accrued_interest(
    facevalue: float,
    coupon_rate_pct: float,
    days_since_last_coupon: int,
    days_in_period: int,
) -> float:
    '''Accrued interest (NCD) in rubles.'''
    if days_in_period <= 0:
        return 0.0
    return round(facevalue * coupon_rate_pct / 100.0 * days_since_last_coupon / days_in_period, 2)


def calculate_duration(
    cashflows: list,
    ytm_pct: float,
) -> float:
    '''Simplified modified duration from event-based cashflows.'''
    if not cashflows or ytm_pct is None:
        return 0.0
    total_pv = 0.0
    weighted_time = 0.0
    for cf in cashflows:
        t = cf.get('year', 0)
        pv = cf.get('net', 0.0) / (1.0 + ytm_pct / 100.0) ** t
        total_pv += pv
        weighted_time += pv * t
    if total_pv <= 0:
        return 0.0
    return round(weighted_time / total_pv, 4)


def calculate_cashflows(
    purchase_date,
    maturity_date,
    coupon_rate_pct: float,
    coupon_period_months: int,
    initial_facevalue: float,
    purchase_price_pct: float,
    amortization_schedule=None,
    tax_rate: float = 0.13,
):
    '''Alias on calculate_full_cashflow for backwards compatibility.'''
    return calculate_full_cashflow(
        purchase_date=purchase_date,
        maturity_date=maturity_date,
        coupon_rate_pct=coupon_rate_pct,
        coupon_period_months=coupon_period_months,
        initial_facevalue=initial_facevalue,
        purchase_price_pct=purchase_price_pct,
        amortization_schedule=amortization_schedule,
        tax_rate=tax_rate,
    )


def calculate_dirty_price(facevalue: float, price_pct: float, accrued_interest: float = 0.0) -> float:
    return round(facevalue * price_pct / 100.0 + accrued_interest, 2)

def calculate_accrued_interest(facevalue: float, coupon_rate_pct: float, days_since_last_coupon: int, days_in_period: int) -> float:
    if days_in_period <= 0: return 0.0
    return round(facevalue * coupon_rate_pct / 100.0 * days_since_last_coupon / days_in_period, 2)

def calculate_duration(cashflows: list, ytm_pct: float) -> float:
    if not cashflows or ytm_pct is None: return 0.0
    total_pv = 0.0; weighted_time = 0.0
    for cf in cashflows:
        t = cf.get('year', 0)
        pv = cf.get('net', 0.0) / (1.0 + ytm_pct / 100.0) ** t
        total_pv += pv; weighted_time += pv * t
    if total_pv <= 0: return 0.0
    return round(weighted_time / total_pv, 4)

def calculate_cashflows(purchase_date, maturity_date, coupon_rate_pct: float, coupon_period_months: int, initial_facevalue: float, purchase_price_pct: float, amortization_schedule=None, tax_rate: float = 0.13):
    return calculate_full_cashflow(purchase_date=purchase_date, maturity_date=maturity_date, coupon_rate_pct=coupon_rate_pct, coupon_period_months=coupon_period_months, initial_facevalue=initial_facevalue, purchase_price_pct=purchase_price_pct, amortization_schedule=amortization_schedule, tax_rate=tax_rate)
