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
    events.sort()

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