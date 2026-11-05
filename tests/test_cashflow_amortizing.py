"""Тесты для модуля cashflow с амортизацией."""

from datetime import date

import pandas as pd
import pytest

from src.cashflow import calculate_full_cashflow


def test_amortization_reduces_coupon_after_payment():
    sched = [(date(2027, 1, 1), 500.0)]  # половина номинала погашена
    cf = calculate_full_cashflow(
        purchase_date=date(2026, 1, 1),
        maturity_date=date(2028, 1, 1),
        coupon_rate_pct=10.0,
        coupon_period_months=6,
        initial_facevalue=1000.0,
        purchase_price_pct=100.0,
        amortization_schedule=sched,
    )
    coupons = cf[cf["type"] == "coupon"].sort_values("date").reset_index(drop=True)
    # До амортизации coupon = 1000*0.1/2 = 50
    assert coupons[coupons["date"] < date(2027, 1, 1)]["coupon"].iloc[0] == 50.0
    # После амортизации coupon = 500*0.1/2 = 25
    after = coupons[coupons["date"] > date(2027, 1, 1)]
    assert (after["coupon"] == 25.0).all()


def test_redemption_returns_remaining_facevalue():
    sched = [(date(2027, 1, 1), 500.0)]
    cf = calculate_full_cashflow(
        purchase_date=date(2026, 1, 1),
        maturity_date=date(2028, 1, 1),
        coupon_rate_pct=10.0,
        coupon_period_months=12,
        initial_facevalue=1000.0,
        purchase_price_pct=100.0,
        amortization_schedule=sched,
    )
    redemption = cf[cf["type"] == "redemption"].iloc[0]
    assert redemption["principal"] == 500.0