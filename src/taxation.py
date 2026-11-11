"""Налоговые расчёты для денежных потоков облигаций.

Модуль отделяет налоговый движок от финансового расчёта:
- купонный доход облагается ставкой в соответствии с режимом;
- НКД (налог на капитал) рассчитывается по разнице между ценой продажи и
  накопленной купонной доходностью;
- финансовый результат от продажи и погашения обрабатывается в рамках
  выбранного налогового режима;
- комиссии и иные расходы учитываются в отдельности.

Уровень налогообложения и правила выбираются при создании объекта
``TaxScenario``; встроенные режимы не являются универсальными.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class TaxScenario:
    """Конфигурация налогового сценария.

    Attributes:
        regime: Наименование режима (например ``"IPF"``, ``"USN"``).
        coupon_rate_pct: Ставка по купону (в процентах), если применима.
        nkd_rate_pct: Ставка НКД (в процентах), если применима.
        sale_rate_pct: Ставка по результату от продажи, если применима.
        payment_rate_pct: Ставка по результату погашения.
        fee_rate_pct: Ставка комиссии (в процентах от суммы сделки).
        effective_date: Дата вступления режима в силу.
    """

    regime: str
    coupon_rate_pct: float = 0.0
    nkd_rate_pct: float = 0.0
    sale_rate_pct: float = 0.0
    payment_rate_pct: float = 0.0
    fee_rate_pct: float = 0.0
    effective_date: date = date(1970, 1, 1)

    # ────────────
    # Вспомогательные ставки
    # ────────────
    @property
    def coupon_rate(self) -> float:
        """Ставка купона в долях (0.0–1.0)."""
        return self.coupon_rate_pct / 100.0

    @property
    def nkd_rate(self) -> float:
        """Ставка НКД в долях."""
        return self.nkd_rate_pct / 100.0

    @property
    def sale_rate(self) -> float:
        """Ставка по результату от продажи."""
        return self.sale_rate_pct / 100.0

    @property
    def payment_rate(self) -> float:
        """Ставка по результату погашения."""
        return self.payment_rate_pct / 100.0

    @property
    def fee_rate(self) -> float:
        """Ставка комиссии в долях."""
        return self.fee_rate_pct / 100.0


# ────────────
# Финансовый движок с учётом налога
# ────────────
def apply_tax_to_coupon(
    coupon_net: float,
    scenario: TaxScenario,
) -> float:
    """Определить налог на купонный доход.

    Args:
        coupon_net: Гросс-сумма купонного дохода (до налогообложения).
        scenario: Конфигурация налогового сценария.

    Returns:
        float: Сумма налога (0, если режим не применяется).
    """
    if scenario.coupon_rate_pct <= 0:
        return 0.0
    return round(coupon_net * scenario.coupon_rate, 2)


def apply_nkd(
    gross_amount: float,
    basis: float,
    nkd_rate: float,
) -> float:
    """Рассчитать налог на капитал (НКД).

    Args:
        gross_amount: Количество денег, полученное от продажи/погашения.
        basis: Сумма, на которую продан/погашён активы (включая накопленный
            купонный доход).
        nkd_rate: Ставка НКД в долях.

    Returns:
        float: Сумма налога, округлённая до 2 знаков.
    """
    if nkd_rate <= 0 or gross_amount <= basis:
        return 0.0
    return round((gross_amount - basis) * nkd_rate, 2)


def apply_fee(
    amount: float,
    fee_rate: float,
) -> float:
    """Рассчитать условную комиссию.

    Args:
        amount: Сумма сделки.
        fee_rate: Ставка комиссии в долях.

    Returns:
        float: Сумма комиссии, округлённая до 2 знаков.
    """
    if fee_rate <= 0:
        return 0.0
    return round(amount * fee_rate, 2)


# ────────────
# Расчёт операционного потока
# ────────────
def compute_cashflow_net(
    gross_amount: float,
    tax_amount: float,
    fee_amount: float = 0.0,
) -> float:
    """Получить чистую сумму операционного потока.

    Args:
        gross_amount: Гросс-сумма.
        tax_amount: Сумма налога.
        fee_amount: Сумма комиссии.

    Returns:
        float: Чистая сумма (округлённая до 2 знаков).
    """
    return round(gross_amount - tax_amount - fee_amount, 2)


# ────────────
# Выбор режима по умолчанию
# ────────────
def default_scenarios() -> dict[str, TaxScenario]:
    """Вернуть набор заданных налоговых сценариев.

    Returns:
        dict[str, TaxScenario]: Словарь ``имя-режима → сценарий``.
    """
    return {
        "0_percent": TaxScenario(
            regime="0% (ИИС / льгота)",
            coupon_rate=0.0,
            nkd_rate=0.0,
            sale_rate=0.0,
            payment_rate=0.0,
            fee_rate=0.0,
            effective_date=date(2026, 1, 1),
        ),
        "13_percent_ndfl": TaxScenario(
            regime="13% (НДФЛ)",
            coupon_rate=13.0,
            nkd_rate=13.0,
            sale_rate=13.0,
            payment_rate=13.0,
            fee_rate=0.0,
            effective_date=date(2026, 1, 1),
        ),
        "15_percent_usn": TaxScenario(
            regime="15% (УСН)",
            coupon_rate=15.0,
            nkd_rate=15.0,
            sale_rate=15.0,
            payment_rate=15.0,
            fee_rate=0.0,
            effective_date=date(2026, 1, 1),
        ),
    }


def calculate_tax(
    coupon_net: float,
    scenario: TaxScenario,
) -> float:
    return apply_tax_to_coupon(coupon_net, scenario)


def calculate_taxable_income(
    coupon_gross: float,
    nkd_rate: float = 0.0,
    basis: float = 0.0,
) -> float:
    taxable = coupon_gross
    if nkd_rate > 0 and coupon_gross > basis:
        taxable -= (coupon_gross - basis) * nkd_rate
    return round(max(taxable, 0.0), 2)


def calculate_withholding_tax(
    amount: float,
    rate: float,
) -> float:
    if rate <= 0 or amount <= 0:
        return 0.0
    return round(amount * rate, 2)
