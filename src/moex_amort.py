"""Проверка амортизации облигации через MOEX bondization.

Статусы:
    'amortizing'     — есть график амортизации;
    'not_amortizing' — график пуст (подтверждено);
    'unknown'        — сетевая ошибка / некорректный ответ.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

MOEX_BONDIZATION_URL = (
    "https://iss.moex.com/iss/statistics/engines/stock/markets/bonds/bondization/{isin}.json"
)
AmortizationStatus = Literal["amortizing", "not_amortizing", "unknown"]


@dataclass
class AmortizationInfo:
    isin: str
    status: AmortizationStatus
    schedule: list[tuple[str, float]]  # [(date_iso, amount_rub)]
    reason: str = ""


def _check_one(isin: str, timeout: int = 10) -> AmortizationInfo:
    try:
        r = requests.get(
            MOEX_BONDIZATION_URL.format(isin=isin),
            params={"iss.meta": "off"},
            timeout=timeout,
        )
        r.raise_for_status()
        block = r.json().get("amortizations", {})
        columns = block.get("columns") or []
        data = block.get("data") or []

        if not columns:
            return AmortizationInfo(isin, "unknown", [], "нет блока amortizations")
        if not data:
            return AmortizationInfo(isin, "not_amortizing", [])

        idx_date = columns.index("maturity_date")
        idx_val = columns.index("value_rub") if "value_rub" in columns else columns.index("value")
        schedule = [(row[idx_date], float(row[idx_val])) for row in data]
        return AmortizationInfo(isin, "amortizing", schedule)
    except (requests.RequestException, ValueError, KeyError) as e:
        # ВАЖНО: не False, а 'unknown' — иначе бумага молча попадёт в портфель
        return AmortizationInfo(isin, "unknown", [], repr(e))


def check_amortization_bulk(
    isins: list[str], max_workers: int = 8
) -> dict[str, AmortizationInfo]:
    result: dict[str, AmortizationInfo] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_check_one, i): i for i in isins}
        for fut in as_completed(futures):
            info = fut.result()
            result[info.isin] = info
            if info.status == "unknown":
                logger.warning("Амортизация %s: статус unknown (%s)", info.isin, info.reason)
    return result


def drop_amortizing_bonds(
    df,
    *,
    enabled: bool = False,
    amortizing_isins: set[str] | None = None,
    fail_on_missing_data: bool = True,
):
    """
    Исключает амортизируемые бумаги.
    При enabled=True и отсутствии данных о амортизации:
        fail_on_missing_data=True  → ValueError;
        False                      → warning + возврат без изменений.
    Бумаги со статусом 'unknown' по умолчанию исключаются как потенциально амортизируемые.
    """
    if not enabled:
        return df.copy()

    if amortizing_isins is None:
        if "IS_AMORTIZING" not in df.columns:
            msg = (
                "Фильтр амортизации включён, но в данных нет ни колонки IS_AMORTIZING, "
                "ни явного набора amortizing_isins. Отфильтровать невозможно."
            )
            if fail_on_missing_data:
                raise ValueError(msg)
            logger.warning(msg)
            return df.copy()
        mask = df["IS_AMORTIZING"].fillna(True).astype(bool)  # NaN → True (осторожно)
        return df[~mask].reset_index(drop=True)

    return df[~df["ISIN"].isin(amortizing_isins)].reset_index(drop=True)
