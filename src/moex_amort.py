"""Проверка амортизации облигации через MOEX bondization.

Статусы:
    'amortizing'     — есть график амортизации;
    'not_amortizing' — график пуст (подтверждено);
    'unknown'        — сетевая ошибка / некорректный ответ.

Кэш хранит итоговые статусы с временной меткой, чтобы повторные вызовы не
нагружали API. Ключ кэша — ISIN; значение — {'status': str, 'at': float}.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Any

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

MOEX_BONDIZATION_URL = (
    "https://iss.moex.com/iss/statistics/engines/stock/markets/bonds/bondization/{isin}.json"
)
AmortizationStatus = Literal["amortizing", "not_amortizing", "unknown"]

# Путь к файлу кэша со статусами амортизации (папка data/ по умолчанию)
DEFAULT_CACHE_PATH = Path("data/amortizing_cache.json")


def _load_amort_cache(cache_path: Path = DEFAULT_CACHE_PATH) -> dict[str, dict]:
    """Загрузить кэш статусов амортизации из JSON-файла.

    Returns:
        dict: {isin: {'status': str, 'at': float}}.
    """
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return {k: {"status": v.get("status", "unknown"), "at": v.get("at", 0)}
                for k, v in data.items()}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Не удалось прочитать кэш амортизации %s: %s", cache_path, e)
        return {}


def _save_amort_cache(
    cache_data: dict[str, dict],
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> None:
    """Сохранить кэш статусов амортизации в JSON-файл.

    Args:
        cache_data: {isin: {'status': str, 'at': float}}.
        cache_path: Путь для записи.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8-sig") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def _is_cache_fresh(cache_entry: dict, ttl_seconds: int = 24 * 3600) -> bool:
    """Проверить, что запись кэша не устарела (TTL в секундах)."""
    try:
        return (time.time() - cache_entry.get("at", 0)) < ttl_seconds
    except (TypeError, ValueError):
        return False


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


def moex_amortizing_isins(
    isins: list[str],
    cache_ttl_seconds: int = 24 * 3600,
    max_workers: int = 8,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> set[str]:
    """Вернуть множество ISIN с подтверждённым графиком амортизации.

    Логика:
        * результаты, взятые из кэша, считаются верными при наличии и свежести;
        * ISIN со статусом ``unknown`` запрашиваются заново (не попадают в ответ);
        * результаты запрашиваемых заново периода размечаются через пул;
        * итоговый словарь кэшируется обратно на диск.

    Args:
        isins: Список ISIN для проверки.
        cache_ttl_seconds: Время жизни записи кэша в секундах (по умолчанию 24 ч).
        max_workers: Число потоков в пуле.
        cache_path: Путь к JSON-файлу кэша.

    Returns:
        set[str]: ISIN с подтверждённым амортизируемым графиком.
    """
    cache = _load_amort_cache(cache_path)
    to_check: list[str] = []
    amortizing_set: set[str] = set()

    for isin in isins:
        entry = cache.get(isin)
        if entry and _is_cache_fresh(entry, cache_ttl_seconds):
            if entry.get("status") == "amortizing":
                amortizing_set.add(isin)
            continue
        to_check.append(isin)

    if to_check:
        bulk = check_amortization_bulk(to_check, max_workers=max_workers)
        cache.update({i: {"status": info.status, "at": time.time()}
                      for i, info in bulk.items()})
        if cache_path != DEFAULT_CACHE_PATH:
            _save_amort_cache(cache, cache_path)
        amortizing_set.update(i for i, info in bulk.items() if info.status == "amortizing")

    if cache_path != DEFAULT_CACHE_PATH:
        _save_amort_cache(cache, cache_path)

    return amortizing_set


def drop_amortizing_bonds(
    df: pd.DataFrame,
    *,
    enabled: bool = False,
    amortizing_isins: set[str] | None = None,
    fail_on_missing_data: bool = True,
) -> pd.DataFrame:
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
