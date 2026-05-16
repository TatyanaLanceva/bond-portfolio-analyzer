"""
Проверка графика амортизации по открытому API MOEX (bondization).
Считаем бумагу «амортизируемой» только если есть события с data_source == amortization
(не путать с единственной строкой maturity при финальном погашении).

Оптимизация: результаты кэшируются в data/amortizing_cache.json (хеш от списка ISIN),
повторные запуски без изменения набора ISIN — мгновенные.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


BONDIZATION_URL = (
    "https://iss.moex.com/iss/statistics/engines/stock/markets/bonds/bondization/{isin}.json"
    "?iss.meta=off"
)
CACHE_PATH = Path("data/amortizing_cache.json")
CACHE_TTL_HOURS = 24


def moex_bond_has_amortization_schedule(isin: str, timeout: float = 20.0) -> bool:
    """True, если у выпуска есть плановая амортизация номинала (не только финальное погашение)."""
    isin = str(isin).strip()
    if not isin.startswith("RU"):
        return False
    url = BONDIZATION_URL.format(isin=isin)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False

    block = payload.get("amortizations") or {}
    columns = block.get("columns") or []
    rows = block.get("data") or []
    if not rows:
        return False

    try:
        dsi = columns.index("data_source")
    except ValueError:
        return len(rows) > 1

    for row in rows:
        if len(row) > dsi and row[dsi] == "amortization":
            return True
    return False


def _cache_key(isins: list[str]) -> str:
    """Хеш от отсортированного списка ISIN для инвалидации кэша."""
    raw = ",".join(sorted(isins))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_cache() -> dict:
    """Загружает кэш из файла."""
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """Сохраняет кэш в файл."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def moex_amortizing_isins(
    isins: Iterable[str],
    pause_sec: float = 0.0,
    progress_callback=None,
) -> set[str]:
    """
    Запросы по одному ISIN. Результаты кэшируются в файл.

    Если хеш набора ISIN не изменился и кэш свежий (< 24 ч) — возвращает
    сохранённый результат мгновенно, без единого HTTP-запроса.
    """
    isins = list(dict.fromkeys(str(i).strip() for i in isins if str(i).strip().startswith("RU")))
    if not isins:
        return set()

    ckey = _cache_key(isins)
    cache = _load_cache()

    # Проверяем: есть ли в кэше запись с таким же хешом и не старше 24 часов
    cached_entry = cache.get(ckey)
    if cached_entry is not None:
        ts = cached_entry.get("timestamp", 0)
        if time.time() - ts < CACHE_TTL_HOURS * 3600:
            return set(cached_entry.get("amortizing_isins", []))

    # Нет кэша или устарел — делаем реальные запросы
    out: set[str] = set()
    total = len(isins)
    for n, isin in enumerate(isins, start=1):
        if moex_bond_has_amortization_schedule(isin):
            out.add(isin)
        if progress_callback:
            progress_callback(n, total, isin)
        if pause_sec > 0 and n < total:
            time.sleep(pause_sec)

    # Сохраняем в кэш
    cache[ckey] = {
        "timestamp": time.time(),
        "amortizing_isins": list(out),
    }
    _save_cache(cache)

    return out