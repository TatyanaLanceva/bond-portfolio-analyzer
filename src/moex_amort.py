"""
Проверка графика амортизации по открытому API MOEX (bondization).
Считаем бумагу «амортизируемой» только если есть события с data_source == amortization
(не путать с единственной строкой maturity при финальном погашении).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Iterable


BONDIZATION_URL = (
    "https://iss.moex.com/iss/statistics/engines/stock/markets/bonds/bondization/{isin}.json"
    "?iss.meta=off"
)


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


def moex_amortizing_isins(
    isins: Iterable[str],
    pause_sec: float = 0.08,
    progress_callback=None,
) -> set[str]:
    """Запросы по одному ISIN; pause_sec снижает риск ограничений API."""
    out: set[str] = set()
    isins = list(dict.fromkeys(str(i).strip() for i in isins if str(i).strip().startswith("RU")))
    for n, isin in enumerate(isins, start=1):
        if moex_bond_has_amortization_schedule(isin):
            out.add(isin)
        if progress_callback:
            progress_callback(n, len(isins), isin)
        if pause_sec > 0 and n < len(isins):
            time.sleep(pause_sec)
    return out
