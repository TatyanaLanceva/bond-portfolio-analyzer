import json
import logging
import time

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dataclasses import dataclass

@dataclass
class FaceValueResult:
    facevalue: Optional[float]
    faceunit: Optional[str]
    source: str          # 'moex_iss' | 'moex_bondization' | 'missing' | 'non_rub'
    reason: str = ""

REQUEST_TIMEOUT = 30
MOEX_SECURITIES_URL = "https://iss.moex.com/iss/securities/{isin}.json"
MOEX_BONDIZATION_URL = "https://iss.moex.com/iss/statistics/engines/stock/markets/bonds/bondization/{isin}.json"
RUB_UNITS = {"SUR", "RUB"}
DEFAULT_CACHE = Path("data/facevalue_cache.json")
SLEEP_BETWEEN_REQUESTS = 1.0


def _save_cache(cache_data: dict, path: Path = DEFAULT_CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def _load_cache(path: Path = DEFAULT_CACHE) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Не удалось прочитать кэш номиналов %s: %s", path, e)
        return {}


def _row_from_block(block: dict) -> Optional[dict]:
    """Превращает блок {'columns': [...], 'data': [[...]]} в dict первой строки."""
    columns = block.get("columns") or []
    data = block.get("data") or []
    if not columns or not data:
        return None
    return dict(zip(columns, data[0]))


def fetch_facevalue_from_moex_securities(
    isin: str, timeout: int = REQUEST_TIMEOUT
) -> FaceValueResult:
    """Номинал на момент выпуска из MOEX ISS securities."""
    try:
        r = requests.get(
            MOEX_SECURITIES_URL.format(isin=isin),
            params={"iss.meta": "off"},
            timeout=timeout,
        )
        r.raise_for_status()
        row = _row_from_block(r.json().get("securities", {}))
        if row is None:
            return FaceValueResult(None, None, "missing", "securities: пустой ответ")

        fv = row.get("FACEVALUE")
        unit = (row.get("FACEUNIT") or "SUR").upper()
        if fv is None or not isinstance(fv, (int, float)) or fv <= 0:
            return FaceValueResult(None, None, "missing", f"FACEVALUE={fv!r}")
        if unit not in RUB_UNITS:
            return FaceValueResult(float(fv), unit, "non_rub", f"FACEUNIT={unit}")
        return FaceValueResult(float(fv), "SUR", "moex_iss")
    except (requests.RequestException, ValueError, KeyError) as e:
        return FaceValueResult(None, None, "missing", f"securities: {e!r}")


def fetch_amortization_schedule(
    isin: str, timeout: int = REQUEST_TIMEOUT
) -> list[tuple[str, float]]:
    """График амортизации: список (дата ISO, сумма RUB). Пустой список — амортизации нет."""
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
        if not columns or not data:
            return []
        idx_date = columns.index("maturity_date")
        idx_val = columns.index("value_rub") if "value_rub" in columns else columns.index("value")
        return [(row[idx_date], float(row[idx_val])) for row in data]
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("MOEX bondization: ошибка для %s: %s", isin, e)
        return []


def enrich_facevalue(
    df: pd.DataFrame,
    *,
    cache_path: Path = DEFAULT_CACHE,
    use_cache: bool = True,
    allow_non_rub: bool = False,
) -> pd.DataFrame:
    """
    Обогащает df колонками FACEVALUE, FACEUNIT, FACEVALUE_SOURCE.
    Строки со статусом 'missing' и (при allow_non_rub=False) 'non_rub' исключаются
    с предупреждением — никакого fallback на 1000.0.
    """
    if "ISIN" not in df.columns:
        raise ValueError("В DataFrame отсутствует колонка ISIN")

    cache = _load_cache(cache_path) if use_cache else {}
    cache_dirty = False

    facevalue = []
    faceunit = []
    source = []

    for i, isin in enumerate(df["ISIN"].astype(str)):
        isin = isin.strip().upper()

        # 1) уже есть валидный номинал в CSV?
        raw_fv = df.iloc[i].get("FACEVALUE")
        raw_unit = df.iloc[i].get("FACEUNIT", "SUR")
        unit = str(raw_unit).upper() if pd.notna(raw_unit) else "SUR"
        if pd.notna(raw_fv):
            try:
                fv = float(raw_fv)
            except (TypeError, ValueError):
                fv = None
            if fv is not None and fv > 0:
                facevalue.append(fv)
                faceunit.append(unit)
                if unit in RUB_UNITS:
                    source.append("csv")
                else:
                    source.append("csv" if allow_non_rub else "non_rub")
                continue

        # 2) кэш
        cached = cache.get(isin)
        if cached and cached.get("source") in {"moex_iss", "moex_bondization"}:
            facevalue.append(cached["facevalue"])
            faceunit.append(cached.get("faceunit", "SUR"))
            source.append(cached["source"])
            continue

        # 3) MOEX
        res = fetch_facevalue_from_moex_securities(isin)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        # 3a) MOEX securities вернул missing → пробуем bondization (остаточный номинал)
        if res.source == "missing":
            sched = fetch_amortization_schedule(isin)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            if sched:
                # остаточный = initial - сумма амортизаций
                # initial берём из первого известного значения MOEX, если оно есть;
                # если нет — использовать нельзя, оставляем missing
                pass

        if res.source == "non_rub" and not allow_non_rub:
            facevalue.append(res.facevalue)
            faceunit.append(res.faceunit)
            source.append("non_rub")
            continue

        if res.source in {"moex_iss"} or (res.source == "non_rub" and allow_non_rub):
            facevalue.append(res.facevalue)
            faceunit.append(res.faceunit)
            source.append(res.source)
            if use_cache:
                cache[isin] = {
                    "facevalue": res.facevalue,
                    "faceunit": res.faceunit,
                    "source": res.source,
                }
                cache_dirty = True
            continue

        # 4) всё плохо
        logger.warning("Номинал не найден для %s: %s", isin, res.reason)
        facevalue.append(None)
        faceunit.append(None)
        source.append("missing")

    out = df.copy()
    out["FACEVALUE"] = facevalue
    out["FACEUNIT"] = faceunit
    out["FACEVALUE_SOURCE"] = source

    if cache_dirty:
        _save_cache(cache, cache_path)

    bad = out[out["FACEVALUE_SOURCE"].isin(["missing", "non_rub"])]
    if not bad.empty:
        logger.warning(
            "Исключено %d бумаг без валидного RUB-номинала: %s",
            len(bad), ", ".join(bad["ISIN"].astype(str).head(10)),
        )
    return out[~out["FACEVALUE_SOURCE"].isin(["missing", "non_rub"])].reset_index(drop=True)
def get_face_values(df, cache_path=DEFAULT_CACHE, use_cache=True, allow_non_rub=False):
    return enrich_facevalue(df, cache_path=cache_path, use_cache=use_cache, allow_non_rub=allow_non_rub)

def calculate_dirty_price(facevalue: float, price_pct: float, accrued_interest: float = 0.0) -> float:
    from cashflow import calculate_dirty_price as _cf_dirty_price
    return _cf_dirty_price(facevalue, price_pct, accrued_interest)
