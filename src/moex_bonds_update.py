"""
Загрузка списка облигаций и котировок с MOEX ISS для файла bonds_current.csv.
Используются доски TQCB (корпоративные) и TQOB (ОФЗ), валюта RUB (FACEUNIT=SUR).
Рейтинг: TQOB → AAA (госдолг), а для TQCB — опционально парсится со smart-lab.ru
(вместо заглушки BBB).
"""
from __future__ import annotations

import json
import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

BASE_ISS = "https://iss.moex.com/iss/engines/stock/markets/bonds/boards"
DEFAULT_BOARDS: tuple[str, ...] = ("TQCB", "TQOB")


def _http_json(url: str, timeout: float = 120.0) -> dict:
    """HTTP GET → JSON-словарь. При ошибках HTTP/сети логирует и пробрасывает."""
    req = urllib.request.Request(url, headers={"User-Agent": "bond-portfolio-analyzer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            if status >= 400:
                raise urllib.error.HTTPError(
                    url, status, f"HTTP {status} от MOEX", resp.headers, None
                )
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        logger.error("MOEX HTTP %s: %s", e.code, url)
        raise
    except urllib.error.URLError as e:
        logger.error("MOEX сеть: %s — %s", url, e.reason)
        raise
    except json.JSONDecodeError as e:
        logger.error("MOEX ответ не JSON: %s — %s", url, e)
        raise


def _fetch_board(board_id: str, timeout: float = 120.0) -> pd.DataFrame:
    """Одна доска: справочник + доходности; склейка по SECID."""
    q = (
        f"{BASE_ISS}/{board_id}/securities.json?iss.meta=off"
        "&securities.columns=SECID,SHORTNAME,ISIN,MATDATE,FACEVALUE,FACEUNIT,COUPONPERCENT,ACCRUEDINT,STATUS"
        "&marketdata_yields.columns=SECID,EFFECTIVEYIELD,WAPRICE"
    )
    data = _http_json(q, timeout=timeout)
    sec = data.get("securities") or {}
    yld = data.get("marketdata_yields") or {}
    if not sec.get("data"):
        return pd.DataFrame()
    df_s = pd.DataFrame(sec["data"], columns=sec["columns"])
    df_y = pd.DataFrame(yld["data"], columns=yld["columns"]) if yld.get("data") else pd.DataFrame()
    if df_y.empty:
        return pd.DataFrame()
    df = df_s.merge(df_y, on="SECID", how="inner")
    df["BOARD"] = board_id
    return df


def fetch_moex_bonds_dataframe(
    boards: Iterable[str] = DEFAULT_BOARDS,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """
    Возвращает датафрейм в формате, готовом к сохранению как bonds_current.csv
    (до финальных фильтров ядра load_bonds_data).
    """
    parts: list[pd.DataFrame] = []
    for bid in boards:
        bid = str(bid).strip()
        if not bid:
            continue
        chunk = _fetch_board(bid, timeout=timeout)
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame()

    raw = pd.concat(parts, ignore_index=True)
    raw = raw.drop_duplicates(subset=["ISIN"], keep="first")

    # Только рубль и РФ
    raw = raw[raw["ISIN"].astype(str).str.startswith("RU", na=False)]
    raw = raw[raw["FACEUNIT"].astype(str) == "SUR"]
    # Торгуемые / активные выпуски
    raw = raw[raw["STATUS"].astype(str) == "A"]

    raw["MATDATE"] = pd.to_datetime(raw["MATDATE"], errors="coerce")
    raw = raw[raw["MATDATE"].notna()]
    today = pd.Timestamp.today().normalize()
    raw = raw[raw["MATDATE"] > today]

    raw["FACEVALUE"] = pd.to_numeric(raw["FACEVALUE"], errors="coerce")
    raw["COUPONPERCENT"] = pd.to_numeric(raw["COUPONPERCENT"], errors="coerce").fillna(0.0)
    raw["ACCRUEDINT"] = pd.to_numeric(raw["ACCRUEDINT"], errors="coerce").fillna(0.0)
    raw["WAPRICE"] = pd.to_numeric(raw["WAPRICE"], errors="coerce")
    raw["EFFECTIVEYIELD"] = pd.to_numeric(raw["EFFECTIVEYIELD"], errors="coerce")

    raw = raw[raw["WAPRICE"].notna() & raw["EFFECTIVEYIELD"].notna()]
    raw = raw[(raw["FACEVALUE"] > 0) & (raw["WAPRICE"] > 0)]

    # Грязная цена одной облигации, ₽: котировка % от номинала + НКД (как в выдаче MOEX)
    raw["DIRTY_PRICE_RUB"] = raw["FACEVALUE"] * raw["WAPRICE"] / 100.0 + raw["ACCRUEDINT"]
    raw["YTM_PCT"] = raw["EFFECTIVEYIELD"]
    raw["YEARS_TO_MATURITY"] = (raw["MATDATE"] - today).dt.days / 365.25
    raw = raw[raw["YEARS_TO_MATURITY"] > 0]

    raw["SHORTNAME"] = raw["SHORTNAME"].astype(str).str.strip()
    raw["RATING"] = raw["BOARD"].map(lambda b: "AAA" if b == "TQOB" else "BBB")

    out = raw[
        [
            "ISIN",
            "SHORTNAME",
            "RATING",
            "YEARS_TO_MATURITY",
            "DIRTY_PRICE_RUB",
            "YTM_PCT",
            "COUPONPERCENT",
            "FACEVALUE",
            "MATDATE",
        ]
    ].copy()
    out["MATDATE"] = out["MATDATE"].dt.strftime("%Y-%m-%d")
    out["YEARS_TO_MATURITY"] = out["YEARS_TO_MATURITY"].round(4)
    for c in ("DIRTY_PRICE_RUB", "YTM_PCT", "COUPONPERCENT", "FACEVALUE"):
        out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
    return out.sort_values("ISIN").reset_index(drop=True)


def _apply_smartlab_ratings(
    df: pd.DataFrame,
    cache_path: str | Path = "data/rating_cache.json",
    pause_sec: float = 2.0,
) -> pd.DataFrame:
    """
    Последовательное обогащение датафрейма рейтингами со smart-lab.ru.
    Использует только кэш — реальные запросы делаются только через
    отдельный скрипт tools/enrich_ratings.py.
    """
    try:
        from .smartlab_ratings import RatingCache
    except ImportError:
        from smartlab_ratings import RatingCache

    cache = RatingCache(cache_path)
    out = df.copy()
    
    corp_mask = out["RATING"] != "AAA"
    corp_idx = out.index[corp_mask].tolist()
    if not corp_idx:
        return out

    for idx in corp_idx:
        isin = str(out.at[idx, "ISIN"]).strip()
        cached = cache.get(isin)
        if cached is not None and cached[0] not in ("Нет данных", "Нет ISIN", "Ошибка"):
            out.at[idx, "RATING"] = cached[0]

    return out


def save_moex_bonds_csv(
    output_path: str | Path = "data/bonds_current.csv",
    boards: Iterable[str] = DEFAULT_BOARDS,
    backup: bool = True,
    timeout: float = 120.0,
    fetch_ratings: bool = False,
    ratings_pause: float = 2.0,
) -> Path:
    """
    Сохраняет актуальный снимок в CSV.

    Parameters
    ----------
    fetch_ratings : bool
        Если True, парсит реальные рейтинги со smart-lab.ru для корп. бумаг
        (иначе — заглушка BBB).
    ratings_pause : float
        Пауза между запросами к smart-lab.ru (сек). 0 — без паузы.
    """
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if backup and outp.is_file():
        bak = outp.with_suffix(".csv.bak")
        shutil.copy2(outp, bak)

    df = fetch_moex_bonds_dataframe(boards=boards, timeout=timeout)
    if df.empty:
        raise RuntimeError("MOEX вернул пустой набор строк — проверьте сеть или доски.")

    if fetch_ratings:
        df = _apply_smartlab_ratings(df, pause_sec=ratings_pause)

    df.to_csv(outp, index=False, encoding="utf-8-sig")
    return outp


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    p = save_moex_bonds_csv(fetch_ratings=False)
    print(f"Сохранено {p}, строк: ", len(pd.read_csv(p)))
