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
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from urllib3.util.retry import Retry

try:
    from src.moex_client import create_http_session, get_json
except ImportError:  # тесты добавляют src/ в sys.path напрямую
    from moex_client import create_http_session, get_json

logger = logging.getLogger(__name__)

BASE_ISS = "https://iss.moex.com/iss/engines/stock/markets/bonds/boards"
DEFAULT_BOARDS: tuple[str, ...] = ("TQCB", "TQOB")

# Ротация User-Agent для избежания блокировок
USER_AGENTS = [
    "bond-portfolio-analyzer/1.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
]


def fetch_iss_block(
    session: requests.Session,
    url: str,
    block_name: str,
    params: dict[str, Any] | None = None,
    page_size: int = 100,
    max_pages: int = 100,
) -> pd.DataFrame:
    """Получить все страницы заданного блока MOEX ISS одним объединённым DataFrame.

    Args:
        session: HTTP-сессия.
        url: Адрес ISS-эндпоинта.
        block_name: Ключ блока в JSON-ответе (например ``securities``).
        params: Базовые параметры запроса. ``iss.meta=off`` добавляется автоматически.
        page_size: Число записей на одну страницу.
        max_pages: Максимальное число запросов по пагинации.

    Returns:
        pd.DataFrame: Все строки блока, пронумерованные заново.

    Raises:
        ValueError: при отсутствии блока, пустых колонок/строк или исчерпании лимита.
    """
    base_params = dict(params or {})
    base_params["iss.meta"] = "off"

    frames: list[pd.DataFrame] = []

    for page in range(max_pages):
        request_params = {
            **base_params,
            "start": page * page_size,
        }

        response = get_json(
            session=session,
            url=url,
            params=request_params,
        )

        block = response.get(block_name)

        if not isinstance(block, dict):
            raise ValueError(
                f"В ответе отсутствует блок {block_name!r} "
                f"(найденный тип: {type(block).__name__})"
            )

        columns = block.get("columns", [])
        rows = block.get("data", [])

        if not columns or not rows:
            # Пустой блок значит, что выгрузка закончилась
            break

        frame = pd.DataFrame(rows, columns=columns)
        frames.append(frame)

        if len(rows) < page_size:
            # Последняя страница
            break

    else:
        raise RuntimeError(
            f"Превышен лимит пагинации: {max_pages} страниц для блока {block_name!r}"
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _ensure_session() -> requests.Session:
    """Вернуть общую HTTP-сессию, созданную один раз для всего клиента."""
    return create_http_session()


def _fetch_board(
    session: requests.Session,
    board_id: str,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Одна доска: справочник + доходности; склейка по SECID (left join).

    Получает блоки ``securities`` и ``marketdata_yields`` независимо с пагинацией,
    чтобы выпуски без рыночных данных тоже сохранялись.
    """
    securities_url = (
        f"{BASE_ISS}/{board_id}/securities.json"
    )
    marketdata_url = (
        f"{BASE_ISS}/{board_id}/marketdata.json"
    )

    # Параметры справочника
    sec_params: dict[str, Any] = {
        "iss.meta": "off",
        "securities.columns": "SECID,SHORTNAME,ISIN,MATDATE,FACEVALUE,FACEUNIT,"
        "COUPONPERCENT,ACCRUEDINT,STATUS",
    }

    # Параметры рыночных данных
    mkt_params: dict[str, Any] = {
        "iss.meta": "off",
        "marketdata.columns": "SECID,EFFECTIVEYIELD,WAPRICE",
    }

    sec_df = fetch_iss_block(
        session=session,
        url=securities_url,
        block_name="securities",
        params=sec_params,
        page_size=100,
        max_pages=20,
    )

    if sec_df.empty:
        return pd.DataFrame()

    mkt_df = fetch_iss_block(
        session=session,
        url=marketdata_url,
        block_name="marketdata",
        params=mkt_params,
        page_size=100,
        max_pages=20,
    )

    if not mkt_df.empty:
        # Left join: не теряем бумаги без доходности
        sec_df = sec_df.merge(
            mkt_df,
            on="SECID",
            how="left",
            suffixes=("", "_mkt"),
        )

    sec_df["BOARD"] = board_id
    return sec_df


def fetch_moex_bonds_dataframe(
    boards: Iterable[str] = DEFAULT_BOARDS,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """
    Возвращает датафрейм в формате, готовом к сохранению как bonds_current.csv
    (до финальных фильтров ядра load_bonds_data).
    """
    parts: list[pd.DataFrame] = []
    session = _ensure_session()

    for bid in boards:
        bid = str(bid).strip()
        if not bid:
            continue
        chunk = _fetch_board(session=session, board_id=bid, timeout=timeout)
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
    raw["RATING"] = raw["BOARD"].map(lambda b: "AAA" if b == "TQOB" else "N/A")

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


def _apply_ratings(
    df: pd.DataFrame,
    cache_path: str | Path = "data/rating_cache.json",
    fetch: bool = False,
    pause_sec: float = 2.0,
) -> pd.DataFrame:
    """
    Реальные рейтинги вместо заглушки. ОФЗ = AAA по определению.
    """
    try:
        from src.smartlab_ratings import get_ratings_bulk, rating_to_score
    except ImportError:
        from smartlab_ratings import get_ratings_bulk, rating_to_score

    df = df.copy()

    if "BOARD" in df.columns:
        ofz_mask = df["BOARD"] == "TQOB"
        corp_mask = df["BOARD"] == "TQCB"
    else:
        isin_col = df["ISIN"].astype(str).str.strip()
        ofz_mask = isin_col.str.startswith("RU000A")
        corp_mask = ~ofz_mask

    df.loc[ofz_mask, "RATING"] = "AAA"

    if fetch and corp_mask.any():
        isins = df.loc[corp_mask, "ISIN"].dropna().unique().tolist()
        ratings = get_ratings_bulk(isins, force_refresh=False)
        df.loc[corp_mask, "RATING"] = df.loc[corp_mask, "ISIN"].map(ratings).fillna("N/A")

    df["RATING_SCORE"] = df["RATING"].map(rating_to_score)
    return df


def _apply_smartlab_ratings(
    df: pd.DataFrame,
    cache_path: str | Path = "data/rating_cache.json",
    pause_sec: float = 2.0,
) -> pd.DataFrame:
    """Пополнить реальными кредитными рейтингами корпоративные облигации.

    ОФЗ (доска TQOB) считается AAA по определению. Для корпоративных бумаг (TQCB)
    запрашивает рейтинг со smart-lab.ru; результат кэшируется в JSON.

    Args:
        df: Датафрейм с колонками BOARD и ISIN.
        cache_path: Путь к файлу кэша рейтингов.
        pause_sec: Пауза между запросами к smart-lab.ru (секунды).

    Returns:
        pd.DataFrame: Дублированный датафрейм с заполненными RATING и RATING_SCORE.
    """
    try:
        from src.smartlab_ratings import get_ratings_bulk, rating_to_score
    except ImportError:
        from smartlab_ratings import get_ratings_bulk, rating_to_score

    df = df.copy()

    if "BOARD" in df.columns:
        ofz_mask = df["BOARD"] == "TQOB"
        corp_mask = df["BOARD"] == "TQCB"
    else:
        isin_col = df["ISIN"].astype(str).str.strip()
        ofz_mask = isin_col.str.startswith("RU000A")
        corp_mask = ~ofz_mask

    df.loc[ofz_mask, "RATING"] = "AAA"

    if corp_mask.any():
        isins = df.loc[corp_mask, "ISIN"].dropna().unique().tolist()
        ratings = get_ratings_bulk(isins, force_refresh=False)
        df.loc[corp_mask, "RATING"] = (
            df.loc[corp_mask, "ISIN"].map(ratings).fillna("N/A")
        )

    df["RATING_SCORE"] = df["RATING"].map(rating_to_score)
    return df


def enrich_ratings(
    df: pd.DataFrame,
    cache_path: str | Path = "data/rating_cache.json",
    pause_sec: float = 2.0,
) -> pd.DataFrame:
    """Публичный обогащатель кредитных рейтингов без требования колонки BOARD.

    Тип бумаги определяется по ISIN: RU000A* — ОФЗ (РЦБ, рейтинг AAA),
    остальные — корпоративные (рейтинги запрашиваются из Smart-Lab кэша,
    отсутствующий рейтинг помечается как 'N/A', а не подставляется как BBB).

    Args:
        df: DataFrame с колонками ISIN и (опционально) BOARD, RATING.
        cache_path: Путь к JSON-кэшу рейтингов Smart-Lab.
        pause_sec: Пауза между запросами к Smart-Lab (сек).

    Returns:
        pd.DataFrame: копия df с заполненными колонками RATING и RATING_SCORE.
    """
    try:
        from src.smartlab_ratings import get_ratings_bulk, rating_to_score
    except ImportError:
        from smartlab_ratings import get_ratings_bulk, rating_to_score

    df = df.copy()

    if "BOARD" in df.columns:
        ofz_mask = df["BOARD"] == "TQOB"
        corp_mask = df["BOARD"] == "TQCB"
    else:
        isin_col = df["ISIN"].astype(str).str.strip()
        ofz_mask = isin_col.str.startswith("RU000A")
        corp_mask = ~ofz_mask

    df.loc[ofz_mask, "RATING"] = "AAA"

    if corp_mask.any():
        isins = df.loc[corp_mask, "ISIN"].dropna().unique().tolist()
        ratings = get_ratings_bulk(isins, force_refresh=False)
        df.loc[corp_mask, "RATING"] = (
            df.loc[corp_mask, "ISIN"].map(ratings).fillna("N/A")
        )

    df["RATING_SCORE"] = df["RATING"].map(rating_to_score)
    return df


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
        df = _apply_ratings(df, pause_sec=ratings_pause)

    df.to_csv(outp, index=False, encoding="utf-8-sig")
    return outp


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    p = save_moex_bonds_csv(fetch_ratings=False)
    print(f"Сохранено {p}, строк: ", len(pd.read_csv(p)))
