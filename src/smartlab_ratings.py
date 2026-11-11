
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/rating_cache.json")
CACHE_TTL = 24 * 3600

@dataclass
class RatingInfo:
    """диная структура одного кредитного рейтинга.

    оля:
        isin: ISIN бумаги.
        rating: Текст рейтинга (например `""BBB-""`).
        agency: гентство, выдавшее рейтинг.
        scale: ациональная или международная шкала.
        outlook: рогноз или `None`.
        rating_date: ата выставления рейтинга в формате YYYY-MM-DD.
        status: дин из `confirmed` / `not_found` / `unavailable` / `invalid`.
    """

    isin: str
    rating: str | None = None
    agency: str | None = None
    scale: str | None = None
    outlook: str | None = None
    rating_date: str | None = None
    status: str = "confirmed"


def normalize_rating(
    rating: str | None,
) -> str | None:
    """ормализовать обозначение рейтинга.

    Args:
        rating: сходный рейтинг (строка или `None`).

    Returns:
        str | None: ормализованный рейтинг в верхнем регистре без пробелов,
            или `None`, если входной параметр пуст/невалиден.
    """
    if not rating:
        return None

    value = rating.strip().upper()
    value = re.sub(r"[^A-Z0-9+\\-]", "", value)
    return value or None

def _load_cache(cache_path: Path = CACHE_PATH) -> Dict:
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        current_time = time.time()
        filtered = {}
        for isin, rating in data.items():
            if isinstance(rating, dict):
                expiry = rating.get("expiry", 0)
                if current_time - expiry < CACHE_TTL:
                    filtered[isin] = rating.get("rating", "")
            elif isinstance(rating, str):
                filtered[isin] = rating
        return filtered
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache_data: Dict, cache_path: Path = CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    current_time = time.time()
    formatted = {}
    for isin, rating in cache_data.items():
        if isinstance(rating, str):
            formatted[isin] = {"rating": rating, "expiry": current_time}
        else:
            formatted[isin] = rating
    with open(cache_path, "w", encoding="utf-8-sig") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)


def _get_rating_from_smartlab(isin: str) -> Optional[str]:
    url = "https://smart-lab.ru/q/bonds/{isin}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")
        rating_text = None
        for string in soup.stripped_strings:
            if "рейтинг" in string.lower():
                rating_text = string
                break
        if rating_text is None:
            rating_divs = soup.find_all("div", class_=lambda x: x and "rating" in str(x).lower())
            if rating_divs:
                rating_text = rating_divs[0].get_text(strip=True)
        if rating_text is None:
            return None
        match = re.search(r"[A-Z][A-Za-z\+\-]*/?[0-9]?", rating_text)
        if match:
            rating = match.group(0).strip()
            rating = re.sub(r"[^A-Za-z\+\-]", "", rating)
            if rating and len(rating) >= 1:
                return rating
        return None
    except Exception as e:
        logger.exception("Ошибка парсинга для %s: %s", isin, e)
        return None


def get_ratings_bulk(
    isins: List[str],
    cache_path: Path = CACHE_PATH,
    force_refresh: bool = False,
) -> Dict[str, Optional[str]]:
    cache = _load_cache(cache_path)
    results: Dict[str, Optional[str]] = {}
    to_parse = []
    for isin in isins:
        isin_clean = isin.strip().upper()
        if force_refresh:
            to_parse.append(isin_clean)
            continue
        if isin_clean in cache:
            cached_rating = cache[isin_clean]
            if isinstance(cached_rating, dict):
                rating = cached_rating.get("rating", "")
            else:
                rating = cached_rating
            if rating:
                results[isin_clean] = rating
            else:
                results[isin_clean] = None
                to_parse.append(isin_clean)
            continue
        to_parse.append(isin_clean)
    for isin in to_parse:
        time.sleep(2.0)
        rating = _get_rating_from_smartlab(isin)
        results[isin] = rating
        if rating:
            cache[isin] = rating
            _save_cache(cache)
    return results


def rating_to_score(rating: Optional[str]) -> int:
    RATING_MAP = {
        "AAA": 0,
        "AA+": 1,
        "AA": 2,
        "AA-": 3,
        "A+": 4,
        "A": 5,
        "A-": 6,
        "BBB+": 7,
        "BBB": 8,
        "BBB-": 9,
        "BB+": 10,
        "BB": 11,
        "BB-": 12,
        "B+": 13,
        "B": 14,
        "B-": 15,
        "CCC": 16,
    }
    if rating is None or rating == "":
        return 99
    return RATING_MAP.get(rating, 99)
