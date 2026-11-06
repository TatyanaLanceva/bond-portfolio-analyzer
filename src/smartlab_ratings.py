"""Парсер кредитных рейтингов российских облигаций со smart-lab.ru.

Адаптирован из Colab-ноутбука дипломной работы.

Маппинг ширины прогресс-бара в рейтинг (по данным smart-lab.ru):
    100% → AAA
     95% → AA+
     90% → AA
     85% → AA-
     80% → A+
     75% → A
     70% → A-
     65% → BBB+
     60% → BBB
     55% → BBB-
"""
from __future__ import annotations

import csv
import json
import logging
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

RatingStatus = Literal["ok", "not_found", "error"]


@dataclass
class RatingResult:
    isin: str
    rating: str | None      # None если не нашли
    status: RatingStatus    # 'ok' | 'not_found' | 'error'
    reason: str = ""


USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
)

# Ширина прогресс-бара → рейтинг; BBB как fallback больше НЕ ИСПОЛЬЗУЕТСЯ.
# Если ширины нет или она вне Known range — status='not_found'.
RATING_BARS = {
    95: "AAA", 90: "AA+", 85: "AA", 80: "AA-",
    75: "A+",  70: "A",   65: "A-",
    60: "BBB+", 55: "BBB", 50: "BBB-",
    45: "BB+", 40: "BB",  35: "BB-",
}


def _pick_ua() -> str:
    return random.choice(USER_AGENTS)


def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=4, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "bond-portfolio-analyzer/1.0 (educational)"})
    return s


def load_cached_ratings() -> dict[str, RatingResult]:
    cache_path = Path("data/rating_cache.json")
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding='utf-8'))
            return {k: RatingResult(**v) for k, v in data.items()}
        except Exception:
            return {}
    return {}


def save_cached_ratings(cache: dict[str, RatingResult]) -> None:
    cache_path = Path("data/rating_cache.json")
    Path("data").mkdir(exist_ok=True)
    data = {k: asdict(v) for k, v in cache.items()}
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def _width_to_rating(width_pct: float) -> tuple[str | None, str]:
    """Возвращает (рейтинг, reason). None если рейтинг не найден."""
    if width_pct < 55:
        return None, "widthbelow55"
    # Ищем самый близкий Known ключ
    nearest = min(RATING_BARS, key=lambda k: abs(k - width_pct))
    if abs(nearest - width_pct) > 5:  # слишком далеко — рейтинг ненадежен
        return None, f"width={width_pct} вне диапазона"
    return RATING_BARS[nearest], ""


def get_bond_rating(isin: str, timeout: int = 10) -> RatingResult:
    url = f"https://smart-lab.ru/q/bonds/{isin}/"
    try:
        r = requests.get(
            url,
            headers={"User-Agent": _pick_ua(), "Accept-Language": "ru,en;q=0.9"},
            timeout=timeout,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        return RatingResult(isin, None, "error", repr(e))

    # Ищем ширину прогресс-бара
    m = re.search(r'linear-progress-bar__filed[^>]*style="[^"]*width:\s*(\d+)%", r.text)
    if not m:
        return RatingResult(isin, None, "not_found", "селектор не найден —, вероятно, изменилась вёрстка")

    width = int(m.group(1))
    rating, reason = _width_to_rating(width)
    if rating is None:
        return RatingResult(isin, None, "not_found", reason)
    return RatingResult(isin, rating, "ok", reason)


def enrich_ratings_bulk(isins: list[str], sleep_between: float = 0.5) -> dict[str, RatingResult]:
    out: dict[str, RatingResult] = {}
    for isin in isins:
        res = get_bond_rating(isin)
        out[isin] = res
        if res.status == "error":
            logger.warning("Рейтинг %s: error (%s)", isin, res.reason)
        elif res.status == "not_found":
            logger.info("Рейтинг %s: not_found (%s)", isin, res.reason)
        time.sleep(sleep_between)
    return out