"""
Парсер кредитных рейтингов российских облигаций со smart-lab.ru.
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
   <55% → BB+ и ниже (спекулятивные)
"""
from __future__ import annotations

import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# --- Маппинг ширины прогресс-бара → рейтинг ---
WIDTH_TO_RATING: tuple[tuple[int, str], ...] = (
    (100, "AAA"),
    (95, "AA+"),
    (90, "AA"),
    (85, "AA-"),
    (80, "A+"),
    (75, "A"),
    (70, "A-"),
    (65, "BBB+"),
    (60, "BBB"),
    (55, "BBB-"),
)


def _width_to_rating(width_pct: float) -> str:
    """Конвертирует ширину прогресс-бара в наилучший рейтинг."""
    if width_pct < 55:
        return "BB+"  # Спекулятивная зона
    for w, r in WIDTH_TO_RATING:
        if abs(width_pct - w) < 2.5:
            return r
    # Интерполяция между порогами
    for w, r in WIDTH_TO_RATING:
        if width_pct >= w:
            return r
    return "BBB-"


# --- Карта HTML-классов, которые могут быть прогресс-баром (версии smart-lab) ---
PROGRESS_BAR_CLASSES = (
    "linear-progress-bar",
    "linear-progress",
    "progress-bar",
    "progress",
    "rating-progress",
    "bonds-rating-progress",
    "new-progress",
    "ProgressBar_root",
    "progress__bar",
)

# --- Пользовательские агенты для ротации ---
USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.5 Safari/605.1.15",
)


# ---------------------------------------------------------------------------
#  Основная функция парсинга одной бумаги
# ---------------------------------------------------------------------------
def get_bond_rating(
    isin: str,
    timeout: float = 15.0,
    session: requests.Session | None = None,
    max_retries: int = 3,
) -> tuple[str, str]:
    """
    Получает рейтинг облигации со smart-lab.ru по её ISIN.
    Использует requests + html.parser (как в проверенном рабочем примере).
    """
    import time as _time
    url = f"https://smart-lab.ru/q/bonds/{isin}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    sess = session or requests.Session()

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = sess.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            # Успех
            break
        except Exception as e:
            last_error = str(e)
            status_code = getattr(e, 'response', None) and e.response.status_code or 0
            if attempt < max_retries and status_code in (502, 503, 504, 0):
                _time.sleep(2 ** attempt)
                continue
            logger.warning("ISIN %s — ошибка: %s", isin, last_error)
            return "Нет данных", f"Ошибка: {last_error}"

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Шаг 1: ищем строку "рейтинг"
    rating_text = soup.find(string=lambda text: text and 'рейтинг' in text.lower())
    if not rating_text:
        return "Нет данных", "Блок рейтинга не найден"

    # Шаг 2: поднимаемся к родительскому div
    rating_parent = rating_text.find_parent('div')
    if not rating_parent:
        return "Нет данных", "Родительский div не найден"

    # Шаг 3: ищем прогресс-бар внутри родителя или рядом
    progress_bar = rating_parent.find('div', class_='linear-progress-bar')
    if not progress_bar:
        progress_bar = rating_parent.find_next('div', class_='linear-progress-bar')
    if not progress_bar:
        return "Нет данных", "Прогресс-бар не найден"

    # Шаг 4: ищем filled (класс linear-progress-bar__filed — с опечаткой, как на smart-lab)
    rating_filled = progress_bar.find('div', class_='linear-progress-bar__filed')
    if not rating_filled:
        return "Нет данных", "Filled не найден"

    # Шаг 5: извлекаем рейтинг и цвет
    color = rating_filled.get('style', '')
    rating_value = rating_filled.get_text(strip=True)
    
    if not rating_value:
        return "Нет данных", "Рейтинг пуст"
    
    return rating_value, color


# ---------------------------------------------------------------------------
#  Кэширование результатов (чтобы избежать повторных запросов)
# ---------------------------------------------------------------------------
class RatingCache:
    """Простой JSON-кэш для результатов парсинга рейтингов."""

    def __init__(self, path: str | Path = "data/rating_cache.json"):
        self.path = Path(path)
        self._data: dict[str, list[str]] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, isin: str) -> tuple[str, str] | None:
        entry = self._data.get(isin)
        if entry and len(entry) >= 2:
            return entry[0], entry[1]
        return None

    def set(self, isin: str, rating: str, color: str):
        self._data[isin] = [rating, color]

    def __contains__(self, isin: str) -> bool:
        return isin in self._data


# ---------------------------------------------------------------------------
#  Batch-обработка CSV
# ---------------------------------------------------------------------------
def enrich_csv_with_ratings(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    *,
    pause_sec: float = 3.0,
    use_cache: bool = True,
    cache_path: str | Path = "data/rating_cache.json",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[int, int]:
    """
    Читает CSV (разделитель `;`), добавляет колонки Рейтинг и Цвет рейтинга,
    сохраняет результат.

    Parameters
    ----------
    input_csv : путь к исходному CSV.
    output_csv : путь для сохранения (по умолчанию input_csv с суффиксом _with_ratings).
    pause_sec : пауза между запросами (сек).
    use_cache : использовать/обновлять JSON-кэш.
    cache_path : путь к файлу кэша.
    progress_callback : (номер, всего, isin).

    Returns
    -------
    (обработано, всего) — количество успешно обработанных и общее количество.
    """
    inp = Path(input_csv)
    if not inp.exists():
        raise FileNotFoundError(f"Файл {inp} не найден")

    if output_csv is None:
        stem = inp.stem
        output_csv = inp.with_name(f"{stem}_with_ratings.csv")

    cache = RatingCache(cache_path) if use_cache else None
    session = requests.Session()

    with open(inp, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    total = len(rows)
    logger.info("Начинаем обработку %d записей из %s", total, inp)

    # Добавляем колонки
    extra_cols = ["Рейтинг", "Цвет рейтинга"]
    for ec in extra_cols:
        if ec not in fieldnames:
            fieldnames.append(ec)

    processed = 0
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            isin = (row.get("ISIN") or "").strip()

            if not isin.startswith("RU"):
                row["Рейтинг"] = "Нет ISIN"
                row["Цвет рейтинга"] = "Нет ISIN"
                writer.writerow(row)
                continue

            # Проверяем кэш
            if cache and isin in cache:
                r, c = cache.get(isin)
                row["Рейтинг"] = r
                row["Цвет рейтинга"] = c
                if r not in ("Нет данных", "Нет ISIN", "Ошибка"):
                    processed += 1
                writer.writerow(row)
                if progress_callback:
                    progress_callback(i, total, isin)
                continue

            secid = row.get("SECID", row.get("SHORTNAME", isin))
            msg = f"[{i}/{total}] {secid} ({isin})…"
            logger.info(msg)

            rating, color = get_bond_rating(isin, session=session)
            row["Рейтинг"] = rating
            row["Цвет рейтинга"] = color

            if rating not in ("Нет данных", "Нет ISIN", "Ошибка"):
                processed += 1

            if cache:
                cache.set(isin, rating, color)

            writer.writerow(row)

            if progress_callback:
                progress_callback(i, total, isin)

            if pause_sec > 0 and i < total:
                time.sleep(pause_sec)

        if cache:
            cache.save()

    session.close()
    logger.info("Готово. Обработано %d / %d, сохранено в %s", processed, total, output_csv)
    return processed, total


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="Парсер кредитных рейтингов облигаций со smart-lab.ru"
    )
    parser.add_argument("input", nargs="?", default="data/bonds_filter.csv",
                        help="Входной CSV-файл (разделитель ;)")
    parser.add_argument("-o", "--output", default=None,
                        help="Выходной CSV (по умолчанию input_with_ratings.csv)")
    parser.add_argument("--pause", type=float, default=3.0,
                        help="Пауза между запросами, сек (по умолчанию 3)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Не использовать кэш")
    args = parser.parse_args()

    processed, total = enrich_csv_with_ratings(
        args.input,
        args.output,
        pause_sec=args.pause,
        use_cache=not args.no_cache,
    )
    print(f"\nГотово! {processed}/{total} бумаг с рейтингом.")