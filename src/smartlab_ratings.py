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
    timeout: float = 20.0,
    session: requests.Session | None = None,
) -> tuple[str, str]:
    """
    Получает рейтинг облигации со smart-lab.ru по её ISIN.

    Returns
    -------
    (rating: str, color_or_width: str)
        rating — текстовый код рейтинга (AAA, AA+, …) или "Нет данных".
        color_or_width — сырое значение ширины прогресс-бара (например "width: 90%;")
                         или сообщение об ошибке.
    """
    import random
    url = f"https://smart-lab.ru/q/bonds/{isin}/"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    sess = session or requests.Session()

    try:
        resp = sess.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("HTTP-ошибка для ISIN %s: %s", isin, e)
        return "Нет данных", f"Ошибка HTTP: {e}"

    soup = BeautifulSoup(resp.text, "lxml")

    # --- Шаг 1: ищем блок/секцию, содержащую слово "рейтинг" ---
    rating_section = soup.find(
        lambda tag: (
            tag.name in ("div", "section", "span", "td", "th")
            and tag.get("class")
            and any("rating" in (c or "").lower() for c in tag.get("class", []))
        )
    ) or soup.find(string=lambda t: t and "рейтинг" in t.lower())

    if rating_section is None:
        return "Нет данных", "Блок рейтинга не найден"

    # Если нашли текстовый узел — поднимаемся к родительскому элементу
    if isinstance(rating_section, str):
        parent = soup.find_all(string=re.compile("рейтинг", re.IGNORECASE))
        if not parent:
            return "Нет данных", "Нет строки с 'рейтинг'"
        rating_section = parent[0].parent if parent[0].parent else soup

    # --- Шаг 2: ищем прогресс-бар внутри блока ---
    progress_bar = None
    for cls in PROGRESS_BAR_CLASSES:
        progress_bar = rating_section.find("div", class_=cls)
        if progress_bar:
            break

    # Если не нашли внутри — ищем по всей странице
    if progress_bar is None:
        for cls in PROGRESS_BAR_CLASSES:
            progress_bar = soup.find("div", class_=cls)
            if progress_bar:
                break

    if progress_bar is None:
        return "Нет данных", "Прогресс-бар не найден"

    # --- Шаг 3: извлекаем ширину (дочерний div с классом или inline style) ---
    filled = progress_bar.find(
        lambda t: t.name == "div" and t.get("class") and any(
            "fill" in (c or "").lower() or "value" in (c or "").lower()
            for c in t.get("class", [])
        )
    )

    style = filled.get("style", "") if filled else progress_bar.get("style", "")
    width_text = ""
    # style = "width: 90%; background: ..."
    m = re.search(r"width\s*:\s*([\d.]+)\s*%", style)
    if m:
        width_pct = float(m.group(1))
        rating = _width_to_rating(width_pct)
        width_text = f"width: {width_pct:.0f}%;"
        return rating, width_text

    # Если ширина в числовом значении атрибута (например data-width="90")
    for attr in ("data-width", "data-value", "data-percent", "aria-valuenow"):
        val = (filled or progress_bar).get(attr)
        if val is not None:
            try:
                width_pct = float(val)
                rating = _width_to_rating(width_pct)
                width_text = f"width: {width_pct:.0f}%;"
                return rating, width_text
            except (ValueError, TypeError):
                continue

    # Последняя попытка — текст внутри filled
    if filled:
        text_val = filled.get_text(strip=True)
        if text_val:
            m2 = re.search(r"(\d+)", text_val)
            if m2:
                width_pct = float(m2.group(1))
                rating = _width_to_rating(width_pct)
                width_text = f"width: {width_pct:.0f}%;"
                return rating, width_text

    return "Нет данных", "Ширина не распознана"


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