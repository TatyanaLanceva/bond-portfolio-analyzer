"""
Разовый прогон: парсит рейтинги для всех корпоративных бумаг из bonds_current.csv.
Используется однократно для первичного заполнения датасета. 
Затем обновления делать через enrich_ratings.py (по умолчанию использует кэш).

Запуск: python tools/first_ratings_run.py
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.smartlab_ratings import RatingCache, get_bond_rating
from src.core import load_bonds_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_PATH = "data/rating_cache.json"
CSV_PATH = "data/bonds_current.csv"
PAUSE_SEC = 2.5
RETRY_COUNT = 3
SAVE_INTERVAL = 50  # сохранять кэш каждые N запросов


def main():
    cache = RatingCache(CACHE_PATH)
    
    # 1. Очищаем "Нет данных" из кэша — они с предыдущего неудачного прогона
    to_clear = [k for k, v in cache._data.items() if v[0] in ("Нет данных", "")]
    if to_clear:
        logger.info("Очищаем %d устаревших записей 'Нет данных' из кэша...", len(to_clear))
        for k in to_clear:
            del cache._data[k]
        cache.save()
    
    # 2. Загружаем CSV
    logger.info("Загрузка %s...", CSV_PATH)
    df = load_bonds_data(CSV_PATH)
    total = len(df)
    
    # 3. Отбираем корпоративные (RATING != AAA)
    corp_mask = df["RATING"].astype(str).str.strip() != "AAA"
    corp_idx = df.index[corp_mask].tolist()
    
    # 4. Определяем, что нужно парсить
    to_fetch = {}  # index -> isin
    already_ok = 0
    
    for idx in corp_idx:
        isin = str(df.at[idx, "ISIN"]).strip()
        if not isin.startswith("RU"):
            continue
        
        cached = cache.get(isin)
        if cached and cached[0] not in ("Нет данных", "Нет ISIN", "Ошибка", ""):
            df.at[idx, "RATING"] = cached[0]
            already_ok += 1
        else:
            to_fetch[idx] = isin
    
    total_corp = len(corp_idx)
    total_fetch = len(to_fetch)
    
    logger.info(
        "Всего: %d | Корп: %d | Уже в кэше: %d | Нужно спарсить: %d",
        total, total_corp, already_ok, total_fetch,
    )
    
    if not to_fetch:
        logger.info("Все рейтинги уже в кэше. Просто обновляем CSV...")
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print("✅ CSV обновлён. Ничего парсить не нужно.")
        return
    
    # 5. Парсинг
    success = 0
    failed = 0
    start_time = time.time()
    estimated = total_fetch * (PAUSE_SEC + 3)  # ~3 сек на запрос + пауза
    
    logger.info(
        "Начинаем парсинг %d бумаг (пауза %.1f сек). Оценка: ~%.0f мин",
        total_fetch, PAUSE_SEC, estimated / 60,
    )
    print(f"\n{'='*60}")
    print(f"📡 Парсинг {total_fetch} корпоративных облигаций со smart-lab.ru")
    print(f"⏱  Оценочное время: ~{estimated/60:.0f} мин")
    print(f"{'='*60}\n")
    
    for n, (idx, isin) in enumerate(to_fetch.items(), 1):
        rating, color = get_bond_rating(isin, timeout=20.0, max_retries=RETRY_COUNT)
        
        if rating not in ("Нет данных", "Нет ISIN", "Ошибка", ""):
            df.at[idx, "RATING"] = rating
            cache.set(isin, rating, color)
            success += 1
        else:
            failed += 1
            if not color:
                color = ""
            cache.set(isin, rating, color)
        
        # Прогресс
        if n % max(1, total_fetch // 20) == 0 or n == 1 or n == total_fetch:
            elapsed = time.time() - start_time
            pct = n / total_fetch * 100
            rate = n / elapsed if elapsed > 0 else 0
            remaining = (total_fetch - n) / rate if rate > 0 else 0
            print(
                f"[{n:>4d}/{total_fetch}] ({pct:5.1f}%) "
                f"{rate:.1f} б/мин | осталось ~{remaining/60:.0f} мин | "
                f"✅{success} ❌{failed} | {isin} → {rating}"
            )
        
        # Сохраняем кэш
        if n % SAVE_INTERVAL == 0:
            cache.save()
        
        # Пауза
        if n < total_fetch:
            time.sleep(PAUSE_SEC)
    
    # 6. Финальное сохранение
    elapsed_total = time.time() - start_time
    cache.save()
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    
    print(f"\n{'='*60}")
    print(f"✅ ГОТОВО! За {elapsed_total/60:.1f} мин")
    print(f"   Успешно: {success}")
    print(f"   Не найдено: {failed}")
    print(f"   CSV обновлён: {CSV_PATH}")
    print(f"   Кэш сохранён: {CACHE_PATH}")
    
    # Статистика
    print(f"\n📊 Распределение рейтингов (корпоративные, {total_corp} бумаг):")
    ratings = df.loc[corp_idx, "RATING"].value_counts()
    for r, cnt in ratings.items():
        pct = cnt / total_corp * 100
        print(f"   {r:>6s}: {cnt:>4d} ({pct:5.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Первичный парсинг рейтингов")
    parser.add_argument("--pause", type=float, default=2.5, help="Пауза между запросами (сек)")
    parser.add_argument("--csv", default="data/bonds_current.csv", help="Путь к CSV")
    parser.add_argument("--retries", type=int, default=3, help="Кол-во повторов при ошибке")
    args = parser.parse_args()
    
    PAUSE_SEC = args.pause
    CSV_PATH = args.csv
    RETRY_COUNT = args.retries
    
    main()