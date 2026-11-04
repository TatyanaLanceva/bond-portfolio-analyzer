"""CLI: обогащение CSV реальными рейтингами smart-lab.

Использование:
    python enrich_ratings.py                        # обогатить data/moex_bonds.csv
    python enrich_ratings.py --out data/bonds_r.csv # сохранить в другой файл
    python enrich_ratings.py --limit 20             # отладка на 20 строках
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.smartlab_ratings import enrich_csv_with_ratings

DEFAULT_IN = Path("data/moex_bonds.csv")

log = logging.getLogger("enrich_ratings")


def main() -> int:
    parser = argparse.ArgumentParser(description="Обогатить CSV рейтингами smart-lab")
    parser.add_argument("--in", dest="src", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", dest="dst", type=Path, default=None,
                        help="куда сохранить (по умолчанию перезаписать исходный)")
    parser.add_argument("--limit", type=int, default=None,
                        help="обработать только первые N строк")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.src.exists():
        log.error("Файл не найден: %s", args.src)
        return 1

    log.info("Загружено %d строк из %s", 0, args.src)

    if args.limit:
        log.info("Ограничено до %d строк (--limit)", args.limit)

    try:
        processed, total = enrich_csv_with_ratings(args.src, args.dst or args.src, limit=args.limit)
    except Exception:
        log.exception("Обогащение упало")
        return 2

    log.info("Готово: %d/%d строк с рейтингом → %s", processed, total, args.dst or args.src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
