#!/usr/bin/env python3
"""Добавить в CSV колонку AMORTIZING (0/1) по данным MOEX bondization. Результат — отдельный файл."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.moex_amort import moex_amortizing_isins  # noqa: E402


def main():
    src = ROOT / "data" / "bonds_current.csv"
    if not src.is_file():
        print(f"Нет файла: {src}")
        sys.exit(1)
    df = pd.read_csv(src)
    df.columns = [c.strip().upper() for c in df.columns]
    if "ISIN" not in df.columns:
        print("В файле нет колонки ISIN")
        sys.exit(1)
    isins = df["ISIN"].astype(str).tolist()
    print(f"Запрос MOEX для {len(isins)} строк (уникальные ISIN)…")
    amort = moex_amortizing_isins(
        isins,
        pause_sec=0.08,
        progress_callback=lambda n, total, isin: print(f"  {n}/{total} {isin}", flush=True),
    )
    df["AMORTIZING"] = df["ISIN"].astype(str).map(lambda x: 1 if x in amort else 0)
    out = ROOT / "data" / "bonds_current_with_amort_flags.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Готово: {out} (амортизируемых ISIN в датасете: {df['AMORTIZING'].sum()})")


if __name__ == "__main__":
    main()
