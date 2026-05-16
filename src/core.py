"""
src/core.py
Ядро системы: загрузка данных, оптимизация портфеля, метрики и журнал.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

JOURNAL_PATH = "data/portfolio_journal.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 1. ЗАГРУЗКА ДАННЫХ
# ─────────────────────────────────────────────────────────────────────────────
def load_bonds_data(file_path: str = "data/bonds_current.csv") -> pd.DataFrame:
    """Загружает CSV и подготавливает данные."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Файл {file_path} не найден. Положите bonds_current.csv в папку data/")
    
    df = pd.read_csv(file_path)
    # Очистка имён колонок от пробелов
    df.columns = [c.strip().upper() for c in df.columns]

    required = ['ISIN', 'SHORTNAME', 'RATING', 'YEARS_TO_MATURITY', 
                'DIRTY_PRICE_RUB', 'YTM_PCT', 'COUPONPERCENT', 'FACEVALUE', 'MATDATE']
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
    
    # Приведение типов
    for col in ['DIRTY_PRICE_RUB', 'YTM_PCT', 'YEARS_TO_MATURITY', 'COUPONPERCENT', 'FACEVALUE']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df['COUPONPERCENT'] = df['COUPONPERCENT'].fillna(0.0)
    df['FACEVALUE'] = df['FACEVALUE'].fillna(1000.0)
    
    # Фильтрация
    df = df[df['ISIN'].astype(str).str.startswith('RU')]
    df = df[(df['DIRTY_PRICE_RUB'] > 0) & (df['YTM_PCT'] >= 0) & (df['YEARS_TO_MATURITY'] > 0)]
    df = df.dropna(subset=['MATDATE'])
    df['MATDATE'] = pd.to_datetime(df['MATDATE'], errors='coerce').dt.normalize()

    # Кодирование рейтинга
    RATING_MAP = {
        "AAA": 0, "AA+": 1, "AA": 2, "AA-": 3,
        "A+": 4, "A": 5, "A-": 6,
        "BBB+": 7, "BBB": 8, "BBB-": 9,
        "BB+": 10, "BB": 11, "BB-": 12,
        "B+": 13, "B": 14, "B-": 15, "CCC": 16
    }
    df['RATING_SCORE'] = df['RATING'].str.strip().map(RATING_MAP).fillna(20)
    return df.reset_index(drop=True)


def _csv_marked_as_amortizing(value) -> bool:
    """True, если в CSV бумага помечена как с амортизацией номинала (строку исключаем)."""
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return float(value) != 0.0
        except (TypeError, ValueError):
            return False
    s = str(value).strip().upper()
    return s in ('1', 'TRUE', 'YES', 'Y', 'ДА', '+', 'ИСТИНА', 'AMORT', 'AMORTIZATION')


AMORTIZING_CSV_COLUMNS = ('AMORTIZING', 'IS_AMORTIZING', 'HAS_AMORTIZATION')


def drop_amortizing_bonds(
    df: pd.DataFrame,
    *,
    enabled: bool,
    amortizing_isins: set | frozenset | None = None,
) -> pd.DataFrame:
    """
    Исключает амортизируемые выпуски: по столбцу CSV (приоритет) или по множеству ISIN (например, из MOEX).
    Если enabled, но нет ни столбца, ни множества — возвращает df без изменений.
    """
    if not enabled or df is None or df.empty:
        return df.copy() if df is not None else df
    out = df.copy()
    for col in AMORTIZING_CSV_COLUMNS:
        if col in out.columns:
            keep = ~out[col].map(_csv_marked_as_amortizing)
            return out.loc[keep].reset_index(drop=True)
    if amortizing_isins:
        sid = out['ISIN'].astype(str)
        return out[~sid.isin(amortizing_isins)].reset_index(drop=True)
    return out.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1б. ОТОБРАЖЕНИЕ И ВЫГРУЗКА
# ─────────────────────────────────────────────────────────────────────────────
BUYLIST_EXPORT_COLUMNS = [
    'ISIN', 'SHORTNAME', 'RATING', 'MATDATE', 'FACEVALUE', 'COUPONPERCENT',
    'YTM_PCT', 'YEARS_TO_MATURITY', 'DIRTY_PRICE_RUB', 'QUANTITY', 'INVESTED',
]


def portfolio_table_view(df: pd.DataFrame) -> pd.DataFrame:
    """Копия для таблиц в UI: дата как ДД.ММ.ГГГГ, аккуратные округления."""
    if df.empty:
        return df
    v = df.copy()
    if 'MATDATE' in v.columns:
        v['MATDATE'] = pd.to_datetime(v['MATDATE'], errors='coerce').dt.strftime('%d.%m.%Y')
    rnd_map = [
        ('DIRTY_PRICE_RUB', 2), ('INVESTED', 2), ('YTM_PCT', 2), ('COUPONPERCENT', 2),
        ('YEARS_TO_MATURITY', 2), ('FACEVALUE', 0),
    ]
    for col, nd in rnd_map:
        if col in v.columns:
            v[col] = pd.to_numeric(v[col], errors='coerce').round(nd)
    return v


def buy_list_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Фиксированный набор колонок для CSV: дата ГГГГ-ММ-ДД, без дублей и лишних полей."""
    if df.empty:
        return df
    cols = [c for c in BUYLIST_EXPORT_COLUMNS if c in df.columns]
    ex = df[cols].copy()
    if 'MATDATE' in ex.columns:
        ex['MATDATE'] = pd.to_datetime(ex['MATDATE'], errors='coerce').dt.strftime('%Y-%m-%d')
    rnd_map = [
        ('DIRTY_PRICE_RUB', 2), ('INVESTED', 2), ('YTM_PCT', 2), ('COUPONPERCENT', 2),
        ('YEARS_TO_MATURITY', 2), ('FACEVALUE', 0),
    ]
    for col, nd in rnd_map:
        if col in ex.columns:
            ex[col] = pd.to_numeric(ex[col], errors='coerce').round(nd)
    return ex


# ─────────────────────────────────────────────────────────────────────────────
# 2. ЖУРНАЛ СДЕЛОК
# ─────────────────────────────────────────────────────────────────────────────
def load_journal() -> pd.DataFrame:
    p = Path(JOURNAL_PATH)
    if p.exists():
        return pd.read_csv(p, dtype={'ISIN': str})
    return pd.DataFrame(columns=['DATE', 'ISIN', 'SHORTNAME', 'RATING', 'QUANTITY', 'PRICE', 'INVESTED'])

def get_current_holdings(journal_df: pd.DataFrame) -> dict:
    if journal_df.empty: return {}
    return journal_df.groupby('ISIN')['INVESTED'].sum().to_dict()


def clear_journal() -> None:
    """Удаляет файл журнала сделок (если есть)."""
    p = Path(JOURNAL_PATH)
    if p.exists():
        p.unlink()


def save_to_journal(journal_df: pd.DataFrame, new_portfolio: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if new_portfolio.empty: return journal_df
    new_rows = new_portfolio[['ISIN', 'SHORTNAME', 'RATING', 'QUANTITY', 'DIRTY_PRICE_RUB', 'INVESTED']].copy()
    new_rows.rename(columns={'DIRTY_PRICE_RUB': 'PRICE'}, inplace=True)
    new_rows['DATE'] = pd.Timestamp.today().strftime('%Y-%m-%d')
    new_rows['STRATEGY'] = strategy
    updated = pd.concat([journal_df, new_rows], ignore_index=True)
    updated.to_csv(JOURNAL_PATH, index=False, encoding='utf-8-sig')
    return updated

# ─────────────────────────────────────────────────────────────────────────────
# 3. АЛГОРИТМ ОПТИМИЗАЦИИ
# ─────────────────────────────────────────────────────────────────────────────
def build_portfolio(df: pd.DataFrame, strategy: str, budget: float,
                    min_rating: str, max_years: float, max_pct: float,
                    existing_holdings: dict = None, total_portfolio_value: float = None,
                    min_target_yield: float = 0.0) -> pd.DataFrame:
    if existing_holdings is None: existing_holdings = {}
    if total_portfolio_value is None: total_portfolio_value = budget

    RATING_ORDER = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC"]
    min_score = RATING_ORDER.index(min_rating) if min_rating in RATING_ORDER else 99

    pool = df[(df['RATING_SCORE'] <= min_score) & (df['YEARS_TO_MATURITY'] <= max_years)].copy()
    if min_target_yield is not None and min_target_yield > 0:
        pool = pool[pool['YTM_PCT'] >= float(min_target_yield)]
    pool = pool.sort_values('YTM_PCT', ascending=False)
    
    if pool.empty: return pd.DataFrame()
    
    portfolio = []
    remaining_budget = budget
    bought_isins = set()

    def try_buy(bond_row, target_spend=None):
        nonlocal remaining_budget
        price = bond_row['DIRTY_PRICE_RUB']
        if remaining_budget < price: return False
        
        current_in_bond = existing_holdings.get(bond_row['ISIN'], 0.0)
        max_allowed = (total_portfolio_value * max_pct) - current_in_bond
        
        limit = target_spend if target_spend else remaining_budget
        can_spend = min(remaining_budget, max(0, max_allowed), limit)
        
        qty = int(can_spend // price)
        if qty > 0:
            cost = qty * price
            portfolio.append({**bond_row.to_dict(), 'QUANTITY': qty, 'INVESTED': cost})
            bought_isins.add(bond_row['ISIN'])
            remaining_budget -= cost
            return True
        return False

    if strategy == "Ladder":
        target_per_year = budget / 5.0
        for year in range(1, 6):
            bucket = pool[(pool['YEARS_TO_MATURITY'] >= year - 0.5) & (pool['YEARS_TO_MATURITY'] < year + 0.5)]
            if bucket.empty: continue
            if not try_buy(bucket.iloc[0], target_per_year): break
            
    elif strategy == "Barbell":
        for part_pool in [pool[pool['YEARS_TO_MATURITY'] <= 1.5], pool[pool['YEARS_TO_MATURITY'] >= 4.0]]:
            if part_pool.empty: continue
            for _, bond in part_pool.head(2).iterrows():
                if not try_buy(bond, budget * 0.25): break
    else:  # Wheel
        for _, bond in pool.head(8).iterrows():
            if not try_buy(bond, budget / 6.0): break

    # Добиваем остаток
    if remaining_budget > budget * 0.1:
        for _, bond in pool.iterrows():
            if bond['ISIN'] not in bought_isins:
                if remaining_budget > bond['DIRTY_PRICE_RUB']:
                    try_buy(bond)

    return pd.DataFrame(portfolio) if portfolio else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# 4. МЕТРИКИ И КУПОНЫ
# ─────────────────────────────────────────────────────────────────────────────
def calculate_metrics(portfolio: pd.DataFrame) -> dict:
    if portfolio.empty:
        return {'Yield_Pct': 0.0, 'Risk_Years': 0.0, 'Coupon_Yearly': 0.0, 'Count': 0, 'Cost': 0.0}
    total_inv = portfolio['INVESTED'].sum()
    if total_inv == 0:
        return {'Yield_Pct': 0.0, 'Risk_Years': 0.0, 'Coupon_Yearly': 0.0, 'Count': 0, 'Cost': 0.0}
    
    coupon_yearly = (
        (pd.to_numeric(portfolio['FACEVALUE'], errors='coerce') *
         pd.to_numeric(portfolio['COUPONPERCENT'], errors='coerce').fillna(0.0) / 100.0 *
         pd.to_numeric(portfolio['QUANTITY'], errors='coerce').fillna(0))
        .fillna(0.0).sum()
    )
    return {
        'Yield_Pct': round(np.average(portfolio['YTM_PCT'], weights=portfolio['INVESTED']), 2),
        'Risk_Years': round(np.average(portfolio['YEARS_TO_MATURITY'], weights=portfolio['INVESTED']), 2),
        'Coupon_Yearly': round(coupon_yearly, 2),
        'Count': len(portfolio),
        'Cost': round(total_inv, 2)
    }

def calculate_full_cashflow(portfolio: pd.DataFrame, tax_rate: float = 0.0) -> pd.DataFrame:
    if portfolio.empty: return pd.DataFrame()
    all_flows = []
    today = pd.Timestamp.today().normalize()
    for _, bond in portfolio.iterrows():
        face = pd.to_numeric(bond['FACEVALUE'], errors='coerce')
        qty = pd.to_numeric(bond['QUANTITY'], errors='coerce')
        cpn = pd.to_numeric(bond['COUPONPERCENT'], errors='coerce')
        mat = bond['MATDATE']
        if pd.isna(face) or pd.isna(qty) or qty == 0:
            continue
        # Погашение номинала — всегда, даже при нулевом купоне
        if pd.notna(mat):
            all_flows.append({'YEAR': mat.year, 'TYPE': 'PRINCIPAL', 'NET': face * qty})
        if not pd.isna(cpn) and cpn > 0:
            annual = face * (cpn / 100.0) * qty
            semi = annual / 2.0
            pay = today + pd.DateOffset(months=6)
            while pay <= mat:
                all_flows.append({'YEAR': pay.year, 'TYPE': 'COUPON', 'NET': semi * (1 - tax_rate)})
                pay += pd.DateOffset(months=6)
    
    if not all_flows: return pd.DataFrame()
    df_f = pd.DataFrame(all_flows).groupby(['YEAR', 'TYPE']).agg({'NET': 'sum'}).reset_index()
    pv = df_f.pivot_table(index='YEAR', columns='TYPE', values='NET', fill_value=0).reset_index()
    for c in ['COUPON', 'PRINCIPAL']:
        if c not in pv.columns: pv[c] = 0.0
    pv['COUPON'] = pv['COUPON'].round(2)
    pv['PRINCIPAL'] = pv['PRINCIPAL'].round(2)
    pv['TOTAL'] = (pv['COUPON'] + pv['PRINCIPAL']).round(2)
    pv['CUMULATIVE'] = pv['TOTAL'].cumsum().round(2)
    return pv[['YEAR', 'COUPON', 'PRINCIPAL', 'TOTAL', 'CUMULATIVE']]

# ─────────────────────────────────────────────────────────────────────────────
# 5. РУСИФИКАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────
def ru(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'SHORTNAME': 'Эмитент', 'ISIN': 'ISIN', 'RATING': 'Рейтинг',
        'YTM_PCT': 'Доходность (YTM, %)', 'CURRENT_YIELD_PCT': 'Текущая доходность (%)',
        'YEARS_TO_MATURITY': 'Лет до погашения', 'DIRTY_PRICE_RUB': 'Цена с НКД (₽)',
        'FACEVALUE': 'Номинал (₽)', 'QUANTITY': 'Количество', 'INVESTED': 'Инвестировано (₽)',
        'MATDATE': 'Дата погашения', 'COUPONPERCENT': 'Купон (%)',
        'RATING_SCORE': 'Риск (код рейтинга)',
        'AMORTIZING': 'Амортизация (флаг)',
        'YEAR': 'Год', 'COUPON': 'Купоны (₽)', 'PRINCIPAL': 'Погашение (₽)',
        'TOTAL': 'Итого (₽)', 'CUMULATIVE': 'Накоплено (₽)',
        'DATE': 'Дата', 'PRICE': 'Цена покупки (₽)', 'STRATEGY': 'Стратегия',
        'PREDICTED_YTM': 'Прогноз YTM (%)', 'PREDICTED': 'Прогноз YTM (%)',
        'YTM_DIFF': 'Отклонение (%)', 'DIFF': 'Отклонение (п.п.)',
        'EXCESS_YTM': 'Доп. доходность к модели (п.п.)',
        'ML_GROUP': 'Группа ML (риск · срок)', 'ML_RANK_IN_GROUP': 'Ранг в группе (1 — лучше)',
        'SIGNAL': 'Сигнал',
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})