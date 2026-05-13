"""
src/core.py
Ядро системы: загрузка, жадный алгоритм с учетом накопленного портфеля,
метрики, купонный поток и ведение журнала сделок.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

JOURNAL_PATH = "data/portfolio_journal.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 1. ЗАГРУЗКА ДАННЫХ И ЖУРНАЛА
# ─────────────────────────────────────────────────────────────────────────────
def load_bonds_data(file_path: str = "data/bonds_current.csv") -> pd.DataFrame:
    if not Path(file_path).exists():
        raise FileNotFoundError(f"❌ Файл {file_path} не найден. Положите bonds_current.csv в папку data/")
    
    df = pd.read_csv(file_path)
    df.columns = [c.strip().upper() for c in df.columns]
    
    required = ['ISIN', 'SHORTNAME', 'RATING', 'YEARS_TO_MATURITY', 
                'DIRTY_PRICE_RUB', 'YTM_PCT', 'COUPONPERCENT', 'FACEVALUE', 'MATDATE']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Отсутствуют колонки: {missing}")
        
    for col in ['DIRTY_PRICE_RUB', 'YTM_PCT', 'YEARS_TO_MATURITY']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['COUPONPERCENT'] = pd.to_numeric(df['COUPONPERCENT'], errors='coerce').fillna(0.0)
    df['FACEVALUE'] = pd.to_numeric(df['FACEVALUE'], errors='coerce').fillna(1000.0)
    
    df = df[df['ISIN'].astype(str).str.startswith('RU')]
    df = df[(df['DIRTY_PRICE_RUB'] > 0) & (df['YTM_PCT'] >= 0) & (df['YEARS_TO_MATURITY'] > 0)]
    df = df.dropna(subset=['MATDATE'])
    df['MATDATE'] = pd.to_datetime(df['MATDATE'])
    
    RATING_MAP = {"AAA":0, "AA+":1, "AA":2, "AA-":3, "A+":4, "A":5, "A-":6, 
                  "BBB+":7, "BBB":8, "BBB-":9, "BB+":10, "BB":11, "BB-":12, 
                  "B+":13, "B":14, "B-":15, "CCC":16}
    df['RATING_SCORE'] = df['RATING'].map(RATING_MAP).fillna(20)
    return df.reset_index(drop=True)

def load_journal() -> pd.DataFrame:
    """Загружает историю покупок из журнала."""
    p = Path(JOURNAL_PATH)
    if p.exists():
        return pd.read_csv(p, dtype={'ISIN': str})
    return pd.DataFrame(columns=['DATE', 'ISIN', 'SHORTNAME', 'RATING', 'QUANTITY', 'PRICE', 'INVESTED'])

def get_current_holdings(journal_df: pd.DataFrame) -> dict:
    """Возвращает {ISIN: сумма_вложений} для учета существующих позиций."""
    if journal_df.empty: return {}
    return journal_df.groupby('ISIN')['INVESTED'].sum().to_dict()

def save_to_journal(journal_df: pd.DataFrame, new_portfolio: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """Добавляет новые покупки в журнал и сохраняет CSV."""
    if new_portfolio.empty: return journal_df
    
    new_rows = new_portfolio[['ISIN', 'SHORTNAME', 'RATING', 'QUANTITY', 'DIRTY_PRICE_RUB', 'INVESTED']].copy()
    new_rows.rename(columns={'DIRTY_PRICE_RUB': 'PRICE'}, inplace=True)
    new_rows['DATE'] = pd.Timestamp.today().strftime('%Y-%m-%d')
    new_rows['STRATEGY'] = strategy_name
    
    updated = pd.concat([journal_df, new_rows], ignore_index=True)
    updated.to_csv(JOURNAL_PATH, index=False, encoding='utf-8-sig')
    return updated

# ─────────────────────────────────────────────────────────────────────────────
# 2. ЖАДНЫЙ АЛГОРИТМ С УЧЕТОМ НАКОПЛЕННОГО ПОРТФЕЛЯ
# ─────────────────────────────────────────────────────────────────────────────
def build_portfolio(df: pd.DataFrame, strategy: str, budget: float, 
                    min_rating: str, max_years: float, max_pct: float,
                    existing_holdings: dict = None, total_portfolio_value: float = None) -> pd.DataFrame:
    """
    Формирует портфель с учетом уже купленных бумаг.
    max_pct применяется к (текущая_стоимость + новый_бюджет).
    """
    if existing_holdings is None: existing_holdings = {}
    if total_portfolio_value is None: total_portfolio_value = budget
    
    RATING_ORDER = ["AAA","AA+","AA","AA-","A+","A","A-","BBB+","BBB","BBB-","BB+","BB","BB-","B+","B","B-","CCC"]
    min_score = RATING_ORDER.index(min_rating) if min_rating in RATING_ORDER else 99
    
    pool = df[(df['RATING_SCORE'] <= min_score) & (df['YEARS_TO_MATURITY'] <= max_years)].copy()
    pool = pool.sort_values('YTM_PCT', ascending=False)
    if pool.empty: return pd.DataFrame()
        
    portfolio = []
    remaining_budget = budget
    bought_isins = set()

    def try_buy(bond_row, target_spend=None):
        nonlocal remaining_budget
        price = bond_row['DIRTY_PRICE_RUB']
        if remaining_budget < price: return False
        
        # Лимит на бумагу = (Весь портфель * max_pct) - уже_вложено
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

    # --- ЛОГИКА СТРАТЕГИЙ ---
    if strategy == "Ladder":
        target_per_year = budget / 5.0
        for year in range(1, 6):
            bucket = pool[(pool['YEARS_TO_MATURITY'] >= year - 0.5) & (pool['YEARS_TO_MATURITY'] < year + 0.5)]
            if bucket.empty: continue
            if not try_buy(bucket.iloc[0], target_per_year): break
            
    elif strategy == "Barbell":
        short_pool = pool[pool['YEARS_TO_MATURITY'] <= 1.5]
        long_pool = pool[pool['YEARS_TO_MATURITY'] >= 4.0]
        for part_pool in [short_pool, long_pool]:
            if part_pool.empty: continue
            for _, bond in part_pool.head(2).iterrows():
                if not try_buy(bond, budget * 0.25): break
                
    else: # Wheel / Default
        for _, bond in pool.head(8).iterrows():
            if not try_buy(bond, budget / 6.0): break

    # Жадное заполнение остатка (>10% бюджета)
    if remaining_budget > budget * 0.1:
        for _, bond in pool.iterrows():
            if bond['ISIN'] in bought_isins: continue
            if remaining_budget <= bond['DIRTY_PRICE_RUB']: break
            try_buy(bond)
            
    return pd.DataFrame(portfolio) if portfolio else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# 3. МЕТРИКИ И КУПОНЫ
# ─────────────────────────────────────────────────────────────────────────────
def calculate_metrics(portfolio: pd.DataFrame) -> dict:
    if portfolio.empty:
        return {'Yield_Pct': 0.0, 'Risk_Years': 0.0, 'Coupon_Yearly': 0.0, 'Count': 0, 'Cost': 0.0}
    total_inv = portfolio['INVESTED'].sum()
    if total_inv == 0: return {'Yield_Pct': 0.0, 'Risk_Years': 0.0, 'Coupon_Yearly': 0.0, 'Count': 0, 'Cost': 0.0}
    
    return {
        'Yield_Pct': round(np.average(portfolio['YTM_PCT'], weights=portfolio['INVESTED']), 2),
        'Risk_Years': round(np.average(portfolio['YEARS_TO_MATURITY'], weights=portfolio['INVESTED']), 2),
        'Coupon_Yearly': round(sum(row['FACEVALUE']*(row['COUPONPERCENT']/100)*row['QUANTITY'] 
                                   for _, row in portfolio.iterrows() if pd.notna(row['COUPONPERCENT'])), 2),
        'Count': len(portfolio),
        'Cost': round(total_inv, 2)
    }

def calculate_full_cashflow(portfolio: pd.DataFrame, tax_rate: float = 0.0) -> pd.DataFrame:
    if portfolio.empty: return pd.DataFrame()
    all_flows = []
    today = pd.Timestamp.today().normalize()
    
    for _, bond in portfolio.iterrows():
        face, qty, cpn, mat = bond['FACEVALUE'], bond['QUANTITY'], bond['COUPONPERCENT'], bond['MATDATE']
        if pd.isna(cpn) or cpn == 0 or qty == 0: continue
            
        annual = face * (cpn / 100.0) * qty
        semi = annual / 2.0
        pay = today + pd.DateOffset(months=6)
        while pay <= mat:
            all_flows.append({'YEAR': pay.year, 'TYPE': 'COUPON', 'GROSS': semi, 'NET': semi*(1-tax_rate)})
            pay += pd.DateOffset(months=6)
        all_flows.append({'YEAR': mat.year, 'TYPE': 'PRINCIPAL', 'GROSS': face*qty, 'NET': face*qty})
        
    if not all_flows: return pd.DataFrame()
    df_f = pd.DataFrame(all_flows).groupby(['YEAR','TYPE']).agg({'GROSS':'sum','NET':'sum'}).reset_index()
    pv = df_f.pivot_table(index='YEAR', columns='TYPE', values='NET', fill_value=0).reset_index()
    for c in ['COUPON','PRINCIPAL']:
        if c not in pv.columns: pv[c] = 0.0
    pv['TOTAL'] = pv['COUPON'] + pv['PRINCIPAL']
    pv['CUMULATIVE'] = pv['TOTAL'].cumsum()
    return pv[['YEAR', 'COUPON', 'PRINCIPAL', 'TOTAL', 'CUMULATIVE']]

def get_top_picks(df, min_rating, max_years, top_n=5):
    RATING_ORDER = ["AAA","AA+","AA","AA-","A+","A","A-","BBB+","BBB","BBB-","BB+","BB","BB-","B+","B","B-","CCC"]
    min_score = RATING_ORDER.index(min_rating) if min_rating in RATING_ORDER else 99
    pool = df[(df['RATING_SCORE'] <= min_score) & (df['YEARS_TO_MATURITY'] <= max_years) & (df['YTM_PCT'] > 0)]
    pool['SCORE'] = (pool['YTM_PCT'] * 2) - (pool['YEARS_TO_MATURITY'] * 0.5) - (pool['RATING_SCORE'] * 0.2)
    return pool.sort_values('SCORE', ascending=False).head(top_n)[['SHORTNAME','ISIN','RATING','YTM_PCT','YEARS_TO_MATURITY','COUPONPERCENT','DIRTY_PRICE_RUB']]