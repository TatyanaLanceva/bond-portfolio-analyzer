"""
src/ml_model.py
ML-прогноз справедливой доходности и ранжирование выпусков внутри однородных групп
(рейтинг × горизонт) по дополнительной доходности относительно модели.

Улучшенная версия: RandomForestRegressor + новые признаки (флаг ОФЗ,
модифицированная дюрация, число оставшихся купонов) + GridSearch гиперпараметров.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "data/ml_model.pkl"
SCALER_PATH = "data/ml_scaler.pkl"
METRICS_PATH = "data/ml_metrics.json"

# Признаки (базовые + новые)
FEATURES = [
    'YEARS_TO_MATURITY',      # срок до погашения (лет)
    'COUPONPERCENT',           # купонная ставка (%)
    'LASTPRICE',               # цена в % от номинала
    'RATING_SCORE',            # рейтинг-скор (0=AAA … 16=CCC)
    'FACEVALUE',               # номинал (₽)
    'IS_GOVERNMENT',           # 1 — ОФЗ, 0 — корпоративная
    'MOD_DURATION',            # модифицированная дюрация (годы)
    'COUPONS_REMAINING',       # число оставшихся купонных выплат
]


def _ensure_lastprice(df_m: pd.DataFrame) -> pd.DataFrame:
    """Цена в % от номинала; нужна для признака LASTPRICE."""
    out = df_m.copy()
    if 'LASTPRICE' in out.columns and not out['LASTPRICE'].isna().all():
        out['LASTPRICE'] = pd.to_numeric(out['LASTPRICE'], errors='coerce')
    else:
        fv = pd.to_numeric(out.get('FACEVALUE', 1000.0), errors='coerce').replace(0, np.nan).fillna(1000.0)
        dirty = pd.to_numeric(out.get('DIRTY_PRICE_RUB', fv), errors='coerce').fillna(fv)
        out['LASTPRICE'] = dirty / fv * 100.0
    return out


def _add_engineered_features(df_m: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет новые признаки:
    - IS_GOVERNMENT — флаг государственной облигации (RATING == 'AAA' и имя начинается с 'ОФЗ')
    - MOD_DURATION — модифицированная дюрация ≈ YEARS_TO_MATURITY / (1 + YTM_PCT/100)
    - COUPONS_REMAINING — количество оставшихся полугодовых купонов до погашения
    """
    out = df_m.copy()

    # IS_GOVERNMENT: ОФЗ — если рейтинг AAA и краткое имя начинается с "ОФЗ"
    out['IS_GOVERNMENT'] = (
        (out.get('RATING', '').astype(str).str.strip() == 'AAA') &
        (out.get('SHORTNAME', '').astype(str).str.strip().str.upper().str.startswith('ОФЗ'))
    ).astype(int)

    # MOD_DURATION: приближённая модифицированная дюрация
    ytm = pd.to_numeric(out.get('YTM_PCT', 0), errors='coerce').fillna(0)
    yrs = pd.to_numeric(out.get('YEARS_TO_MATURITY', 0), errors='coerce').fillna(0)
    out['MOD_DURATION'] = yrs / (1.0 + ytm / 100.0)
    out['MOD_DURATION'] = out['MOD_DURATION'].replace([np.inf, -np.inf], 0).fillna(0)

    # COUPONS_REMAINING: целое число купонных периодов (полугодия) до погашения
    cpn = pd.to_numeric(out.get('COUPONPERCENT', 0), errors='coerce').fillna(0)
    # Если купон > 0, то число выплат = 2 * YEARS_TO_MATURITY (округляем вверх)
    out['COUPONS_REMAINING'] = np.where(
        cpn > 0,
        np.ceil(yrs * 2.0).astype(int),
        0
    )
    return out


def prepare_data(df, ytm_min: float = 0.0, ytm_max: float = 50.0):
    """
    X, y и список фактических колонок для обучения / инференса.
    Фильтрует YTM_PCT в диапазон [ytm_min, ytm_max], чтобы
    аномальные выбросы MOEX (например, 60000%) не портили модель.
    """
    df_m = _ensure_lastprice(df)
    df_m = _add_engineered_features(df_m)
    use_cols = [c for c in FEATURES if c in df_m.columns]
    y = pd.to_numeric(df_m['YTM_PCT'], errors='coerce')
    mask = (y >= ytm_min) & (y <= ytm_max)
    df_m = df_m[mask].copy()
    y = y[mask]
    X = df_m[use_cols].fillna(0)
    return X, y, use_cols


def assign_ml_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Грубые однородные группы: класс кредитного риска × корзина по сроку.
    Внутри каждой группы сравниваем выпуски между собой.
    """
    out = df.copy()
    dur = pd.cut(
        np.clip(out['YEARS_TO_MATURITY'], 0.01, 50),
        bins=[0, 2.0, 5.0, 100.0],
        labels=['до 2 лет', '2–5 лет', 'свыше 5 лет'],
    )
    rscore = pd.to_numeric(out['RATING_SCORE'], errors='coerce').fillna(20)
    rt = pd.cut(
        rscore,
        bins=[-1, 5, 9, 30],
        labels=['рейтинг AAA–A', 'рейтинг BBB', 'рейтинг BB и ниже'],
    )
    out['ML_GROUP'] = rt.astype(str) + ' · ' + dur.astype(str)
    return out


def train_and_save_model(df):
    """
    Обучение RandomForestRegressor с GridSearch гиперпараметров.

    **Архитектура модели:**
    - Алгоритм: RandomForestRegressor (scikit-learn) — ансамбль решающих деревьев
    - Признаки (8): срок, купон, цена (% от номинала), рейтинг-скор, номинал,
      флаг ОФЗ, модифицированная дюрация, число оставшихся купонов
    - Целевая переменная: YTM_PCT (доходность к погашению)
    - Валидация: 80/20 train/test split; GridSearch 5-fold CV
    - Метрики: R² и MAE на тестовой выборке
    - Финальная модель: лучшая по GridSearch, обучена на всех данных
    """
    X, y, cols = prepare_data(df)
    if len(X) < 50:
        raise ValueError("Недостаточно данных для обучения (нужно ≥ 50 строк)")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # --- GridSearch для RandomForest ---
    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    gs = GridSearchCV(
        base_rf, param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_tr_s, y_tr)
    best_params = gs.best_params_
    best_est = gs.best_estimator_

    y_hat_te = best_est.predict(X_te_s)

    metrics = {
        "R2": float(r2_score(y_te, y_hat_te)),
        "MAE": float(mean_absolute_error(y_te, y_hat_te)),
        "features": cols,
        "algorithm": "RandomForestRegressor",
        "hyperparameters": {
            "grid_search_params": best_params,
            "cv_folds": 5,
            "scoring": "neg_mean_absolute_error",
        },
    }

    # --- Финальная модель на всех данных (с лучшими параметрами) ---
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)

    final_model = RandomForestRegressor(
        n_estimators=best_params.get('n_estimators', 200),
        max_depth=best_params.get('max_depth', 10),
        min_samples_split=best_params.get('min_samples_split', 2),
        min_samples_leaf=best_params.get('min_samples_leaf', 1),
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_full_s, y)

    Path("data").mkdir(exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler_full, f)
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return final_model, scaler_full, metrics


def load_model():
    """Загрузка артефактов; список признаков подхватывается из ml_metrics.json при наличии."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    features = list(FEATURES)
    p = Path(METRICS_PATH)
    if p.exists():
        try:
            meta = json.loads(p.read_text(encoding='utf-8'))
            features = meta.get('features') or features
        except json.JSONDecodeError:
            pass
    return model, scaler, features


def predict_ytm(df_input, model, scaler, features):
    """Точечный прогноз YTM (для совместимости с вкладкой и отчётами)."""
    df_m = _ensure_lastprice(df_input)
    df_m = _add_engineered_features(df_m)
    X = df_m[[c for c in features if c in df_m.columns]].reindex(columns=features, fill_value=0).fillna(0)
    return model.predict(scaler.transform(X))


def enrich_with_ml_scores(df_input, model, scaler, features):
    """
    Прогноз YTM, избыточная доходность (YTM − модель), группа и ранг внутри группы.
    Ранг 1 — наибольшая доп. доходность к модели среди «похожих» по рейтингу и сроку.
    """
    df = df_input.copy()
    df['PREDICTED_YTM'] = predict_ytm(df, model, scaler, features)
    df['EXCESS_YTM'] = pd.to_numeric(df['YTM_PCT'], errors='coerce') - df['PREDICTED_YTM']
    df = assign_ml_groups(df)
    df['ML_RANK_IN_GROUP'] = df.groupby('ML_GROUP', dropna=False)['EXCESS_YTM'].rank(
        ascending=False, method='min'
    ).astype(int)
    return df


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from src.core import load_bonds_data

    _, _, m = train_and_save_model(load_bonds_data())
    print("Обучение завершено:", json.dumps(m, indent=2, ensure_ascii=False))