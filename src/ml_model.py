"""
src/ml_model.py
ML-прогноз справедливой доходности и ранжирование выпусков внутри однородных групп
(рейтинг × горизонт) по дополнительной доходности относительно модели.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "data/ml_model.pkl"
SCALER_PATH = "data/ml_scaler.pkl"
METRICS_PATH = "data/ml_metrics.json"

# Признаки без целевой YTM: модель учится на «рыночном профиле» бумаги
FEATURES = ['YEARS_TO_MATURITY', 'COUPONPERCENT', 'LASTPRICE', 'RATING_SCORE', 'FACEVALUE']


def _ensure_lastprice(df_m: pd.DataFrame) -> pd.DataFrame:
    """Цена в % от номинала (как в исходном проекте); нужна для признака LASTPRICE."""
    out = df_m.copy()
    if 'LASTPRICE' in out.columns and not out['LASTPRICE'].isna().all():
        out['LASTPRICE'] = pd.to_numeric(out['LASTPRICE'], errors='coerce')
    else:
        fv = pd.to_numeric(out.get('FACEVALUE', 1000.0), errors='coerce').replace(0, np.nan).fillna(1000.0)
        dirty = pd.to_numeric(out.get('DIRTY_PRICE_RUB', fv), errors='coerce').fillna(fv)
        out['LASTPRICE'] = dirty / fv * 100.0
    return out


def prepare_data(df, ytm_min: float = 0.0, ytm_max: float = 50.0):
    """
    X, y и список фактических колонок для обучения / инференса.
    Фильтрует YTM_PCT в диапазон [ytm_min, ytm_max], чтобы
    аномальные выбросы MOEX (например, 60000%) не портили модель.
    """
    df_m = _ensure_lastprice(df)
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
    Обучение HistGradientBoostingRegressor: предсказание YTM по структуре бумаги,
    метрики — на отложенной выборке; итоговая модель — на полном наборе с early stopping.

    **Архитектура модели:**
    - Алгоритм: градиентный бустинг над гистограммными деревьями (HistGradientBoostingRegressor)
    - Гиперпараметры: max_depth=5, learning_rate=0.08, early_stopping с patience=15 итераций
    - Признаки: срок (лет), купон (%), цена (% от номинала), рейтинг-скор, номинал (₽)
    - Целевая переменная: YTM_PCT (доходность к погашению)
    - Валидация: 80/20 train/test split; метрики R² и MAE на тестовой выборке
    - Финальная модель: обучается на 100% данных с той же архитектурой, но без валидационного
      разделения (использует всю информацию). Количество итераций фиксировано = best_iter,
      полученному на валидационном обучении.
    """
    X, y, cols = prepare_data(df)
    if len(X) < 30:
        raise ValueError("Недостаточно данных для обучения (нужно ≥ 30 строк)")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # --- Базовая модель с early stopping для определения оптимального числа итераций ---
    base_est = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=5,
        learning_rate=0.08,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=15,
    )
    base_est.fit(X_tr_s, y_tr)
    y_hat_te = base_est.predict(X_te_s)

    metrics = {
        "R2": float(r2_score(y_te, y_hat_te)),
        "MAE": float(mean_absolute_error(y_te, y_hat_te)),
        "features": cols,
        "algorithm": "HistGradientBoostingRegressor",
        "hyperparameters": {
            "max_depth": 5,
            "learning_rate": 0.08,
            "early_stopping": True,
            "n_iter_no_change": 15,
            "validation_fraction": 0.12,
            "initial_max_iter": 200,
        },
    }

    # Определяем наилучшее число итераций по валидационной модели
    _nit = getattr(base_est, "n_iter_", None)
    _arr = np.asarray(_nit).ravel() if _nit is not None else np.array([120])
    best_iter = int(_arr[0])
    # Небольшой запас + ограничение сверху для предотвращения переобучения
    final_iter = min(max(best_iter + 10, 50), 250)
    metrics["final_n_iter"] = final_iter

    # --- Финальная модель на всех данных ---
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X)

    model = HistGradientBoostingRegressor(
        max_iter=final_iter,
        max_depth=5,
        learning_rate=0.08,
        random_state=42,
    )
    model.fit(X_full_s, y)

    Path("data").mkdir(exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler_full, f)
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return model, scaler_full, metrics


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
    print("Обучение завершено:", m)
