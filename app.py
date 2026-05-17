"""
Bond Portfolio Analyzer v2.0
Главное приложение Streamlit
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import date
import json

# Добавляем папку src в путь
sys.path.append(str(Path(__file__).parent))

from src.core import (
    load_bonds_data,
    build_portfolio,
    calculate_metrics,
    calculate_full_cashflow,
    ru,
    load_journal,
    get_current_holdings,
    save_to_journal,
    portfolio_table_view,
    buy_list_export_frame,
    drop_amortizing_bonds,
    AMORTIZING_CSV_COLUMNS,
)

try:
    from src.core import clear_journal
except ImportError:
    # Совместимость со старыми копиями core.py / кэшем; путь как в core.JOURNAL_PATH
    def clear_journal() -> None:
        p = Path(__file__).resolve().parent / "data" / "portfolio_journal.csv"
        if p.is_file():
            p.unlink()

# Безопасный импорт ML
try:
    from src.ml_model import load_model, enrich_with_ml_scores
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


@st.cache_data(ttl=3600)
def get_bonds_base():
    return load_bonds_data()


@st.cache_data(ttl=86400)
def cached_moex_amortizing_isins(isins_key: tuple) -> frozenset:
    from src.moex_amort import moex_amortizing_isins

    return frozenset(moex_amortizing_isins(list(isins_key)))


# ─────────────────────────────────────────────────────────────────────────────
# 🎨 Настройка страницы и CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Анализатор портфеля v2.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stExpander {
        border: 1px solid #00382b;
        border-radius: 10px;
        background-color: #f0f9f4;
    }
    .stMetric label { font-size: 0.8em; }
    .metric-value { font-size: 1.5em; color: #00382b; }
    /* Красная кнопка удаления */
    div.stButton > button[data-baseweb="button"][style*="background-color: rgb(255, 75, 75)"] {
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Анализатор облигационного портфеля")
st.caption("📊 Bond Portfolio Analyzer — автоматизированный подбор и анализ облигационных портфелей")

if st.session_state.pop("moex_refresh_ok", False):
    st.success("Котировки обновлены с MOEX → `data/bonds_current.csv` (резервная копия: `.csv.bak`).")

# ─────────────────────────────────────────────────────────────────────────────
# 📥 Загрузка данных (сырой CSV + опциональное исключение амортизируемых)
# ─────────────────────────────────────────────────────────────────────────────
try:
    _bonds_base = get_bonds_base()
except Exception as e:
    st.error(f"❌ Ошибка загрузки: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️ Боковая панель (Настройки)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Параметры расчета")

    exclude_amortizing = st.checkbox(
        "Исключить амортизируемые облигации",
        value=True,
        help=(
            "Приоритет: столбец AMORTIZING / IS_AMORTIZING в CSV (1 — есть амортизация номинала). "
            "Иначе — запрос MOEX bondization (только события с типом amortization)."
        ),
    )

    st.divider()
    st.subheader("Данные MOEX")

    fetch_smartlab = st.checkbox(
        "🔍 Парсить рейтинги со smart-lab.ru",
        value=False,
        help=(
            "Вместо заглушки BBB для корпоративных бумаг: парсит реальные рейтинги "
            "(АКРА, Эксперт РА, НКР, НРА). Запросы идут с паузой ~2 сек. "
            "Результат кэшируется в data/rating_cache.json."
        ),
    )

    if st.button(
        "🌐 Обновить котировки с MOEX",
        help=(
            "Доски TQCB + TQOB, рубли. Перезаписывает data/bonds_current.csv "
            "(резерв .csv.bak). Рейтинг: ОФЗ→AAA, корп →" + ("smart-lab.ru (если включено)" if fetch_smartlab else "BBB (заглушка)") + "."
        ),
        use_container_width=True,
    ):
        try:
            from src.moex_bonds_update import save_moex_bonds_csv

            with st.spinner(f"Загрузка с MOEX… {'+ парсинг smart-lab.ru' if fetch_smartlab else ''} обычно 30–90 с"):
                save_moex_bonds_csv(fetch_ratings=fetch_smartlab, ratings_pause=2.0)
            get_bonds_base.clear()
            cached_moex_amortizing_isins.clear()
            st.session_state["moex_refresh_ok"] = True
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"MOEX: {e}")

    budget = st.number_input("💰 Бюджет (₽)", min_value=10000, value=500000, step=50000)
    
    tax_rate = st.selectbox(
        "🏛️ Налоговый режим",
        options=[("0% (ИИС / Льгота)", 0.0), ("13% (НДФЛ)", 0.13)],
        format_func=lambda x: x[0],
        index=0,
        help="Влияет на расчет чистого купонного дохода в отчетах"
    )[1]

    target_yield = st.number_input(
        "🎯 Мин. YTM (%, целевая доходность)",
        min_value=0.0,
        max_value=30.0,
        value=12.0,
        step=0.5,
        help="В пул для подбора попадают только облигации с YTM ≥ этого значения. 0 — без ограничения по доходности.",
    )

    min_rating = st.selectbox("🛡️ Мин. рейтинг", 
        ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"], index=2)
    
    max_years = st.slider("⏳ Макс. срок (лет)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    max_pct = st.slider("📊 Макс. доля бумаги (%)", min_value=5, max_value=50, value=20, step=5)
    
    strategy = st.radio(
        "🏗️ Стратегия",
        options=["Ladder", "Barbell", "Wheel"],
        format_func=lambda x: {"Ladder": "🪜 Лестница", "Barbell": "🏋️ Гантеля", "Wheel": "🎡 Колесо"}[x]
    )

    # Пояснения к стратегиям
    with st.expander("ℹ️ Описание стратегий"):
        st.markdown("""
        **🪜 Лестница (Ladder)**  
        Равномерное распределение бюджета по годам погашения (1, 2, 3, 4, 5 лет).  
        *Плюсы:* Стабильный поток погашений, низкий риск изменения ставки.*

        **🏋️ Гантеля (Barbell)**  
        Инвестиции только в короткие (до 1.5 лет) и длинные (от 4 лет) бумаги.  
        *Плюсы:* Баланс ликвидности и фиксации высокой доходности.*

        **🎡 Колесо (Wheel / Max Yield)**  
        Жадный алгоритм: покупка самых доходных бумаг в рамках лимитов.  
        *Плюсы:* Максимальная прибыль на короткой дистанции.*
        """)

    st.divider()
    run_opt = st.button("🔄 Рассчитать портфель", type="primary", use_container_width=True)

    st.divider()
    if st.button("🗑️ Очистить журнал сделок", type="secondary", use_container_width=True):
        clear_journal()
        st.rerun()

_amort_isins = None
if exclude_amortizing and not any(c in _bonds_base.columns for c in AMORTIZING_CSV_COLUMNS):
    _amort_isins = set(cached_moex_amortizing_isins(tuple(sorted(_bonds_base["ISIN"].astype(str).unique()))))

df_raw = drop_amortizing_bonds(
    _bonds_base,
    enabled=exclude_amortizing,
    amortizing_isins=_amort_isins,
)

st.success(
    f"✅ В выборке **{len(df_raw)}** облигаций "
    f"(исходный файл: **{len(_bonds_base)}** строк)."
)
if exclude_amortizing and len(df_raw) < len(_bonds_base):
    _src = "CSV (столбец AMORTIZING)" if any(
        c in _bonds_base.columns for c in AMORTIZING_CSV_COLUMNS
    ) else "MOEX bondization"
    st.caption(
        f"Исключены амортизируемые по данным **{_src}**: **{len(_bonds_base) - len(df_raw)}** строк. "
        f"Кэш MOEX обновляется не чаще раза в сутки."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Инициализация состояния сессии
# ─────────────────────────────────────────────────────────────────────────────
if "portfolio_cache" not in st.session_state:
    st.session_state.portfolio_cache = {}
if "last_params" not in st.session_state:
    st.session_state.last_params = None

# ─────────────────────────────────────────────────────────────────────────────
# 🚀 Логика оптимизации
# ─────────────────────────────────────────────────────────────────────────────
current_params = (
    budget, target_yield, min_rating, max_years, max_pct, strategy, tax_rate, exclude_amortizing,
)

if run_opt or st.session_state.last_params != current_params:
    with st.spinner("⏳ Оптимизация..."):
        try:
            journal_df = load_journal()
            existing = get_current_holdings(journal_df)
            total_val = budget + sum(existing.values())
            
            portfolio = build_portfolio(
                df=df_raw, strategy=strategy, budget=budget, 
                min_rating=min_rating, max_years=max_years, 
                max_pct=max_pct / 100.0, 
                existing_holdings=existing, total_portfolio_value=total_val,
                min_target_yield=target_yield,
            )
            
            if not portfolio.empty:
                st.session_state.portfolio_cache['current'] = portfolio
                st.session_state.last_params = current_params
            else:
                st.warning("⚠️ Нет подходящих бумаг: проверьте рейтинг, срок, **мин. YTM** и лимиты.")
                st.session_state.portfolio_cache['current'] = pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
            st.session_state.portfolio_cache['current'] = pd.DataFrame()

portfolio = st.session_state.portfolio_cache.get('current', pd.DataFrame())

# ─────────────────────────────────────────────────────────────────────────────
# 📑 Вкладки
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Портфель", "💰 Поток", "🛒 Покупки", "📜 Журнал", "🤖 ML-Прогноз"
])

# ▼ Вкладка 1: Портфель
with tab1:
    st.subheader("📊 Результаты оптимизации")
    if portfolio.empty:
        st.info("Нажмите «Рассчитать портфель» в меню слева.")
    else:
        m = calculate_metrics(portfolio)
        # Учитываем налог в метриках для отображения
        net_yield = m['Yield_Pct'] * (1 - tax_rate) if tax_rate > 0 else m['Yield_Pct']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 Бумаг", m['Count'])
        c2.metric("💵 Вложено", f"{m['Cost']:,.0f} ₽")
        c3.metric(f"📈 Доходность ({'ИИС' if tax_rate==0 else 'НДФЛ'})", f"{net_yield:.2f}%")
        c4.metric("🎁 Купон (Чистыми)", f"{m['Coupon_Yearly'] * (1-tax_rate):,.0f} ₽")

        # Таблица с красивым форматированием
        st.dataframe(
            ru(portfolio_table_view(portfolio)).style.format({
                'Доходность (YTM, %)': '{:.2f}%',
                'Лет до погашения': '{:.2f}',
                'Цена с НКД (₽)': '{:,.2f}',
                'Купон (%)': '{:.2f}',
                'Номинал (₽)': '{:,.0f}',
                'Риск (код рейтинга)': '{:.0f}',
                'Количество': '{:.0f}',
                'Инвестировано (₽)': '{:,.2f}',
            }).highlight_max(subset=['Доходность (YTM, %)'], color='#caeddb'),
            use_container_width=True, height=450
        )
        
        # График портфеля
        fig = px.scatter(
            portfolio, x='YEARS_TO_MATURITY', y='YTM_PCT', size='INVESTED', color='RATING',
            hover_name='SHORTNAME', title="Доходность vs Срок"
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Срок (лет)", yaxis_title="YTM (%)")
        st.plotly_chart(fig, use_container_width=True)

# ▼ Вкладка 2: Купонный поток
with tab2:
    st.subheader("💰 Денежный поток")
    if portfolio.empty:
        st.info("Сначала сформируйте портфель.")
    else:
        with st.expander("ℹ️ Логика расчета потока"):
            st.markdown(f"""
            1. **Купоны**: Учитывается ставка `COUPONPERCENT`. Выплаты раз в полгода.
            2. **Налог**: Применена ставка **{int(tax_rate*100)}%** к купонному доходу.
            3. **Погашение**: Номинал (`FACEVALUE`) возвращается в дату `MATDATE`. Налог на курсовую разницу не учитывается в упрощенной модели.
            4. **Накопленный доход**: Сумма всех поступлений по годам.
            """)

        # Расчет с учетом выбранного налога
        cf = calculate_full_cashflow(portfolio, tax_rate=tax_rate)
        
        if not cf.empty:
            # График: Stack Bar (Купоны + Погашение)
            fig_cf = go.Figure()
            fig_cf.add_trace(go.Bar(name='Купоны (Net)', x=cf['YEAR'], y=cf['COUPON'], marker_color='#caeddb'))
            fig_cf.add_trace(go.Bar(name='Погашение номинала', x=cf['YEAR'], y=cf['PRINCIPAL'], marker_color='#00382b'))
            
            fig_cf.update_layout(
                barmode='stack', 
                title="Поступления по годам (Купоны + Погашение)",
                xaxis_title="Год", yaxis_title="Сумма (₽)",
                plot_bgcolor='rgba(0,0,0,0)', 
                height=400
            )
            st.plotly_chart(fig_cf, use_container_width=True)

            st.dataframe(
                ru(cf).style.format({
                    'Купоны (₽)': '{:.2f}',
                    'Погашение (₽)': '{:.2f}',
                    'Итого (₽)': '{:.2f}',
                    'Накоплено (₽)': '{:.2f}',
                }),
                use_container_width=True
            )
        else:
            st.warning("Нет данных о купонах для выбранных бумаг.")

# ▼ Вкладка 3: Покупки
with tab3:
    st.subheader("🛒 Заявка на покупку")
    if portfolio.empty:
        st.info("Нет портфеля для покупки.")
    else:
        buy_list = portfolio[portfolio['QUANTITY'] > 0].copy()
        
        # Анализ долей относительно ВСЕГО портфеля
        journal_df = load_journal()
        existing_val = sum(journal_df['INVESTED']) if not journal_df.empty else 0
        total_portfolio_val = existing_val + m['Cost']
        
        st.info(f"📈 Общая стоимость портфеля: **{total_portfolio_val:,.0f} ₽** (Учитывая старые сделки)")
        
        st.dataframe(
            ru(portfolio_table_view(buy_list)).style.format({
                'Цена с НКД (₽)': '{:,.2f}',
                'Купон (%)': '{:.2f}',
                'Доходность (YTM, %)': '{:.2f}%',
                'Номинал (₽)': '{:,.0f}',
                'Количество': '{:.0f}',
                'Инвестировано (₽)': '{:,.2f}',
            }),
            use_container_width=True
        )
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Добавить выбранные бумаги в портфель", type="primary", use_container_width=True):
                save_to_journal(journal_df, buy_list, strategy)
                st.success("🎉 Сделки сохранены! Обновите портфель.")
        
        with c2:
            if st.button("📥 Скачать CSV заявки", use_container_width=True):
                csv_df = ru(buy_list_export_frame(buy_list))
                csv = csv_df.to_csv(index=False, encoding='utf-8-sig', lineterminator='\n')
                st.download_button("Скачать CSV", csv.encode('utf-8-sig'), f"buy_{date.today()}.csv", "text/csv;charset=utf-8")

# ▼ Вкладка 4: Журнал
with tab4:
    st.subheader("📜 История сделок")
    j_df = load_journal()
    if j_df.empty:
        st.info("Журнал пуст.")
    else:
        st.dataframe(ru(j_df).style.format({'Цена покупки (₽)': '{:,.2f}', 'Инвестировано (₽)': '{:,.0f}'}), use_container_width=True, height=400)

# ▼ Вкладка 5: ML
with tab5:
    st.subheader("🤖 Прогноз справедливой доходности и ранжирование")
    
    with st.expander("ℹ️ Описание ML-модуля"):
        st.markdown("""
        **🎯 Задача:** Оценить «базовую» YTM по структуре бумаги (срок, купон, цена, рейтинг, номинал)
        и посчитать **дополнительную доходность** относительно модели. Внутри однородных групп
        (**класс рейтинга × корзина срока**) выпуски **ранжируются** — удобно сравнивать кандидатов
        и итоговые портфели по стратегиям.

        ---
        ### 🏗 Архитектура модели

        | Компонент | Описание |
        |---|---|
        | **Алгоритм** | `RandomForestRegressor` (scikit-learn) — ансамбль решающих деревьев |
        | **GridSearch** | 5-fold CV по n_estimators, max_depth, min_samples_split, min_samples_leaf |
        | **Гиперпараметры** | Подобраны автоматически (лучшая комбинация по MAE) |
        | **Валидация** | 80/20 train/test split (random_state=42) |

        **Признаки (8):**
        - `YEARS_TO_MATURITY` — срок до погашения (лет)
        - `COUPONPERCENT` — купонная ставка (%)
        - `LASTPRICE` — цена в % от номинала
        - `RATING_SCORE` — рейтинг-скор (0=AAA … 16=CCC)
        - `FACEVALUE` — номинал (₽)
        - `IS_GOVERNMENT` — флаг ОФЗ (1) / корпоративная (0)
        - `MOD_DURATION` — модифицированная дюрация (годы)
        - `COUPONS_REMAINING` — число оставшихся купонных выплат

        **Целевая переменная:** `YTM_PCT` — доходность к погашению (%)

        ---
        ### 🔄 Процесс обучения
        1. Данные разделяются на train (80%) и test (20%) с сохранением стратификации по рейтингу.
        2. Все признаки масштабируются через `StandardScaler`.
        3. Базовая модель обучается с `early_stopping` — определяется оптимальное число итераций,
           при котором валидационная ошибка перестаёт улучшаться.
        4. Финальная модель обучается на **100% данных** с тем же числом итераций (без валидационного разделения).
        5. На тестовой выборке считаются метрики R² и MAE.

        ---
        ### 📊 Предобработка датасета

        Исходный CSV-файл с MOEX содержит **~2100 строк** по рублёвым облигациям РФ.
        Этапы предобработки перед обучением:

        1. **Загрузка и очистка колонок:** имена приводятся к верхнему регистру, пробелы удаляются.
        2. **Проверка наличия обязательных полей:** ISIN, SHORTNAME, RATING, YEARS_TO_MATURITY,
           DIRTY_PRICE_RUB, YTM_PCT, COUPONPERCENT, FACEVALUE, MATDATE.
        3. **Приведение типов:** числовые поля конвертируются через `pd.to_numeric`; пропуски
           заполняются (`COUPONPERCENT → 0`, `FACEVALUE → 1000`).
        4. **Фильтрация по ISIN:** только бумаги с префиксом `RU` (российские эмитенты).
        5. **Фильтрация по цене и доходности:** `DIRTY_PRICE_RUB > 0`, `YTM_PCT ≥ 0`,
           `YEARS_TO_MATURITY > 0`.
        6. **Расчёт признака LASTPRICE:** цена в % от номинала = `DIRTY_PRICE_RUB / FACEVALUE × 100`.
        7. **Расчёт RATING_SCORE:** текстовый рейтинг маппится в числовой скор (0 = AAA … 16 = CCC).
        8. **Фильтрация выбросов YTM:** для ML-модели YTM ограничивается диапазоном **0–50%**,
           чтобы аномальные значения (например, 60628%) не искажали обучение.
           После фильтрации остаётся **~1770 строк**.

        ---
        ### 📊 Метрики качества (на отложенной выборке)

        * **R² (коэффициент детерминации):** доля дисперсии YTM, объяснённая моделью.
          1.0 = идеальное предсказание, 0.0 = предсказание среднего.
        * **MAE (средняя абсолютная ошибка):** среднее отклонение прогноза от
          фактической YTM в процентных пунктах.

        **Интерпретация результатов:**
        * **Дополнительная доходность** = фактическая YTM − прогноз модели.
          Положительное значение → рынок даёт больше, чем типично для похожих бумаг.
        * **Ранг в группе** = 1 — наибольшая доп. доходность среди соседей по рейтингу и сроку.
        """)

    if ML_AVAILABLE:
        try:
            model, scaler, features = load_model()
            st.success("✅ Модель загружена")

            # Статистика по датасету (из bonds_current.csv)
            df_bonds = _bonds_base
            total_raw = len(df_bonds)
            ytm_col = pd.to_numeric(df_bonds['YTM_PCT'], errors='coerce')
            after_ml_filter = ((ytm_col >= 0) & (ytm_col <= 50)).sum()

            metrics_path = Path("data/ml_metrics.json")
            if metrics_path.exists():
                try:
                    _m = json.loads(metrics_path.read_text(encoding="utf-8"))
                    hp = _m.get("hyperparameters", {})
                    r2_val = _m.get('R2', '—')
                    mae_val = _m.get('MAE', '—')
                    final_n = _m.get('final_n_iter', '—')

                    # --- Карточка метрик с пояснениями ---
                    st.markdown("### 📈 Результаты обученной модели")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("R² (тест)", f"{r2_val}")
                    c2.metric("MAE (тест), п.п.", f"{mae_val}")
                    c3.metric("Итераций (финал)", f"{final_n}")

                    # Пояснение метрик прямо под карточкой
                    with st.expander("📖 Разъяснение метрик", expanded=True):
                        st.markdown(f"""
                        - **R² = {r2_val}** — модель объясняет **{float(r2_val)*100:.1f}%** дисперсии доходности YTM.
                          Чем ближе к 1.0, тем точнее прогноз.
                        - **MAE = {mae_val} п.п.** — средняя абсолютная ошибка прогноза.
                          Например, для бумаги с фактической YTM = 15%, модель в среднем предскажет
                          значение в диапазоне 15% ± {mae_val} п.п.
                        - **Итераций = {final_n}** — количество деревьев в ансамбле после early stopping
                          (больше не всегда лучше — ранняя остановка предотвращает переобучение).
                        """)

                    # --- Предобработка датасета ---
                    with st.expander("📋 Данные о предобработке датасета", expanded=True):
                        st.markdown(f"""
                        | Параметр | Значение |
                        |---|---|
                        | **Исходное число строк** (из MOEX) | {total_raw} |
                        | **После фильтрации YTM [0, 50]** | {after_ml_filter} |
                        | **Отсеяно выбросов** | {total_raw - after_ml_filter} |
                        | **Признаков для модели** | {len(features)} |
                        | **Соотношение train/test** | 80% / 20% |

                        **Почему фильтруем YTM?**  
                        MOEX иногда возвращает аномальные значения доходности
                        (например, -99.98% или 60628%). Они не отражают реальной
                        рыночной ситуации и могут быть связаны с техническими сбоями
                        в расчёте эффективной доходности для коротких/дефолтных бумаг.
                        Модель обучается только на реалистичном диапазоне 0–50%,
                        что даёт **устойчивое качество предсказаний**.
                        """)

                    # --- Архитектура ---
                    with st.expander("🏗 Архитектура и гиперпараметры", expanded=True):
                        gsp = hp.get("grid_search_params", {})
                        arch_cols = st.columns(2)
                        with arch_cols[0]:
                            st.markdown("**Алгоритм:** `{}`".format(_m.get("algorithm", "—")))
                            st.markdown("**Признаки ({}):** `{}`".format(len(_m.get("features", [])), ", ".join(_m.get("features", []))))
                            st.markdown("**n_estimators:** {}".format(gsp.get("n_estimators", "—")))
                            st.markdown("**max_depth:** {}".format(gsp.get("max_depth", "—")))
                        with arch_cols[1]:
                            st.markdown("**min_samples_split:** {}".format(gsp.get("min_samples_split", "—")))
                            st.markdown("**min_samples_leaf:** {}".format(gsp.get("min_samples_leaf", "—")))
                            st.markdown("**CV folds:** {}".format(hp.get("cv_folds", "—")))
                            st.markdown("**Scoring:** `{}`".format(hp.get("scoring", "—")))

                    # Пометка о последнем обучении
                    st.caption(
                        f"Последнее обучение: R² ≈ {r2_val}, "
                        f"MAE ≈ {mae_val} п.п.; "
                        f"данных после фильтрации: {after_ml_filter} строк"
                    )
                except json.JSONDecodeError:
                    pass

            strat_labels = {"Ladder": "🪜 Лестница", "Barbell": "🏋️ Гантеля", "Wheel": "🎡 Колесо"}

            # Сравнение стратегий по ML-скору при текущих параметрах боковой панели
            st.markdown("**Сравнение стратегий (текущие ограничения слева)**")
            journal_ml = load_journal()
            existing_ml = get_current_holdings(journal_ml)
            total_val_ml = budget + sum(existing_ml.values())
            comp_rows = []
            for strat_name in ["Ladder", "Barbell", "Wheel"]:
                p_strat = build_portfolio(
                    df=df_raw,
                    strategy=strat_name,
                    budget=budget,
                    min_rating=min_rating,
                    max_years=max_years,
                    max_pct=max_pct / 100.0,
                    existing_holdings=existing_ml,
                    total_portfolio_value=total_val_ml,
                    min_target_yield=target_yield,
                )
                if p_strat.empty:
                    continue
                sc = enrich_with_ml_scores(p_strat, model, scaler, features)
                w = sc["INVESTED"] / sc["INVESTED"].sum()
                comp_rows.append({
                    "Стратегия": strat_labels.get(strat_name, strat_name),
                    "Взвеш. YTM, %": round(float((sc["YTM_PCT"] * w).sum()), 2),
                    "Взвеш. доп. доходн., п.п.": round(float((sc["EXCESS_YTM"] * w).sum()), 2),
                    "Взвеш. ранг в группе": round(float((sc["ML_RANK_IN_GROUP"] * w).sum()), 2),
                    "Бумаг": int(len(sc)),
                })
            if comp_rows:
                st.caption(
                    "Меньший **взвеш. ранг в группе** обычно лучше: в среднем по портфелю бумаги ближе к лидерам "
                    "по доп. доходности среди «похожих» по рейтингу и сроку."
                )
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Нет портфелей для сравнения — ослабьте фильтры или расширьте пул облигаций.")

            st.divider()

            if not portfolio.empty:
                pp = enrich_with_ml_scores(portfolio.copy(), model, scaler, features)
                pp['SIGNAL'] = pp['EXCESS_YTM'].apply(
                    lambda x: '📈 Выше модели' if x > 0.5 else ('📉 Ниже модели' if x < -0.5 else '⚖️ Около модели')
                )
                show_cols = [
                    'SHORTNAME', 'YTM_PCT', 'PREDICTED_YTM', 'EXCESS_YTM',
                    'ML_GROUP', 'ML_RANK_IN_GROUP', 'SIGNAL',
                ]
                display_df = ru(pp[show_cols])
                st.dataframe(
                    display_df.style.format({
                        'Доходность (YTM, %)': '{:.2f}',
                        'Прогноз YTM (%)': '{:.2f}',
                        'Доп. доходность к модели (п.п.)': '{:.2f}',
                        'Ранг в группе (1 — лучше)': '{:.0f}',
                    }),
                    use_container_width=True,
                )
            else:
                st.info("Сначала рассчитайте портфель на вкладке «Портфель», чтобы увидеть ранжирование по бумагам.")

        except FileNotFoundError:
            st.warning("⚠️ Модель не обучена. Запустите из корня проекта: `python src/ml_model.py`")
    else:
        st.warning("⚠️ Файл `ml_model.py` не найден.")

# Подвал
st.divider()
st.caption("🎓 Дипломный проект | Data Science")