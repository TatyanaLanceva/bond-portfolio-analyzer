"""
app.py
Интерактивный дашборд для анализа облигационного портфеля (Финальная версия для защиты)
Запуск: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from src.core import (load_bonds_data, load_journal, get_current_holdings, 
                      save_to_journal, build_portfolio, calculate_metrics, 
                      calculate_full_cashflow, get_top_picks)

# ─────────────────────────────────────────────────────────────────────────────
# НАСТРОЙКИ И ПЕРЕВОДЫ
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Анализатор облигационного портфеля", page_icon="📊", layout="wide")

COL_RU = {
    'Strategy': 'Стратегия', 'Yield_Pct': 'Доходность (YTM, %)',
    'Risk_Years': 'Срок до погашения (лет)', 'Coupon_Yearly': 'Купонный доход в год (₽)',
    'Count': 'Кол-во бумаг', 'Cost': 'Сумма вложений (₽)',
    'SHORTNAME': 'Название выпуска', 'ISIN': 'ISIN', 'RATING': 'Кредитный рейтинг',
    'YEARS_TO_MATURITY': 'Срок (лет)', 'QUANTITY': 'Лотов к покупке',
    'INVESTED': 'Стоимость покупки (₽)', 'YTM_PCT': 'Доходность %',
    'DIRTY_PRICE_RUB': 'Грязная цена (₽)', 'COUPONPERCENT': 'Ставка купона %'
}
STRATEGY_NAMES = {"Ladder": "🪜 Лестница", "Barbell": "🏋️ Штанга", "Wheel": "🎡 Колесо"}

def ru(df, cols=None):
    mapping = {k: v for k, v in COL_RU.items() if k in df.columns}
    return df.rename(columns=mapping)

st.title("📊 Анализатор облигационного портфеля")
st.markdown("*Дипломный проект по Data Science. Автоматизированный подбор, накопление и сравнение стратегий.*")

# ─────────────────────────────────────────────────────────────────────────────
# БОКОВАЯ ПАНЕЛЬ (С ЯВНЫМИ КЛЮЧАМИ ДЛЯ РЕАКТИВНОСТИ)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Параметры моделирования")

@st.cache_data(show_spinner=False)
def get_data(): return load_bonds_data()

try:
    df_raw = get_data()
    st.sidebar.success(f"✅ База данных: {len(df_raw)} выпусков")
except Exception as e:
    st.sidebar.error(f"❌ Ошибка загрузки: {e}")
    st.stop()

# Журнал
journal_df = load_journal()
current_holdings = get_current_holdings(journal_df)
current_total = journal_df['INVESTED'].sum() if not journal_df.empty else 0.0
st.sidebar.metric("📦 Текущий портфель", f"{current_total:,.0f} ₽")

with st.sidebar.expander("📖 Что означают параметры?"):
    st.markdown("""
    - **Бюджет**: Новая сумма для инвестирования.
    - **Мин. рейтинг**: Фильтр надежности (AAA — высший, BBB- — порог инвест. уровня).
    - **Макс. срок**: Ограничение по времени возврата денег.
    - **Макс. доля (%)**: Лимит концентрации. Считается от `(Текущий портфель + Новый бюджет)`.
    - **Налог**: 0% (ИИС) или 13% (обычный счет). Влияет на чистый купонный поток.
    """)

# Виджеты с явными ключами
budget = st.sidebar.number_input("💰 Новый бюджет (₽)", min_value=10000, value=40000, step=10000, key="inp_budget")
min_rating = st.sidebar.selectbox("🛡 Мин. рейтинг", ["AAA","AA+","AA","A+","A","BBB+","BBB","BBB-"], index=4, key="sel_rating")
max_years = st.sidebar.slider("📅 Макс. срок (лет)", 1, 15, 10, key="sld_years")
max_pct = st.sidebar.slider("🔒 Макс. доля одной бумаги (%)", 1, 50, 15, key="sld_pct") / 100.0

tax_choice = st.sidebar.radio("💸 Налог на купоны", ["0% (ИИС)", "13% (Брокерский счет)"], key="rad_tax")
tax_rate = 0.0 if tax_choice.startswith("0") else 0.13

strategy_raw = st.sidebar.selectbox("📈 Стратегия подбора", list(STRATEGY_NAMES.values()), index=0, key="sel_strategy")
STRATEGY_MAP = {v: k for k, v in STRATEGY_NAMES.items()}
strategy = STRATEGY_MAP[strategy_raw]

# ─────────────────────────────────────────────────────────────────────────────
# РАСЧЁТЫ
# ─────────────────────────────────────────────────────────────────────────────
total_portfolio_val = current_total + budget
strategies_to_run = [strategy] if strategy != "Сравнить все" else ["Ladder", "Barbell", "Wheel"]

portfolio_cache = {}
metrics_cache = {}

for s in strategies_to_run:
    port = build_portfolio(df_raw, s, budget, min_rating, max_years, max_pct, 
                           current_holdings, total_portfolio_val)
    portfolio_cache[s] = port
    metrics_cache[s] = calculate_metrics(port)

# ─────────────────────────────────────────────────────────────────────────────
# ВКЛАДКИ
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Сравнение стратегий", "💰 Купонный поток", "📋 Список покупок и алгоритм", "📂 Журнал портфеля"])

st.caption(f"🔄 Активный режим: `{STRATEGY_NAMES.get(strategy, strategy)}` | Бюджет: `{budget:,} ₽`")

with tab1:
    st.subheader("Сравнение эффективности стратегий")
    
    # ✅ НОВЫЙ БЛОК: Справочное описание стратегий
    with st.expander("📖 Суть инвестиционных стратегий (справка для непрофессионалов)"):
        st.markdown("""
        **🪜 Лестница (Ladder)**  
        - *Идея:* Покупать облигации с разными сроками погашения (например, на 1, 2, 3, 4 и 5 лет).  
        - *Как работает:* Каждый год одна бумага гасится, возвращая номинал. Эти деньги реинвестируются в новую долгосрочную облигацию.  
        - *Плюсы:* Регулярный доступ к деньгам, защита от резких изменений процентных ставок, стабильный денежный поток.  
        - *Для кого:* Для консервативных инвесторов, которым важна предсказуемость и ликвидность.

        **🏋️ Штанга (Barbell)**  
        - *Идея:* Сочетать в портфеле только короткие (до 2 лет) и длинные (от 4 лет) облигации, исключая средние сроки.  
        - *Как работает:* Короткая часть дает ликвидность и защиту. Длинная часть фиксирует высокую доходность на долгий срок.  
        - *Плюсы:* Гибкость: можно быстро перестроить портфель при изменении экономики, получая при этом повышенный доход.  
        - *Для кого:* Для умеренных инвесторов, готовых к небольшому риску ради баланса дохода и маневренности.

        **🎡 Колесо (Wheel)**  
        - *Идея:* Циклическое распределение бюджета по разным эмитентам и срокам в несколько "раундов".  
        - *Как работает:* Бюджет делится на равные части. В каждом раунде алгоритм выбирает лучшую доступную бумагу, постепенно формируя широко диверсифицированный портфель.  
        - *Плюсы:* Максимальное покрытие рынка, снижение зависимости от одного эмитента, простая автоматизация.  
        - *Для кого:* Для инвесторов, стремящихся к максимальной диверсификации и системному подходу без попыток угадать лучшие сроки.
        """)

    with st.expander("📖 Как читать графики и метрики?"):
        st.markdown("- **Доходность (YTM %)**: Годовая доходность с учётом всех купонов и разницы между ценой покупки и номиналом.\n- **Срок (лет)**: Среднее время возврата капитала. Используется как прокси-метрика риска (чем дольше срок, тем чувствительнее цена к изменению ключевой ставки).\n- **Купонный доход**: Прогнозируемая сумма процентов, которую вы будете получать ежегодно.")
        
    df_m = pd.DataFrame(metrics_cache).T.reset_index().rename(columns={'index':'Strategy'})
    df_m['Strategy'] = df_m['Strategy'].map(lambda x: STRATEGY_NAMES.get(x, x))
    
    c1, c2 = st.columns(2)
    with c1:
        f1 = px.bar(df_m, x='Strategy', y='Yield_Pct', color='Strategy', title="Средневзвешенная доходность", text='Yield_Pct')
        f1.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(f1, width="stretch")
    with c2:
        f2 = px.scatter(df_m, x='Risk_Years', y='Yield_Pct', size='Count', color='Strategy',
                        hover_data=['Coupon_Yearly','Cost'], title="Матрица Риск / Доходность")
        st.plotly_chart(f2, width="stretch")
        
    st.dataframe(ru(df_m).style.format({'Доходность (YTM, %)':'{:.2f}%', 'Срок до погашения (лет)':'{:.2f}', 'Купонный доход в год (₽)':'{:,.0f}'}), width="stretch")

with tab2:
    st.subheader("Прогноз денежных потоков до погашения")
    with st.expander("📖 Что такое Waterfall-график?"):
        st.markdown("- **Синие столбцы**: Чистые купоны (после вычета налога).\n- **Зелёные столбцы**: Возврат номинала облигаций при их погашении.\n- **Оранжевая линия**: Накопленный доход. Показывает, как растёт ваша прибыль с течением времени.\n- Расчёт ведётся до даты погашения самой дальней бумаги в портфеле.")
        
    tgt = strategy if strategy != "Сравнить все" else "Ladder"
    port = portfolio_cache.get(tgt)
    if port is not None and not port.empty:
        cf = calculate_full_cashflow(port, tax_rate)
        if not cf.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cf['YEAR'], y=cf['COUPON'], name='Купоны (нетто)', marker_color='#2196F3'))
            fig.add_trace(go.Bar(x=cf['YEAR'], y=cf['PRINCIPAL'], name='Возврат номинала', marker_color='#4CAF50', base=cf['COUPON']))
            fig.add_trace(go.Scatter(x=cf['YEAR'], y=cf['CUMULATIVE'], name='Накопленный доход', line=dict(color='#FF9800', width=3)))
            fig.update_layout(title=f"💰 Денежный поток: {STRATEGY_NAMES.get(tgt)}", barmode='stack')
            st.plotly_chart(fig, width="stretch")
            st.info(f"📊 Итого купонов: {cf['COUPON'].sum():,.0f} ₽ | Номинал: {cf['PRINCIPAL'].sum():,.0f} ₽")
    else:
        st.warning("Портфель пуст. Смягчите фильтры или увеличьте бюджет.")

with tab3:
    st.subheader("📋 Точный список покупок")
    with st.expander("🤖 Как работает жадный алгоритм подбора?"):
        st.markdown("""
        1. **Фильтрация**: Отсеиваются бумаги с низким рейтингом или долгим сроком.\n
        2. **Ограничение доли**: `Лимит = (Текущий портфель + Бюджет) × Макс.% - Уже_вложено`. Не даёт перегрузить одну бумагу.\n
        3. **Приоритет доходности**: Кандидаты сортируются по YTM (сверху вниз).\n
        4. **Целевое распределение**: Лестница → по годам, Штанга → 50/50 короткое/длинное, Колесо → циклично.\n
        5. **Жадное заполнение остатка**: Если >10% бюджета свободно, алгоритм покупает следующие лучшие доступные лоты, пока не упрётся в лимит или цену.\n
        *Итог: Максимально возможное вложение при заданных ограничениях.*
        """)
        
    port_show = portfolio_cache.get(strategy)
    if port_show is not None and not port_show.empty:
        cols_s = ['SHORTNAME','ISIN','RATING','YEARS_TO_MATURITY','YTM_PCT','QUANTITY','INVESTED']
        st.dataframe(ru(port_show[cols_s]).style.format({'Доходность %':'{:.2f}%', 'Стоимость покупки (₽)':'{:,.0f}'}), width="stretch")
        
        cost = port_show['INVESTED'].sum()
        st.metric("Вложено в портфель", f"{cost:,.0f} ₽", delta=f"Остаток бюджета: {(budget-cost):,.0f} ₽")
        
        if st.button("💾 Сохранить новые покупки в журнал", type="primary"):
            journal_df = save_to_journal(journal_df, port_show, strategy)
            st.rerun()
    else:
        st.warning("Портфель пуст. Увеличьте бюджет или смягчите фильтры.")

with tab4:
    st.subheader("📂 Журнал портфеля")
    st.markdown("Здесь хранится история всех подтверждённых покупок. При расчёте лимитов система учитывает эти данные, чтобы не нарушить диверсификацию.")
    if not journal_df.empty:
        st.dataframe(journal_df.style.format({'INVESTED':'{:,.0f}', 'PRICE':'{:,.2f}'}), width="stretch")
        st.metric("Общая стоимость портфеля", f"{journal_df['INVESTED'].sum():,.0f} ₽")
    else:
        st.info("Журнал пуст. Сформируйте и сохраните портфель во вкладке «Список покупок».")
        
    if st.button("🗑 Очистить журнал (для нового теста)"):
        Path(JOURNAL_PATH).unlink(missing_ok=True)
        st.success("Журнал очищен. Перезагрузите страницу (F5).")
        st.stop()

    st.markdown("---")
    st.caption("Дипломный проект | Data Science | 2026")