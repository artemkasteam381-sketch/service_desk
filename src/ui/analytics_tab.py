from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
from html import escape
import logging
import os
from pathlib import Path
from time import perf_counter

import pandas as pd
import streamlit as st

from src.analytics.charts import (
    make_breakdown_chart,
    make_problem_summary_chart,
    make_sla_risk_chart,
    make_timeseries_chart,
)
from src.analytics.data import (
    AnalyticsDataError,
    apply_filters,
    export_view,
    get_filter_options,
    load_analytics_data,
)
from src.analytics.metrics import (
    calculate_breakdown,
    calculate_kpis,
    calculate_sla_risk,
    calculate_timeseries,
    get_problem_cases,
    summarize_problem_cases,
)


LOGGER = logging.getLogger(__name__)
PROFILE_ANALYTICS = os.getenv("ANALYTICS_PROFILE", "0") == "1"
if PROFILE_ANALYTICS:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.setLevel(logging.INFO)


@contextmanager
def _profile(name: str):
    if not PROFILE_ANALYTICS:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        LOGGER.info("%s: %.1f ms", name, (perf_counter() - started) * 1000)


ANALYTICS_CSS = r"""
<style>
/* Content width: readable on large screens, fluid on laptops. */
.block-container {
    max-width: 1240px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}

.analytics-heading-row {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;
    margin: 2px 0 16px 0;
}
.analytics-heading h1 {
    font-size: 2rem;
    line-height: 1.08;
    margin: 0;
    font-weight: 740;
    letter-spacing: -0.02em;
}
.analytics-heading p {
    margin: 7px 0 0 0;
    color: #8E9AAF;
    font-size: 0.92rem;
}

/* The four inner analytics modes. */
.st-key-analytics_section [data-testid="stSegmentedControl"] {
    max-width: 650px;
}
.st-key-analytics_section button {
    min-height: 40px;
}

/* Compact filter card. */
.st-key-analytics_filters {
    background: linear-gradient(180deg, rgba(24,32,45,0.72), rgba(14,20,30,0.72));
    border: 1px solid rgba(132, 148, 171, 0.23);
    border-radius: 12px;
    padding: 13px 14px 10px 14px;
    margin: 10px 0 16px 0;
}
.st-key-analytics_filters [data-testid="stWidgetLabel"] p {
    color: #8E9AAF;
    font-size: 0.79rem;
}
.filter-count {
    text-align: right;
    color: #9AA6B9;
    margin: 1px 3px 0 0;
    font-size: 0.86rem;
}
.filter-count strong { color: #E8EDF5; }

/* KPI cards. */
.kpi-card {
    height: 92px;
    border: 1px solid rgba(132, 148, 171, 0.20);
    border-radius: 12px;
    background: linear-gradient(145deg, rgba(23,31,44,0.93), rgba(15,22,32,0.93));
    padding: 15px 17px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}
.kpi-label {
    color: #A4AFC0;
    font-size: 0.82rem;
    line-height: 1.15;
    margin-bottom: 6px;
}
.kpi-value {
    color: #F3F6FA;
    font-size: 1.62rem;
    font-weight: 720;
    letter-spacing: -0.02em;
    line-height: 1;
}
.kpi-value .accent-red { color: #FF6464; font-size: 1.06rem; font-weight: 700; }
.kpi-value .accent-orange { color: #FFAA4D; font-size: 1.06rem; font-weight: 700; }
.kpi-info {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 15px;
    height: 15px;
    border: 1px solid #738095;
    border-radius: 50%;
    font-size: 10px;
    color: #8E9AAF;
    margin-left: 4px;
}

/* Chart/table cards. */
.st-key-analytics_chart_card,
.st-key-breakdown_chart_card,
.st-key-breakdown_table_card,
.st-key-problem_chart_card,
.st-key-problem_table_card {
    border: 1px solid rgba(132, 148, 171, 0.20);
    border-radius: 12px;
    background: linear-gradient(145deg, rgba(22,30,42,0.83), rgba(13,19,29,0.83));
    padding: 10px 12px 6px 12px;
}
.card-title {
    font-size: 1.04rem;
    font-weight: 700;
    color: #F0F3F8;
    margin: 2px 0 6px 0;
}
.card-subtitle {
    color: #8E9AAF;
    font-size: 0.81rem;
    margin-top: -2px;
    margin-bottom: 7px;
}

.method-note {
    color: #8E9AAF;
    font-size: 0.82rem;
    line-height: 1.45;
}

/* Reduce vertical noise of default Streamlit separators. */
hr { margin: 0.65rem 0 0.9rem 0; }

@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .analytics-heading h1 { font-size: 1.7rem; }
    .kpi-card { height: 86px; }
}
</style>
"""


SECTION_OPTIONS = ["Обзор", "Динамика", "Разрезы", "Проблемные зоны"]
DYNAMIC_METRICS = {
    "Обращения": "requests",
    "Просрочки": "overdue_rate",
    "Время обработки": "median_duration",
}
GRANULARITIES = {"День": "day", "Неделя": "week", "Месяц": "month"}
BREAKDOWN_DIMENSIONS = {
    "Услуги": "service",
    "Виды обращений": "request_type",
    "Приоритеты": "priority",
    "Линии поддержки": "final_line",
}
BREAKDOWN_METRICS = {
    "Нагрузка": "requests",
    "Доля нагрузки": "load_share",
    "Просрочки": "overdue_rate",
    "Медиана времени": "median_duration",
    "P90": "p90_duration",
    "Уточнения": "median_clarifications",
    "Несколько линий": "multi_line_rate",
}
PROBLEM_MODES = {
    "Риск SLA": "sla_risk",
    "Долгая обработка": "long_duration",
    "Много уточнений": "clarifications",
    "Смена линии": "multi_line",
    "Просроченные": "overdue",
}


def _fmt_int(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "Недоступно"
    return f"{int(round(value)):,}".replace(",", " ")


def _fmt_float(value: float | None, suffix: str = "", digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "Недоступно"
    text = f"{value:.{digits}f}".replace(".", ",")
    return f"{text}{suffix}"


def _fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Недоступно"
    return f"{value * 100:.1f}%".replace(".", ",")


def _kpi_card(label: str, main: str, *, secondary: str | None = None, accent: str = "red") -> None:
    secondary_html = ""
    if secondary:
        css_class = "accent-orange" if accent == "orange" else "accent-red"
        secondary_html = f' <span class="{css_class}">· {escape(secondary)}</span>'

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{escape(label)} <span class="kpi-info" title="Показатель рассчитан после применения всех выбранных фильтров.">i</span></div>
            <div class="kpi-value">{escape(main)}{secondary_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _ensure_filter_state(options: dict) -> None:
    defaults = {
        "analytics_period": (options["min_date"], options["max_date"]),
        "analytics_service": "Все услуги",
        "analytics_request_type": "Все виды",
        "analytics_priority": "Все приоритеты",
        "analytics_line": "Все линии",
        "analytics_overdue": "Все",
        "analytics_status": "Все состояния",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Source data can change while a session is alive. Repair stale selections.
    valid = {
        "analytics_service": {"Все услуги", *options["services"]},
        "analytics_request_type": {"Все виды", *options["request_types"]},
        "analytics_priority": {"Все приоритеты", *options["priorities"]},
        "analytics_line": {"Все линии", *options["lines"]},
        "analytics_status": {"Все состояния", *options["statuses"]},
    }
    for key, allowed in valid.items():
        if st.session_state.get(key) not in allowed:
            st.session_state[key] = defaults[key]


def _reset_filters(options: dict) -> None:
    # Remove widget-backed filter keys so Streamlit recreates them from defaults
    # on the next rerun. This avoids stale filter state when switching sections.
    keys = [
        "analytics_period",
        "analytics_service",
        "analytics_request_type",
        "analytics_priority",
        "analytics_line",
        "analytics_overdue",
        "analytics_status",
    ]
    for key in keys:
        st.session_state.pop(key, None)


def _render_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    options = get_filter_options(df)
    _ensure_filter_state(options)

    with st.container(key="analytics_filters"):
        cols = st.columns([1.45, 1.20, 1.20, 0.85, 0.72], vertical_alignment="bottom")

        with cols[0]:
            st.date_input(
                "Период",
                min_value=options["min_date"],
                max_value=options["max_date"],
                key="analytics_period",
                format="DD.MM.YYYY",
            )

        with cols[1]:
            st.selectbox(
                "Услуга",
                ["Все услуги", *options["services"]],
                key="analytics_service",
            )

        with cols[2]:
            st.selectbox(
                "Вид обращения",
                ["Все виды", *options["request_types"]],
                key="analytics_request_type",
            )

        with cols[3]:
            with st.popover("Ещё фильтры", icon=":material/tune:", width="stretch"):
                st.selectbox(
                    "Приоритет",
                    ["Все приоритеты", *options["priorities"]],
                    key="analytics_priority",
                )
                st.selectbox(
                    "Линия поддержки",
                    ["Все линии", *options["lines"]],
                    key="analytics_line",
                    help="Фильтр использует итоговую группу, решившую обращение.",
                )
                st.selectbox(
                    "Статус SLA",
                    ["Все", "Просроченные", "Не просроченные"],
                    key="analytics_overdue",
                )
                if len(options["statuses"]) > 1:
                    st.selectbox(
                        "Состояние обращения",
                        ["Все состояния", *options["statuses"]],
                        key="analytics_status",
                    )

        with cols[4]:
            st.button(
                "Сбросить",
                icon=":material/refresh:",
                width="stretch",
                on_click=_reset_filters,
                args=(options,),
            )

        period = st.session_state["analytics_period"]
        if isinstance(period, (tuple, list)):
            if len(period) >= 2:
                start_date, end_date = period[0], period[1]
            elif len(period) == 1:
                start_date = end_date = period[0]
            else:
                start_date, end_date = options["min_date"], options["max_date"]
        else:
            # Streamlit can temporarily expose a single date while a range is edited.
            start_date = end_date = period

        filter_state = {
            "start_date": start_date,
            "end_date": end_date,
            "service": st.session_state["analytics_service"],
            "request_type": st.session_state["analytics_request_type"],
            "priority": st.session_state["analytics_priority"],
            "final_line": st.session_state["analytics_line"],
            "overdue_status": st.session_state["analytics_overdue"],
            "status": st.session_state["analytics_status"],
        }

        filtered = apply_filters(df, **filter_state)
        st.markdown(
            f'<div class="filter-count">В выборке: <strong>{len(filtered):,}</strong> обращений</div>'.replace(",", " "),
            unsafe_allow_html=True,
        )

    return filtered, filter_state


def _default_granularity(start_date, end_date) -> str:
    days = max(1, (end_date - start_date).days + 1)
    if days <= 45:
        return "День"
    if days <= 180:
        return "Неделя"
    return "Месяц"


def _render_overview(filtered: pd.DataFrame, filter_state: dict) -> None:
    with _profile("analytics.kpis"):
        kpis = calculate_kpis(filtered)

    row1 = st.columns(3, gap="small")
    with row1[0]:
        _kpi_card("Обращения", _fmt_int(kpis.requests))
    with row1[1]:
        _kpi_card(
            "Просроченные",
            _fmt_int(kpis.overdue_count),
            secondary=_fmt_pct(kpis.overdue_rate),
            accent="red",
        )
    with row1[2]:
        _kpi_card("Медиана обработки", _fmt_float(kpis.median_duration, " ч"))

    st.space("small")
    row2 = st.columns(3, gap="small")
    with row2[0]:
        _kpi_card("P90 обработки", _fmt_float(kpis.p90_duration, " ч"))
    with row2[1]:
        _kpi_card(
            "Смена линии",
            _fmt_int(kpis.multi_line_count),
            secondary=_fmt_pct(kpis.multi_line_rate),
            accent="orange",
        )
    with row2[2]:
        _kpi_card("Медиана уточнений", _fmt_float(kpis.median_clarifications, digits=0))

    st.space("small")
    with st.container(key="analytics_chart_card"):
        title_col, control_col = st.columns([0.55, 0.45], vertical_alignment="center")
        with title_col:
            st.markdown('<div class="card-title">Общая динамика</div>', unsafe_allow_html=True)
        with control_col:
            selected = st.segmented_control(
                "Показатель обзора",
                list(DYNAMIC_METRICS),
                default="Обращения",
                key="overview_metric",
                label_visibility="collapsed",
                width="stretch",
            )

        if filtered.empty:
            st.info("По выбранным фильтрам обращений не найдено.")
            return

        granularity_label = _default_granularity(filter_state["start_date"], filter_state["end_date"])
        metric = DYNAMIC_METRICS[selected or "Обращения"]
        with _profile("analytics.overview_timeseries"):
            series = calculate_timeseries(
                filtered,
                metric=metric,
                granularity=GRANULARITIES[granularity_label],
            )
            fig = make_timeseries_chart(series, metric=metric, height=325)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _render_dynamics(df: pd.DataFrame, filtered: pd.DataFrame, filter_state: dict) -> None:
    c1, c2, c3 = st.columns([1.65, 1.0, 0.9], vertical_alignment="bottom")
    with c1:
        metric_label = st.segmented_control(
            "Показатель",
            list(DYNAMIC_METRICS),
            default="Обращения",
            key="dynamics_metric",
            width="stretch",
        )
    with c2:
        granularity_label = st.segmented_control(
            "Группировка",
            list(GRANULARITIES),
            default=_default_granularity(filter_state["start_date"], filter_state["end_date"]),
            key="dynamics_granularity",
            width="stretch",
        )
    with c3:
        compare = st.toggle("Сравнить период", key="dynamics_compare")

    metric = DYNAMIC_METRICS[metric_label or "Обращения"]
    granularity = GRANULARITIES[granularity_label or "Месяц"]

    with st.container(key="analytics_chart_card"):
        st.markdown('<div class="card-title">Динамика</div>', unsafe_allow_html=True)
        if filtered.empty:
            st.info("По выбранным фильтрам обращений не найдено.")
            return

        with _profile("analytics.dynamics_current"):
            current = calculate_timeseries(filtered, metric=metric, granularity=granularity)

        previous_series = None
        if compare:
            current_start = filter_state["start_date"]
            current_end = filter_state["end_date"]
            inclusive_days = (current_end - current_start).days + 1
            previous_end = current_start - timedelta(days=1)
            previous_start = previous_end - timedelta(days=inclusive_days - 1)

            previous_filters = dict(filter_state)
            previous_filters["start_date"] = previous_start
            previous_filters["end_date"] = previous_end
            previous_df = apply_filters(df, **previous_filters)
            previous_series = calculate_timeseries(
                previous_df,
                metric=metric,
                granularity=granularity,
            )
            # Shift to the current x-axis so equal-length periods overlay cleanly.
            if not previous_series.empty:
                previous_series = previous_series.copy()
                previous_series["period"] = previous_series["period"] + pd.Timedelta(days=inclusive_days)

            st.caption(
                f"Сравнение с равным предыдущим интервалом: "
                f"{previous_start:%d.%m.%Y}–{previous_end:%d.%m.%Y}."
            )

        with _profile("analytics.dynamics_chart"):
            fig = make_timeseries_chart(
                current,
                metric=metric,
                previous=previous_series,
                height=430,
            )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _render_breakdowns(filtered: pd.DataFrame) -> None:
    c1, c2 = st.columns([1.35, 0.85], vertical_alignment="bottom")
    with c1:
        dimension_label = st.segmented_control(
            "Разрез",
            list(BREAKDOWN_DIMENSIONS),
            default="Услуги",
            key="breakdown_dimension",
            width="stretch",
        )
    with c2:
        metric_label = st.selectbox(
            "Показатель на графике",
            list(BREAKDOWN_METRICS),
            key="breakdown_metric",
        )

    dimension = BREAKDOWN_DIMENSIONS[dimension_label or "Услуги"]
    metric = BREAKDOWN_METRICS[metric_label]

    if filtered.empty:
        st.info("По выбранным фильтрам обращений не найдено.")
        return

    with _profile("analytics.breakdown"):
        breakdown = calculate_breakdown(filtered, dimension=dimension)

    left, right = st.columns([0.47, 0.53], gap="medium")
    with left:
        with st.container(key="breakdown_chart_card"):
            st.markdown(
                f'<div class="card-title">{escape(metric_label)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="card-subtitle">Топ групп по выбранному показателю</div>',
                unsafe_allow_html=True,
            )
            fig = make_breakdown_chart(breakdown, metric=metric, top_n=12, height=440)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        with st.container(key="breakdown_table_card"):
            st.markdown('<div class="card-title">Показатели по группам</div>', unsafe_allow_html=True)
            table = breakdown.rename(
                columns={
                    "group": "Группа",
                    "requests": "N",
                    "load_share": "Доля нагрузки",
                    "overdue_count": "Просрочено",
                    "overdue_rate": "Просрочено, %",
                    "median_duration": "Медиана, ч",
                    "p90_duration": "P90, ч",
                    "median_clarifications": "Уточнения",
                    "multi_line_rate": "Неск. линий, %",
                }
            ).copy()
            for col in ("Доля нагрузки", "Просрочено, %", "Неск. линий, %"):
                table[col] = table[col] * 100.0

            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                height=430,
                column_config={
                    "Доля нагрузки": st.column_config.NumberColumn(format="%.1f%%"),
                    "Просрочено, %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Медиана, ч": st.column_config.NumberColumn(format="%.1f"),
                    "P90, ч": st.column_config.NumberColumn(format="%.1f"),
                    "Уточнения": st.column_config.NumberColumn(format="%.1f"),
                    "Неск. линий, %": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )


def _problem_table(cases: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticket_id",
        "created_at",
        "service",
        "request_type",
        "priority",
        "final_line",
        "duration_hours",
        "clarifications",
        "line_count",
        "is_overdue",
        "source_url",
    ]
    existing = [c for c in columns if c in cases.columns]
    if "source_url" in existing and cases["source_url"].isna().all():
        existing.remove("source_url")
    out = cases[existing].copy()
    return out.rename(
        columns={
            "ticket_id": "ID",
            "created_at": "Дата",
            "service": "Услуга",
            "request_type": "Вид обращения",
            "priority": "Приоритет",
            "final_line": "Линия",
            "duration_hours": "Время, ч",
            "clarifications": "Уточнения",
            "line_count": "Линий",
            "is_overdue": "Просрочено",
            "source_url": "Ссылка",
        }
    )


def _render_problem_zones(filtered: pd.DataFrame) -> None:
    mode_label = st.segmented_control(
        "Проблемная зона",
        list(PROBLEM_MODES),
        default="Риск SLA",
        key="problem_mode",
        width="stretch",
    )
    mode = PROBLEM_MODES[mode_label or "Риск SLA"]

    if filtered.empty:
        st.info("По выбранным фильтрам обращений не найдено.")
        return

    if mode == "sla_risk":
        dimension_label = st.segmented_control(
            "Группировать риск по",
            ["Виды обращений", "Линии поддержки"],
            default="Виды обращений",
            key="sla_risk_dimension",
        )
        dimension = "request_type" if dimension_label == "Виды обращений" else "final_line"
        risk = calculate_sla_risk(
            filtered,
            dimension=dimension,
            alpha=10.0,
            min_group_size=5,
            uplift_threshold=0.03,
        )

        left, right = st.columns([0.48, 0.52], gap="medium")
        with left:
            with st.container(key="problem_chart_card"):
                st.markdown('<div class="card-title">Исторический риск SLA</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="card-subtitle">Красным отмечены группы, прошедшие аналитический порог риска</div>',
                    unsafe_allow_html=True,
                )
                fig = make_sla_risk_chart(risk, top_n=12, height=440)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with right:
            with st.container(key="problem_table_card"):
                st.markdown('<div class="card-title">Группы риска</div>', unsafe_allow_html=True)
                table = risk[
                    ["group", "requests", "overdue_count", "overdue_rate", "smoothed_risk", "risk_delta", "elevated_risk"]
                ].copy()
                table = table.rename(
                    columns={
                        "group": "Группа",
                        "requests": "N",
                        "overdue_count": "Просрочено",
                        "overdue_rate": "Историческая доля",
                        "smoothed_risk": "Сглаженная доля",
                        "risk_delta": "Выше общей",
                        "elevated_risk": "Повышенный риск",
                    }
                )
                for col in ("Историческая доля", "Сглаженная доля", "Выше общей"):
                    table[col] = table[col] * 100.0
                st.dataframe(
                    table,
                    hide_index=True,
                    width="stretch",
                    height=430,
                    column_config={
                        "Историческая доля": st.column_config.NumberColumn(format="%.1f%%"),
                        "Сглаженная доля": st.column_config.NumberColumn(format="%.1f%%"),
                        "Выше общей": st.column_config.NumberColumn(format="%+.1f п.п."),
                        "Повышенный риск": st.column_config.CheckboxColumn(),
                    },
                )
        st.caption(
            "Правило риска: сглаженная историческая доля просрочек, минимум 5 обращений; "
            "метка «повышенный риск» — при превышении общей доли минимум на 3 п.п. Это не прогноз."
        )
        return

    cases, meta = get_problem_cases(filtered, mode=mode, limit=200)
    summary = summarize_problem_cases(cases, dimension="request_type")

    if mode == "long_duration" and meta.get("threshold") is not None:
        st.caption(f"Показаны обращения не ниже P90 выбранной выборки: {meta['threshold']:.1f} ч.")
    elif mode == "clarifications" and meta.get("threshold") is not None:
        st.caption(
            f"Показаны верхние 10% по числу уточнений; текущий порог: ≥ {meta['threshold']:.0f}."
        )

    left, right = st.columns([0.40, 0.60], gap="medium")
    with left:
        with st.container(key="problem_chart_card"):
            st.markdown('<div class="card-title">Где сосредоточены случаи</div>', unsafe_allow_html=True)
            if summary.empty:
                st.info("Подходящих обращений в текущей выборке нет.")
            else:
                fig = make_problem_summary_chart(summary, height=430)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        with st.container(key="problem_table_card"):
            total_cases = int(meta.get("total_cases") or len(cases))
            suffix = "" if total_cases <= len(cases) else f" · показаны первые {len(cases):,}".replace(",", " ")
            st.markdown(
                (f'<div class="card-title">Конкретные обращения · {total_cases:,}{suffix}</div>').replace(",", " "),
                unsafe_allow_html=True,
            )
            table = _problem_table(cases)
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                height=430,
                column_config={
                    "Дата": st.column_config.DatetimeColumn(format="DD.MM.YYYY HH:mm"),
                    "Время, ч": st.column_config.NumberColumn(format="%.1f"),
                    "Уточнения": st.column_config.NumberColumn(format="%.0f"),
                    "Ссылка": st.column_config.LinkColumn(display_text="Открыть"),
                },
            )


def _render_methodology(df: pd.DataFrame) -> None:
    historical_share = float((df["overdue_basis"] == "historical_flag").mean()) if len(df) else 0.0
    sla_duration_share = float((df["duration_basis"] == "sla_duration").mean()) if len(df) else 0.0

    with st.popover("ⓘ Методика расчёта"):
        st.markdown(
            f"""
**Просрочка.** В первую очередь используется историческое поле просрочки. В текущем наборе оно доступно примерно для {historical_share:.0%} записей. Если метка отсутствует, используется безопасный fallback по длительности/SLA или сроку выполнения.

**Длительность обработки.** Если заполнена фактическая длительность выполнения по SLA, используется она; иначе — календарное время от регистрации до выполнения. Доля строк с SLA-длительностью в текущем наборе — около {sla_duration_share:.0%}. Это не обязательно чистое активное время специалиста.

**Смена линии.** Обращение считается прошедшим несколько линий, если по двум и более линиям есть ненулевое суммарное время реакции/работы либо линия указана как итоговая. Показатель описывает доступные агрегированные поля и не восстанавливает точную последовательность передач.

**Риск SLA.** Это сглаженная историческая доля просрочек в группе. Правило по умолчанию: минимум 5 обращений и превышение общей доли минимум на 3 процентных пункта. Показатель не является прогнозом будущей просрочки.
            """
        )


def _export_bytes(filtered: pd.DataFrame) -> bytes:
    return export_view(filtered).to_csv(index=False).encode("utf-8-sig")


def render_analytics_tab(data_path: str | Path) -> None:
    """Render the complete analytics page from the approved mockup structure."""
    st.markdown(ANALYTICS_CSS, unsafe_allow_html=True)

    with _profile("analytics.load"):
        try:
            df = load_analytics_data(data_path)
        except (FileNotFoundError, AnalyticsDataError, ValueError) as exc:
            st.error(f"Не удалось подготовить данные аналитики: {exc}")
            return

    title_col, export_col = st.columns([0.82, 0.18], vertical_alignment="center")
    with title_col:
        st.markdown(
            """
            <div class="analytics-heading">
                <h1>Аналитика обработки</h1>
                <p>Исторические данные · показатели рассчитаны по текущей выборке</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section control before filters keeps page hierarchy close to the mockup.
    with st.container(key="analytics_section"):
        section = st.segmented_control(
            "Раздел аналитики",
            SECTION_OPTIONS,
            default="Обзор",
            key="analytics_section_value",
            label_visibility="collapsed",
            width="stretch",
        )

    with _profile("analytics.filters"):
        filtered, filter_state = _render_filters(df)

    with export_col:
        st.download_button(
            "Экспорт",
            data=_export_bytes(filtered),
            file_name="analytics_filtered.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
            help="Скачать текущую отфильтрованную выборку.",
        )

    selected = section or "Обзор"
    if selected == "Обзор":
        _render_overview(filtered, filter_state)
    elif selected == "Динамика":
        _render_dynamics(df, filtered, filter_state)
    elif selected == "Разрезы":
        _render_breakdowns(filtered)
    else:
        _render_problem_zones(filtered)

    st.space("small")
    method_col, _ = st.columns([0.32, 0.68])
    with method_col:
        _render_methodology(df)
