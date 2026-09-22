from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
from html import escape
import logging
import os
from pathlib import Path
from time import perf_counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.charts import (
    make_breakdown_chart,
    make_problem_summary_chart,
    make_sla_risk_chart,
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


/* Compact cards for overview small-multiples and dynamics insights. */
.mini-chart-title {
    color: #F0F3F8;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 2px;
}
.mini-chart-value {
    color: #F7F9FC;
    font-size: 1.45rem;
    font-weight: 760;
    line-height: 1.1;
    margin-bottom: 2px;
}
.mini-chart-note {
    color: #8E9AAF;
    font-size: 0.76rem;
    margin-bottom: 3px;
}
.insight-stack-title {
    color: #F0F3F8;
    font-size: 1rem;
    font-weight: 720;
    margin: 2px 0 8px 0;
}
.insight-card {
    border: 1px solid rgba(132, 148, 171, 0.20);
    border-radius: 11px;
    background: linear-gradient(145deg, rgba(23,31,44,0.93), rgba(15,22,32,0.93));
    padding: 12px 13px;
    margin-bottom: 9px;
}
.insight-label {
    color: #8E9AAF;
    font-size: 0.75rem;
    margin-bottom: 3px;
}
.insight-value {
    color: #F4F7FB;
    font-size: 1.22rem;
    font-weight: 750;
    line-height: 1.1;
}
.insight-note {
    color: #9AA6B9;
    font-size: 0.72rem;
    margin-top: 4px;
    line-height: 1.3;
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
            <div class="kpi-label">{escape(label)} <span class="kpi-info">i</span></div>
            <div class="kpi-value">{escape(main)}{secondary_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



# Technical placeholders found in source systems. They are never useful as
# user-facing categories; the UI exposes a single readable value instead.
_PLACEHOLDER_TEXT = {
    "", "-", "--", "---", "--)", "---)",
    "nan", "none", "null", "не определена", "не определено",
    "не указан", "не указано", "не указана",
}
_CATEGORY_COLUMNS = ("service", "request_type", "priority", "status", "final_line")
_RESAMPLE_FREQ = {"day": "D", "week": "W-MON", "month": "MS"}
_RU_MONTHS = {
    1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр", 5: "Май", 6: "Июн",
    7: "Июл", 8: "Авг", 9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек",
}


def _is_placeholder(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    if text.casefold() in _PLACEHOLDER_TEXT:
        return True
    # Catch technical values such as --), ---), ––) without touching real names.
    return bool(__import__("re").fullmatch(r"[-–—]+\)?", text))


def _normalize_display_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse technical placeholder categories to one user-facing value."""
    out = df.copy()
    for column in _CATEGORY_COLUMNS:
        if column not in out.columns:
            continue
        values = out[column].astype("string").str.strip()
        mask = values.isna() | values.map(_is_placeholder)
        values = values.mask(mask, "Не указано")
        out[column] = values
    return out


def _period_label(value: object, granularity: str) -> str:
    if value is None or pd.isna(value):
        return "—"
    ts = pd.Timestamp(value)
    if granularity == "month":
        return f"{_RU_MONTHS[ts.month]} {ts.year}"
    if granularity == "week":
        return f"с {ts:%d.%m.%Y}"
    return f"{ts:%d.%m.%Y}"


def _timeseries_details(df: pd.DataFrame, *, metric: str, granularity: str) -> pd.DataFrame:
    """Return the chart metric together with the denominator N for every period."""
    columns = ["period", "value", "n", "numerator"]
    if df.empty or "created_at" not in df.columns:
        return pd.DataFrame(columns=columns)

    freq = _RESAMPLE_FREQ[granularity]
    work = df.loc[df["created_at"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work = work.set_index("created_at").sort_index()

    if metric == "requests":
        value = work.resample(freq, label="left", closed="left").size().astype(float)
        n = value.astype(int)
        numerator = n.astype(float)
    elif metric == "overdue_rate":
        numeric = work["is_overdue"].astype("Float64")
        grouped = numeric.resample(freq, label="left", closed="left")
        n = grouped.count().astype(int)
        numerator = grouped.sum(min_count=1).astype(float)
        value = grouped.mean() * 100.0
    elif metric == "median_duration":
        numeric = pd.to_numeric(work["duration_hours"], errors="coerce")
        grouped = numeric.resample(freq, label="left", closed="left")
        n = grouped.count().astype(int)
        numerator = pd.Series(float("nan"), index=n.index)
        value = grouped.median()
    else:
        raise ValueError(f"Unknown timeseries metric: {metric}")

    result = pd.concat(
        [value.rename("value"), n.rename("n"), numerator.rename("numerator")],
        axis=1,
    ).reset_index().rename(columns={"created_at": "period"})
    result.loc[result["n"].eq(0), "value"] = pd.NA
    return result[columns]


def _metric_text(metric: str, value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "Нет данных"
    if metric == "requests":
        return _fmt_int(value)
    if metric == "overdue_rate":
        return _fmt_float(float(value), "%", digits=1)
    return _fmt_float(float(value), " ч", digits=1)


def _overall_metric(df: pd.DataFrame, metric: str) -> tuple[float | None, str]:
    if df.empty:
        return None, "N = 0"
    if metric == "requests":
        return float(len(df)), f"N = {len(df):,}".replace(",", " ")
    if metric == "overdue_rate":
        known = df["is_overdue"].dropna()
        if known.empty:
            return None, "Нет обращений с известным статусом"
        overdue = int(known.fillna(False).sum())
        rate = overdue / len(known) * 100.0
        return rate, f"{overdue} из {len(known)} обращений"
    duration = pd.to_numeric(df["duration_hours"], errors="coerce").dropna()
    if duration.empty:
        return None, "Нет обращений с рассчитанной длительностью"
    return float(duration.median()), f"N = {len(duration):,}".replace(",", " ")


def _change_note(metric: str, current: float | None, previous: float | None) -> str:
    if current is None or previous is None or pd.isna(current) or pd.isna(previous):
        return "Нет данных для сравнения"
    if metric == "overdue_rate":
        delta = float(current) - float(previous)
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta:.1f} п.п. к предыдущему периоду".replace(".", ",")
    if metric == "median_duration":
        delta = float(current) - float(previous)
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta:.1f} ч к предыдущему периоду".replace(".", ",")
    if float(previous) == 0:
        return "Предыдущий период = 0"
    delta_pct = (float(current) / float(previous) - 1.0) * 100.0
    sign = "+" if delta_pct > 0 else ""
    return f"{sign}{delta_pct:.1f}% к предыдущему периоду".replace(".", ",")


def _base_timeseries_layout(fig: go.Figure, *, height: int, y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=42, r=16, t=12, b=38),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B7C1D1", size=12),
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="rgba(132,148,171,0.20)",
        tickfont=dict(color="#8FA0B8"),
    )
    fig.update_yaxes(
        title=y_title,
        gridcolor="rgba(132,148,171,0.13)",
        zeroline=False,
        tickfont=dict(color="#8FA0B8"),
        title_font=dict(color="#8FA0B8"),
    )
    return fig


def _make_metric_figure(
    series: pd.DataFrame,
    *,
    metric: str,
    granularity: str,
    height: int,
    previous: pd.DataFrame | None = None,
    compact: bool = False,
) -> go.Figure:
    """Readable charts: counts=line, overdue=bars, duration=line with explicit gaps."""
    fig = go.Figure()
    x = series.get("period", pd.Series(dtype="datetime64[ns]"))

    if metric == "overdue_rate":
        custom = series[["n", "numerator"]].to_numpy() if not series.empty else None
        fig.add_trace(
            go.Bar(
                x=x,
                y=series.get("value"),
                name="Текущий период",
                marker_color="#FF5C62",
                opacity=0.88,
                customdata=custom,
                hovertemplate=(
                    "%{x|%d.%m.%Y}<br>"
                    "Доля просрочек: %{y:.1f}%<br>"
                    "Просрочено: %{customdata[1]:.0f}<br>"
                    "N с известным статусом: %{customdata[0]:.0f}<extra></extra>"
                ),
            )
        )
        if previous is not None and not previous.empty:
            fig.add_trace(
                go.Bar(
                    x=previous["period"],
                    y=previous["value"],
                    name="Предыдущий период",
                    marker_color="rgba(151,165,186,0.42)",
                    customdata=previous[["n", "numerator"]].to_numpy(),
                    hovertemplate=(
                        "Предыдущий период<br>"
                        "Доля просрочек: %{y:.1f}%<br>"
                        "N: %{customdata[0]:.0f}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(barmode="group", showlegend=True)
        _base_timeseries_layout(fig, height=height, y_title="Просрочено, %")
        fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    else:
        if metric == "requests":
            color = "#3D8BFF"
            y_title = "Обращения, шт."
            hover_value = "Обращений: %{y:.0f}"
            fill = "tozeroy"
        else:
            color = "#8B6CFF"
            y_title = "Медиана, ч"
            hover_value = "Медиана: %{y:.1f} ч"
            fill = "tozeroy" if compact else None

        fig.add_trace(
            go.Scatter(
                x=x,
                y=series.get("value"),
                mode="lines+markers",
                name="Текущий период",
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=color, line=dict(color="#F5F7FA", width=1)),
                fill=fill,
                fillcolor=("rgba(61,139,255,0.10)" if metric == "requests" else "rgba(139,108,255,0.09)"),
                connectgaps=False,
                customdata=series[["n"]].to_numpy() if not series.empty else None,
                hovertemplate=(
                    "%{x|%d.%m.%Y}<br>" + hover_value + "<br>N: %{customdata[0]:.0f}<extra></extra>"
                ),
            )
        )
        if previous is not None and not previous.empty:
            fig.add_trace(
                go.Scatter(
                    x=previous["period"],
                    y=previous["value"],
                    mode="lines+markers",
                    name="Предыдущий период",
                    line=dict(color="#91A0B7", width=2, dash="dash"),
                    marker=dict(size=6, color="#91A0B7"),
                    connectgaps=False,
                    customdata=previous[["n"]].to_numpy(),
                    hovertemplate=(
                        "Предыдущий период<br>" + hover_value + "<br>N: %{customdata[0]:.0f}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(showlegend=True)
        _base_timeseries_layout(fig, height=height, y_title=y_title)
        fig.update_yaxes(rangemode="tozero")

    if not compact and not series.empty:
        tickvals = series["period"].tolist()
        ticktext = [_period_label(v, granularity) for v in tickvals]
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    elif compact:
        fig.update_xaxes(nticks=5)

    if previous is not None and not previous.empty:
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(color="#AAB5C5", size=11),
                bgcolor="rgba(0,0,0,0)",
            )
        )
    return fig


def _insight_card(label: str, value: str, note: str = "") -> None:
    note_html = f'<div class="insight-note">{escape(note)}</div>' if note else ""
    st.markdown(
        f'<div class="insight-card">'
        f'<div class="insight-label">{escape(label)}</div>'
        f'<div class="insight-value">{escape(value)}</div>'
        f'{note_html}</div>',
        unsafe_allow_html=True,
    )


def _render_metric_insights(
    filtered: pd.DataFrame,
    series: pd.DataFrame,
    *,
    metric: str,
    granularity: str,
) -> None:
    st.markdown('<div class="insight-stack-title">Ключевые показатели</div>', unsafe_allow_html=True)
    valid = series.loc[series["value"].notna()].copy()
    if valid.empty:
        st.info("Для выбранного показателя нет данных.")
        return

    peak = valid.loc[valid["value"].idxmax()]
    low = valid.loc[valid["value"].idxmin()]
    overall_value, overall_note = _overall_metric(filtered, metric)
    latest = valid.iloc[-1]
    previous = valid.iloc[-2] if len(valid) >= 2 else None

    _insight_card(
        "Пик",
        _metric_text(metric, peak["value"]),
        f'{_period_label(peak["period"], granularity)} · N = {int(peak["n"])}',
    )
    _insight_card(
        "Минимум",
        _metric_text(metric, low["value"]),
        f'{_period_label(low["period"], granularity)} · N = {int(low["n"])}',
    )
    _insight_card(
        "За выбранный период",
        _metric_text(metric, overall_value),
        overall_note,
    )
    change_note = _change_note(
        metric,
        float(latest["value"]),
        float(previous["value"]) if previous is not None else None,
    )
    _insight_card(
        "Последний период",
        _metric_text(metric, latest["value"]),
        f'{_period_label(latest["period"], granularity)} · N = {int(latest["n"])} · {change_note}',
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
    st.session_state["analytics_period"] = (options["min_date"], options["max_date"])
    st.session_state["analytics_service"] = "Все услуги"
    st.session_state["analytics_request_type"] = "Все виды"
    st.session_state["analytics_priority"] = "Все приоритеты"
    st.session_state["analytics_line"] = "Все линии"
    st.session_state["analytics_overdue"] = "Все"
    st.session_state["analytics_status"] = "Все состояния"


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
        st.markdown('<div class="card-title">Общая динамика</div>', unsafe_allow_html=True)
        if filtered.empty:
            st.info("По выбранным фильтрам обращений не найдено.")
            return

        granularity_label = _default_granularity(filter_state["start_date"], filter_state["end_date"])
        granularity = GRANULARITIES[granularity_label]
        st.markdown(
            f'<div class="card-subtitle">Группировка: {escape(granularity_label.lower())}. '
            'Пропуск означает отсутствие данных, а не нулевое значение.</div>',
            unsafe_allow_html=True,
        )

        metrics = [
            ("Обращения", "requests"),
            ("Просрочки", "overdue_rate"),
            ("Время обработки", "median_duration"),
        ]
        cols = st.columns(3, gap="small")
        for col, (label, metric) in zip(cols, metrics):
            with col:
                series = _timeseries_details(filtered, metric=metric, granularity=granularity)
                overall_value, overall_note = _overall_metric(filtered, metric)
                st.markdown(f'<div class="mini-chart-title">{escape(label)}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="mini-chart-value">{escape(_metric_text(metric, overall_value))}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="mini-chart-note">{escape(overall_note)}</div>',
                    unsafe_allow_html=True,
                )
                fig = _make_metric_figure(
                    series,
                    metric=metric,
                    granularity=granularity,
                    height=255,
                    compact=True,
                )
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
    granularity_label = granularity_label or "Месяц"
    granularity = GRANULARITIES[granularity_label]

    if filtered.empty:
        st.info("По выбранным фильтрам обращений не найдено.")
        return

    with _profile("analytics.dynamics_current"):
        current = _timeseries_details(filtered, metric=metric, granularity=granularity)

    previous_series = None
    comparison_caption = None
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
        previous_series = _timeseries_details(
            previous_df,
            metric=metric,
            granularity=granularity,
        )
        if not previous_series.empty:
            previous_series = previous_series.copy()
            previous_series["period"] = previous_series["period"] + pd.Timedelta(days=inclusive_days)
        comparison_caption = (
            f"Сравнение с равным предыдущим интервалом: "
            f"{previous_start:%d.%m.%Y}–{previous_end:%d.%m.%Y}."
        )

    left, right = st.columns([0.76, 0.24], gap="medium")
    with left:
        with st.container(key="analytics_chart_card"):
            title_map = {
                "requests": "Динамика обращений",
                "overdue_rate": "Доля просроченных обращений",
                "median_duration": "Медианное время обработки",
            }
            subtitle_map = {
                "requests": "Количество зарегистрированных обращений по выбранным периодам.",
                "overdue_rate": "Доля просрочек; в подсказке показаны число просрочек и знаменатель N.",
                "median_duration": "Медиана длительности; N показывает число обращений с доступной длительностью.",
            }
            st.markdown(
                f'<div class="card-title">{escape(title_map[metric])}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="card-subtitle">{escape(subtitle_map[metric])} '
                'Пропуск — нет данных, не ноль.</div>',
                unsafe_allow_html=True,
            )
            if comparison_caption:
                st.caption(comparison_caption)

            fig = _make_metric_figure(
                current,
                metric=metric,
                granularity=granularity,
                previous=previous_series,
                height=455,
                compact=False,
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        _render_metric_insights(
            filtered,
            current,
            metric=metric,
            granularity=granularity,
        )


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


def _export_filename(filter_state: dict) -> str:
    start = filter_state["start_date"]
    end = filter_state["end_date"]
    return f"analytics_{start:%Y-%m-%d}_{end:%Y-%m-%d}.csv"


def render_analytics_tab(data_path: str | Path) -> None:
    """Render the complete analytics page from the approved mockup structure."""
    st.markdown(ANALYTICS_CSS, unsafe_allow_html=True)

    with _profile("analytics.load"):
        try:
            df = load_analytics_data(data_path)
        except (FileNotFoundError, AnalyticsDataError, ValueError) as exc:
            st.error(f"Не удалось подготовить данные аналитики: {exc}")
            return

    # One consistent user-facing value instead of technical placeholders such as --).
    df = _normalize_display_categories(df)

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
            file_name=_export_filename(filter_state),
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
            help=f"Скачать текущую выборку: {len(filtered):,} обращений.".replace(",", " "),
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
