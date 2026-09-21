from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


COLORS = {
    "blue": "#2F81F7",
    "red": "#FF5D5D",
    "orange": "#FF9F43",
    "green": "#35CFA0",
    "purple": "#9B6DFF",
    "text": "#E8EDF5",
    "muted": "#8E9AAF",
    "grid": "rgba(142, 154, 175, 0.14)",
    "panel": "rgba(18, 25, 36, 0.12)",
}

METRIC_META = {
    "requests": {"label": "Обращения", "suffix": "", "color": COLORS["blue"], "digits": 0},
    "overdue_rate": {"label": "Доля просроченных", "suffix": "%", "color": COLORS["red"], "digits": 1},
    "median_duration": {"label": "Медиана обработки", "suffix": " ч", "color": COLORS["green"], "digits": 1},
}

BREAKDOWN_META = {
    "requests": {"label": "Обращения", "suffix": "", "color": COLORS["blue"], "digits": 0},
    "load_share": {"label": "Доля нагрузки", "suffix": "%", "color": COLORS["blue"], "digits": 1},
    "overdue_rate": {"label": "Доля просроченных", "suffix": "%", "color": COLORS["red"], "digits": 1},
    "median_duration": {"label": "Медиана обработки", "suffix": " ч", "color": COLORS["green"], "digits": 1},
    "p90_duration": {"label": "P90 обработки", "suffix": " ч", "color": COLORS["purple"], "digits": 1},
    "median_clarifications": {"label": "Медиана уточнений", "suffix": "", "color": COLORS["orange"], "digits": 1},
    "multi_line_rate": {"label": "Доля нескольких линий", "suffix": "%", "color": COLORS["orange"], "digits": 1},
}


def _base_layout(fig: go.Figure, *, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=16, t=18, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=12),
        hoverlabel=dict(bgcolor="#151D29", font_color=COLORS["text"], bordercolor="#2A3444"),
        showlegend=False,
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color=COLORS["muted"]),
        linecolor=COLORS["grid"],
    )
    fig.update_yaxes(
        gridcolor=COLORS["grid"],
        zeroline=False,
        tickfont=dict(color=COLORS["muted"]),
    )
    return fig


def make_timeseries_chart(
    current: pd.DataFrame,
    *,
    metric: str,
    previous: pd.DataFrame | None = None,
    height: int = 340,
) -> go.Figure:
    meta = METRIC_META[metric]
    fig = go.Figure()

    hover_fmt = f"%{{y:.{meta['digits']}f}}{meta['suffix']}"

    fig.add_trace(
        go.Scatter(
            x=current["period"],
            y=current["value"],
            mode="lines+markers",
            line=dict(color=meta["color"], width=2.6, shape="spline", smoothing=0.6),
            marker=dict(size=7, color=meta["color"], line=dict(width=1.5, color="#D9E8FF")),
            fill="tozeroy" if metric == "requests" else None,
            fillcolor="rgba(47, 129, 247, 0.12)" if metric == "requests" else None,
            name="Текущий период",
            hovertemplate=f"%{{x|%d.%m.%Y}}<br>{meta['label']}: {hover_fmt}<extra></extra>",
        )
    )

    if previous is not None and not previous.empty:
        fig.add_trace(
            go.Scatter(
                x=previous["period"],
                y=previous["value"],
                mode="lines",
                line=dict(color=COLORS["muted"], width=2, dash="dash"),
                name="Предыдущий период",
                hovertemplate=f"Предыдущий период<br>{meta['label']}: {hover_fmt}<extra></extra>",
            )
        )
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0))

    _base_layout(fig, height=height)
    fig.update_yaxes(title=None, rangemode="tozero")
    fig.update_xaxes(title=None)
    if metric == "overdue_rate":
        fig.update_yaxes(ticksuffix="%")
    return fig


def make_breakdown_chart(
    breakdown: pd.DataFrame,
    *,
    metric: str,
    top_n: int = 12,
    height: int = 430,
) -> go.Figure:
    meta = BREAKDOWN_META[metric]
    plot = breakdown.copy()

    if metric in {"load_share", "overdue_rate", "multi_line_rate"}:
        plot["_value"] = plot[metric] * 100.0
    else:
        plot["_value"] = plot[metric]

    plot = plot.dropna(subset=["_value"]).nlargest(top_n, "_value").sort_values("_value")

    fig = go.Figure(
        go.Bar(
            x=plot["_value"],
            y=plot["group"],
            orientation="h",
            marker=dict(color=meta["color"]),
            text=[
                f"{value:.{meta['digits']}f}{meta['suffix']}"
                for value in plot["_value"]
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{y}}<br>{meta['label']}: %{{x:.{meta['digits']}f}}{meta['suffix']}<extra></extra>",
        )
    )
    _base_layout(fig, height=height)
    fig.update_layout(margin=dict(l=8, r=70, t=12, b=16))
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], rangemode="tozero")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=COLORS["text"]))
    return fig


def make_problem_summary_chart(
    summary: pd.DataFrame,
    *,
    height: int = 410,
) -> go.Figure:
    plot = summary.head(12).sort_values("requests")
    fig = go.Figure(
        go.Bar(
            x=plot["requests"],
            y=plot["group"],
            orientation="h",
            marker=dict(color=COLORS["orange"]),
            text=plot["requests"],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Обращений: %{x}<extra></extra>",
        )
    )
    _base_layout(fig, height=height)
    fig.update_layout(margin=dict(l=8, r=48, t=12, b=16))
    fig.update_xaxes(rangemode="tozero", showgrid=True, gridcolor=COLORS["grid"])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=COLORS["text"]))
    return fig


def make_sla_risk_chart(
    risk: pd.DataFrame,
    *,
    top_n: int = 12,
    height: int = 430,
) -> go.Figure:
    plot = risk.dropna(subset=["smoothed_risk"]).copy()
    plot["risk_pct"] = plot["smoothed_risk"] * 100.0
    plot = plot.nlargest(top_n, "risk_pct").sort_values("risk_pct")

    colors = [COLORS["red"] if flag else COLORS["orange"] for flag in plot["elevated_risk"]]
    fig = go.Figure(
        go.Bar(
            x=plot["risk_pct"],
            y=plot["group"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{x:.1f}%" for x in plot["risk_pct"]],
            textposition="outside",
            cliponaxis=False,
            customdata=plot[["requests", "overdue_count", "overdue_rate"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>Сглаженная доля: %{x:.1f}%"
                "<br>Обращений: %{customdata[0]}"
                "<br>Просрочек: %{customdata[1]}"
                "<br>Обычная доля: %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    _base_layout(fig, height=height)
    fig.update_layout(margin=dict(l=8, r=70, t=12, b=16))
    fig.update_xaxes(ticksuffix="%", rangemode="tozero", showgrid=True, gridcolor=COLORS["grid"])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=COLORS["text"]))
    return fig
