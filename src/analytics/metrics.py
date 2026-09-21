from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KPISet:
    requests: int
    overdue_count: int
    overdue_rate: float
    median_duration: float | None
    p90_duration: float | None
    multi_line_count: int
    multi_line_rate: float
    median_clarifications: float | None


def _finite_or_none(value) -> float | None:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return None
    return float(value)


def calculate_kpis(df: pd.DataFrame) -> KPISet:
    total = int(len(df))
    if total == 0:
        return KPISet(0, 0, 0.0, None, None, 0, 0.0, None)

    overdue_count = int(df["is_overdue"].fillna(False).sum())
    multi_line_count = int(df["multi_line"].fillna(False).sum())

    duration = pd.to_numeric(df["duration_hours"], errors="coerce").dropna()
    clarifications = pd.to_numeric(df["clarifications"], errors="coerce").dropna()

    return KPISet(
        requests=total,
        overdue_count=overdue_count,
        overdue_rate=overdue_count / total,
        median_duration=_finite_or_none(duration.median() if not duration.empty else None),
        p90_duration=_finite_or_none(duration.quantile(0.90) if not duration.empty else None),
        multi_line_count=multi_line_count,
        multi_line_rate=multi_line_count / total,
        median_clarifications=_finite_or_none(
            clarifications.median() if not clarifications.empty else None
        ),
    )


FREQ_MAP = {
    "day": "D",
    "week": "W-MON",
    "month": "MS",
}


def calculate_timeseries(
    df: pd.DataFrame,
    *,
    metric: str,
    granularity: str,
) -> pd.DataFrame:
    """Aggregate a selected metric by registration date."""
    if df.empty:
        return pd.DataFrame(columns=["period", "value"])

    freq = FREQ_MAP[granularity]
    work = df.loc[df["created_at"].notna()].set_index("created_at").sort_index()

    if metric == "requests":
        series = work.resample(freq, label="left", closed="left").size().astype(float)
        series = series.fillna(0.0)
    elif metric == "overdue_rate":
        numeric = work["is_overdue"].astype("Float64")
        series = numeric.resample(freq, label="left", closed="left").mean() * 100.0
    elif metric == "median_duration":
        numeric = pd.to_numeric(work["duration_hours"], errors="coerce")
        series = numeric.resample(freq, label="left", closed="left").median()
    else:
        raise ValueError(f"Unknown timeseries metric: {metric}")

    result = series.rename("value").reset_index().rename(columns={"created_at": "period"})
    return result


DIMENSIONS = {
    "service": "service",
    "request_type": "request_type",
    "priority": "priority",
    "final_line": "final_line",
}


def calculate_breakdown(df: pd.DataFrame, *, dimension: str) -> pd.DataFrame:
    if dimension not in DIMENSIONS:
        raise ValueError(f"Unknown dimension: {dimension}")

    column = DIMENSIONS[dimension]
    if df.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "requests",
                "load_share",
                "overdue_count",
                "overdue_rate",
                "median_duration",
                "p90_duration",
                "median_clarifications",
                "multi_line_rate",
            ]
        )

    work = df.copy()
    work["_group"] = work[column].fillna("Не указано").astype(str)
    work["_overdue"] = work["is_overdue"].astype("Float64")
    work["_multi_line"] = work["multi_line"].astype(float)

    grouped = work.groupby("_group", dropna=False, sort=False)
    result = grouped.agg(
        requests=("ticket_id", "size"),
        overdue_count=("_overdue", lambda s: int(s.fillna(0).sum())),
        overdue_rate=("_overdue", "mean"),
        median_duration=("duration_hours", "median"),
        p90_duration=("duration_hours", lambda s: s.dropna().quantile(0.90) if s.notna().any() else np.nan),
        median_clarifications=("clarifications", "median"),
        multi_line_rate=("_multi_line", "mean"),
    ).reset_index().rename(columns={"_group": "group"})

    result["requests"] = result["requests"].astype(int)
    result["overdue_count"] = result["overdue_count"].fillna(0).astype(int)
    total = len(work)
    result["load_share"] = result["requests"] / total
    return result.sort_values(["requests", "group"], ascending=[False, True]).reset_index(drop=True)


def calculate_sla_risk(
    df: pd.DataFrame,
    *,
    dimension: str,
    alpha: float = 10.0,
    min_group_size: int = 5,
    uplift_threshold: float = 0.03,
) -> pd.DataFrame:
    """Historical smoothed overdue rate; this is an analytical rule, not a forecast."""
    breakdown = calculate_breakdown(df, dimension=dimension)
    if breakdown.empty:
        return breakdown.assign(
            smoothed_risk=pd.Series(dtype=float),
            risk_delta=pd.Series(dtype=float),
            elevated_risk=pd.Series(dtype=bool),
        )

    global_rate = float(df["is_overdue"].astype("Float64").mean())
    if not np.isfinite(global_rate):
        global_rate = 0.0

    breakdown = breakdown.copy()
    breakdown["smoothed_risk"] = (
        breakdown["overdue_count"] + alpha * global_rate
    ) / (breakdown["requests"] + alpha)
    breakdown["risk_delta"] = breakdown["smoothed_risk"] - global_rate
    breakdown["elevated_risk"] = (
        (breakdown["requests"] >= min_group_size)
        & (breakdown["risk_delta"] >= uplift_threshold)
    )
    breakdown["global_overdue_rate"] = global_rate

    return breakdown.sort_values(
        ["elevated_risk", "smoothed_risk", "requests"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def get_problem_cases(
    df: pd.DataFrame,
    *,
    mode: str,
    limit: int = 200,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Return concrete requests for a problem mode and the applied threshold metadata."""
    meta: dict[str, float | int | None] = {"threshold": None}

    if df.empty:
        return df.copy(), meta

    if mode == "long_duration":
        duration = pd.to_numeric(df["duration_hours"], errors="coerce").dropna()
        threshold = float(duration.quantile(0.90)) if not duration.empty else np.nan
        meta["threshold"] = threshold if np.isfinite(threshold) else None
        cases = df.loc[df["duration_hours"].ge(threshold)] if np.isfinite(threshold) else df.iloc[0:0]
        cases = cases.sort_values("duration_hours", ascending=False)

    elif mode == "clarifications":
        values = pd.to_numeric(df["clarifications"], errors="coerce").dropna()
        threshold = float(values.quantile(0.90)) if not values.empty else np.nan
        if np.isfinite(threshold):
            threshold = max(1.0, threshold)
            cases = df.loc[df["clarifications"].ge(threshold)]
            cases = cases.sort_values(["clarifications", "duration_hours"], ascending=[False, False])
        else:
            cases = df.iloc[0:0]
        meta["threshold"] = threshold if np.isfinite(threshold) else None

    elif mode == "multi_line":
        cases = df.loc[df["multi_line"].fillna(False)].sort_values(
            ["line_count", "duration_hours"], ascending=[False, False]
        )

    elif mode == "overdue":
        cases = df.loc[df["is_overdue"].fillna(False)].sort_values(
            ["duration_hours", "created_at"], ascending=[False, False]
        )

    else:
        raise ValueError(f"Unknown problem mode: {mode}")

    meta["total_cases"] = int(len(cases))
    return cases.head(limit).copy(), meta


def summarize_problem_cases(
    cases: pd.DataFrame,
    *,
    dimension: str = "request_type",
) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame(columns=["group", "requests", "share"])

    column = DIMENSIONS[dimension]
    work = cases.copy()
    work["_group"] = work[column].fillna("Не указано").astype(str)
    summary = work.groupby("_group", dropna=False).size().rename("requests").reset_index().rename(columns={"_group": "group"})
    summary["share"] = summary["requests"] / len(work)
    return summary.sort_values("requests", ascending=False).reset_index(drop=True)
