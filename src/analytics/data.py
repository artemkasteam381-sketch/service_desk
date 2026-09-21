from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# Canonical field names isolate the analytics/UI code from Russian source headers.
FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "ticket_id": ("Номер запроса", "ID обращения", "id", "record_id"),
    "created_at": ("Дата регистрации", "Дата создания", "created_at"),
    "completed_at": ("Фактическое время выполнения", "Дата закрытия", "closed_at"),
    "deadline_at": ("Крайний срок обработки", "Крайний срок", "deadline_at"),
    "status": ("Статус", "Состояние обращения", "status"),
    "service": ("Услуга", "service"),
    "request_type": ("Вид запроса", "Вид обращения", "request_type", "topic"),
    "priority": ("Приоритет", "priority"),
    "final_line": ("Кем решен (группа)", "Итоговая линия", "Линия поддержки", "support_line"),
    "clarifications": ("Количество уточнений", "clarifications"),
    "sla_limit_raw": (
        "Норматив ное время обработки (SLA)",
        "Нормативное время обработки (SLA)",
        "SLA",
        "sla_limit",
    ),
    "sla_duration_raw": (
        "Фактическая длительность выполнения запроса (SLA)",
        "Фактическая длительность (SLA)",
        "sla_duration",
    ),
    "overdue_raw": ("Просрочен?*", "Просрочен", "SLA статус", "overdue"),
    "source_url": ("source_url", "Ссылка", "Ссылка на обращение"),
}

LINE_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    f"line_{line}_{kind}_raw": (
        f"Суммарное время {'реакции' if kind == 'reaction' else 'работы'} {line} линии",
    )
    for line in range(1, 5)
    for kind in ("reaction", "work")
}

REQUIRED_FIELDS = ("ticket_id", "created_at", "service", "request_type")


class AnalyticsDataError(ValueError):
    """Raised when the source dataset cannot safely power the analytics UI."""


def _first_present(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _canonicalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}

    for canonical, candidates in {**FIELD_CANDIDATES, **LINE_FIELD_CANDIDATES}.items():
        source = _first_present(raw.columns, candidates)
        if source is not None:
            rename_map[source] = canonical

    df = raw.rename(columns=rename_map).copy()

    missing = [field for field in REQUIRED_FIELDS if field not in df.columns]
    if missing:
        readable = ", ".join(missing)
        raise AnalyticsDataError(
            "В датасете не найдены обязательные поля для аналитики: " + readable
        )

    optional = set(FIELD_CANDIDATES) | set(LINE_FIELD_CANDIDATES)
    for field in optional:
        if field not in df.columns:
            df[field] = pd.NA

    return df


def _clean_text(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()

    # Normalize empty / technical placeholder values from the source data.
    out = out.mask(out.isin(["", "nan", "NaN", "None", "<NA>"]))
    out = out.mask(out.str.fullmatch(r"[-—–\s()]+", na=False))
    return out


def _duration_to_hours(series: pd.Series) -> pd.Series:
    """Parse SLA-like durations, including values such as 120:00:00."""
    text = _clean_text(series)
    td = pd.to_timedelta(text, errors="coerce")
    hours = td.dt.total_seconds() / 3600.0
    return hours.mask(hours < 0)


def _parse_overdue(series: pd.Series) -> pd.Series:
    """Parse the historical overdue flag without confusing 'не просрочен'."""
    text = _clean_text(series).str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")

    if text.empty:
        return result

    not_overdue = text.str.contains(r"(?:^|\s)не\s+просроч", regex=True, na=False)
    overdue = text.str.contains(r"просроч", regex=True, na=False) & ~not_overdue

    truthy = text.isin(["true", "1", "да", "yes"])
    falsy = text.isin(["false", "0", "нет", "no"])

    result.loc[overdue | truthy] = True
    result.loc[not_overdue | falsy] = False
    return result


def _read_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise AnalyticsDataError(
        f"Неподдерживаемый формат файла {suffix!r}. Используйте CSV/XLSX."
    )


def prepare_analytics_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical, analysis-ready DataFrame.

    Heavy normalization belongs here so it is performed once and cached by
    ``load_analytics_data`` instead of being repeated on each Streamlit rerun.
    """
    df = _canonicalize_columns(raw)

    # Text/categorical fields.
    for column in ("service", "request_type", "priority", "status", "final_line", "source_url"):
        df[column] = _clean_text(df[column])

    # Replace technical placeholders with user-facing analytical labels.
    for column in ("service", "request_type", "priority", "status", "final_line"):
        df[column] = (
            df[column]
            .replace({"--)": pd.NA, "--": pd.NA, "-": pd.NA})
            .fillna("Не определена")
        )

    # IDs are deliberately kept as strings: they are identifiers, not measures.
    df["ticket_id"] = _clean_text(df["ticket_id"])

    # Dates.
    for column in ("created_at", "completed_at", "deadline_at"):
        df[column] = pd.to_datetime(df[column], errors="coerce")

    if df["created_at"].notna().sum() == 0:
        raise AnalyticsDataError("Поле даты регистрации не содержит корректных дат.")

    # Numeric quantities.
    df["clarifications"] = pd.to_numeric(df["clarifications"], errors="coerce").clip(lower=0)
    df["sla_limit_hours"] = _duration_to_hours(df["sla_limit_raw"])
    df["sla_duration_hours"] = _duration_to_hours(df["sla_duration_raw"])

    calendar_hours = (
        (df["completed_at"] - df["created_at"]).dt.total_seconds() / 3600.0
    )
    df["calendar_duration_hours"] = calendar_hours.mask(calendar_hours < 0)

    # The source has a dedicated SLA duration field. Prefer it when available;
    # otherwise fall back to calendar elapsed time and expose the basis.
    df["duration_hours"] = df["sla_duration_hours"].combine_first(
        df["calendar_duration_hours"]
    )
    df["duration_basis"] = np.where(
        df["sla_duration_hours"].notna(), "sla_duration", "calendar_duration"
    )
    df.loc[df["duration_hours"].isna(), "duration_basis"] = pd.NA

    # Overdue: historical flag first, then safe fallbacks only for missing rows.
    historical_overdue = _parse_overdue(df["overdue_raw"])
    calculated_from_sla = (
        df["sla_duration_hours"].notna()
        & df["sla_limit_hours"].notna()
        & (df["sla_duration_hours"] > df["sla_limit_hours"])
    )
    calculated_from_deadline = (
        df["completed_at"].notna()
        & df["deadline_at"].notna()
        & (df["completed_at"] > df["deadline_at"])
    )

    df["is_overdue"] = historical_overdue
    missing_overdue = df["is_overdue"].isna()
    df.loc[missing_overdue, "is_overdue"] = calculated_from_sla[missing_overdue]
    still_missing = df["is_overdue"].isna()
    df.loc[still_missing, "is_overdue"] = calculated_from_deadline[still_missing]
    df["is_overdue"] = df["is_overdue"].astype("boolean")

    df["overdue_basis"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[historical_overdue.notna(), "overdue_basis"] = "historical_flag"
    df.loc[historical_overdue.isna() & df["sla_duration_hours"].notna() & df["sla_limit_hours"].notna(), "overdue_basis"] = "sla_duration_vs_limit"
    df.loc[df["overdue_basis"].isna() & df["completed_at"].notna() & df["deadline_at"].notna(), "overdue_basis"] = "completion_vs_deadline"

    # Lines involved. The source contains aggregate reaction/work time per line,
    # so it is more informative than inferring transfer only from the final line.
    line_flags: list[pd.Series] = []
    for line in range(1, 5):
        reaction = _duration_to_hours(df[f"line_{line}_reaction_raw"])
        work = _duration_to_hours(df[f"line_{line}_work_raw"])
        df[f"line_{line}_reaction_hours"] = reaction
        df[f"line_{line}_work_hours"] = work

        final_line_match = df["final_line"].str.contains(
            rf"\({line}\s*линия\)", regex=True, na=False
        )
        involved = reaction.gt(0).fillna(False) | work.gt(0).fillna(False) | final_line_match
        df[f"line_{line}_involved"] = involved
        line_flags.append(involved)

    line_matrix = pd.concat(line_flags, axis=1)
    df["line_count"] = line_matrix.sum(axis=1).astype("int16")
    df["multi_line"] = df["line_count"].ge(2)

    line_arrays = [flag.to_numpy(dtype=bool) for flag in line_flags]
    involved_labels: list[str] = []
    for row_i in range(len(df)):
        labels = [f"{line} линия" for line, arr in enumerate(line_arrays, start=1) if arr[row_i]]
        involved_labels.append(", ".join(labels))
    df["involved_lines"] = pd.Series(involved_labels, index=df.index, dtype="string").replace("", pd.NA)

    # Useful derived fields for filters/aggregation.
    df["created_date"] = df["created_at"].dt.date
    df["created_month"] = df["created_at"].dt.to_period("M").dt.to_timestamp()

    # Keep deterministic order for charts and exports.
    return df.sort_values("created_at", kind="stable").reset_index(drop=True)


@st.cache_data(show_spinner="Подготавливаю исторические данные…", max_entries=4)
def _load_analytics_data_cached(
    path_str: str,
    file_mtime_ns: int,
    file_size: int,
) -> pd.DataFrame:
    # mtime/size intentionally participate in the cache key; they are not used
    # inside the function body except to make invalidation explicit.
    del file_mtime_ns, file_size
    path = Path(path_str)
    raw = _read_source(path)
    return prepare_analytics_data(raw)


def load_analytics_data(path: str | Path) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Файл аналитики не найден: {path}")

    stat = path.stat()
    return _load_analytics_data_cached(str(path), stat.st_mtime_ns, stat.st_size)


def clear_analytics_data_cache() -> None:
    _load_analytics_data_cached.clear()


def get_filter_options(df: pd.DataFrame) -> dict[str, Any]:
    created = df["created_at"].dropna()
    if created.empty:
        raise AnalyticsDataError("Нельзя определить доступный период: нет корректных дат.")

    def values(column: str) -> list[str]:
        return sorted(df[column].dropna().astype(str).unique().tolist())

    return {
        "min_date": created.min().date(),
        "max_date": created.max().date(),
        "services": values("service"),
        "request_types": values("request_type"),
        "priorities": values("priority"),
        "lines": values("final_line"),
        "statuses": values("status"),
    }


def apply_filters(
    df: pd.DataFrame,
    *,
    start_date,
    end_date,
    service: str | None = None,
    request_type: str | None = None,
    priority: str | None = None,
    final_line: str | None = None,
    overdue_status: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    # Inclusive calendar-day filter without converting every row to Python date.
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    mask = df["created_at"].ge(start_ts) & df["created_at"].lt(end_exclusive)

    if service and service != "Все услуги":
        mask &= df["service"].eq(service)
    if request_type and request_type != "Все виды":
        mask &= df["request_type"].eq(request_type)
    if priority and priority != "Все приоритеты":
        mask &= df["priority"].eq(priority)
    if final_line and final_line != "Все линии":
        mask &= df["final_line"].eq(final_line)
    if status and status != "Все состояния":
        mask &= df["status"].eq(status)

    if overdue_status == "Просроченные":
        mask &= df["is_overdue"].fillna(False)
    elif overdue_status == "Не просроченные":
        mask &= df["is_overdue"].eq(False).fillna(False)

    return df.loc[mask].copy()


def export_view(df: pd.DataFrame) -> pd.DataFrame:
    """Small, readable export instead of exposing all parser/ML columns."""
    columns = [
        "ticket_id",
        "created_at",
        "completed_at",
        "service",
        "request_type",
        "priority",
        "status",
        "final_line",
        "duration_hours",
        "is_overdue",
        "clarifications",
        "line_count",
        "involved_lines",
        "source_url",
    ]
    existing = [column for column in columns if column in df.columns]
    out = df[existing].copy()
    return out.rename(
        columns={
            "ticket_id": "ID обращения",
            "created_at": "Дата регистрации",
            "completed_at": "Дата выполнения",
            "service": "Услуга",
            "request_type": "Вид обращения",
            "priority": "Приоритет",
            "status": "Статус",
            "final_line": "Итоговая линия",
            "duration_hours": "Длительность, ч",
            "is_overdue": "Просрочено",
            "clarifications": "Количество уточнений",
            "line_count": "Количество линий",
            "involved_lines": "Участвовавшие линии",
            "source_url": "Ссылка",
        }
    )
