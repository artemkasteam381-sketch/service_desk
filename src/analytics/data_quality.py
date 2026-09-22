from pathlib import Path

import pandas as pd

from src.analytics.data_preparation import PROJECT_ROOT, prepare_data


OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "analytics"


def build_quality_flags(df):
    """Создаёт признаки возможных проблем в исходных данных."""

    flags = pd.DataFrame({
        "Номер запроса": df["Номер запроса"],
    })

    flags["Нет длительности SLA"] = (
        df["sla_actual_seconds"].isna()
    )

    flags["Неопределённая услуга"] = (
        df["service_analytics"].eq("Неопределённая услуга")
    )

    flags["Нет времени линий"] = (
        df["recorded_lines_count"].eq(0)
    )

    flags["Уточнения без времени"] = (
        df["clarifications_count"].gt(0)
        & df["clarification_time_seconds"].fillna(0).le(0)
    )

    duration_comparable = (
        df["sla_actual_td"].notna()
        & df["sla_limit_td"].notna()
    )

    overdue_by_duration = (
        df["sla_actual_td"] > df["sla_limit_td"]
    )

    flags["Расхождение с длительностью SLA"] = (
        duration_comparable
        & overdue_by_duration.ne(df["is_overdue"])
    )

    deadline_comparable = (
        df["Фактическое время выполнения"].notna()
        & df["Крайний срок обработки"].notna()
    )

    overdue_by_deadline = (
        df["Фактическое время выполнения"]
        > df["Крайний срок обработки"]
    )

    flags["Расхождение с крайним сроком"] = (
        deadline_comparable
        & overdue_by_deadline.ne(df["is_overdue"])
    )

    return flags




def quality_summary(flags):
    flag_columns = [
        column
        for column in flags.columns
        if column != "Номер запроса"
    ]

    result = pd.DataFrame({
        "Проверка": flag_columns,
        "Количество": [
            int(flags[column].sum())
            for column in flag_columns
        ],
    })

    result["Доля (%)"] = (
        result["Количество"]
        / len(flags)
        * 100
    ).round(2)

    return result


def issue_records(flags):
    flag_columns = [
        column
        for column in flags.columns
        if column != "Номер запроса"
    ]

    has_issue = flags[flag_columns].any(axis=1)

    return flags.loc[has_issue].copy()



if __name__ == "__main__":
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    data = prepare_data()
    flags = build_quality_flags(data)
    summary = quality_summary(flags)
    issues = issue_records(flags)

    summary.to_csv(
        OUTPUT_PATH / "data_quality_summary.csv",
        index=False,
    )

    issues.to_csv(
        OUTPUT_PATH / "data_quality_issues.csv",
        index=False,
    )

    print(summary.to_string(index=False))
    print()
    print(
        "Обращений хотя бы с одной отметкой качества:",
        len(issues),
    )
