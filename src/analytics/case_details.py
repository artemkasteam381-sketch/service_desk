from src.analytics.data_preparation import PROJECT_ROOT, prepare_data


OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "analytics"


def multiline_cases(df):
    """Возвращает обращения с участием минимум двух линий."""

    data = df.loc[
        df["recorded_lines_count"].ge(2)
    ].copy()

    line_columns = {
        "1 линия": "line_1_recorded",
        "2 линия": "line_2_recorded",
        "3 линия": "line_3_recorded",
        "4 линия": "line_4_recorded",
    }

    data["Участвовавшие линии"] = data.apply(
        lambda row: ", ".join(
            line
            for line, column in line_columns.items()
            if row[column]
        ),
        axis=1,
    )

    data["Длительность SLA (часов)"] = (
        data["sla_actual_seconds"] / 3600
    ).round(2)

    result = data[
        [
            "Номер запроса",
            "Дата регистрации",
            "service_analytics",
            "Вид запроса",
            "Приоритет",
            "recorded_lines_count",
            "Участвовавшие линии",
            "is_overdue",
            "Длительность SLA (часов)",
        ]
    ].rename(
        columns={
            "service_analytics": "Услуга",
            "recorded_lines_count": "Количество линий",
            "is_overdue": "Просрочено",
        }
    )

    return result.sort_values(
        ["Количество линий", "Номер запроса"],
        ascending=[False, True],
    )
def many_clarifications_cases(df, minimum=4):
    """Возвращает обращения с большим количеством уточнений."""

    data = df.loc[
        df["clarifications_count"].ge(minimum)
    ].copy()

    data["Количество уточнений"] = (
        data["clarifications_count"].astype(int)
    )

    data["Время уточнений (часов)"] = (
        data["clarification_time_seconds"] / 3600
    ).round(2)

    data["Длительность SLA (часов)"] = (
        data["sla_actual_seconds"] / 3600
    ).round(2)

    result = data[
        [
            "Номер запроса",
            "Дата регистрации",
            "service_analytics",
            "Вид запроса",
            "Приоритет",
            "Количество уточнений",
            "Время уточнений (часов)",
            "is_overdue",
            "Длительность SLA (часов)",
        ]
    ].rename(
        columns={
            "service_analytics": "Услуга",
            "is_overdue": "Просрочено",
        }
    )

    return result.sort_values(
        ["Количество уточнений", "Номер запроса"],
        ascending=[False, True],
    )


if __name__ == "__main__":
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    data = prepare_data()

    multiline = multiline_cases(data)
    clarifications = many_clarifications_cases(data)

    multiline.to_csv(
        OUTPUT_PATH / "multiline_cases.csv",
        index=False,
    )

    clarifications.to_csv(
        OUTPUT_PATH / "many_clarifications_cases.csv",
        index=False,
    )

    print("Созданы:")
    print("- multiline_cases.csv")
    print("- many_clarifications_cases.csv")
