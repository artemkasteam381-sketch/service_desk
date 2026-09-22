import pandas as pd


LINE_COLUMNS = {
    "1 линия": "line_1_recorded",
    "2 линия": "line_2_recorded",
    "3 линия": "line_3_recorded",
    "4 линия": "line_4_recorded",
}


def filter_data(
    df,
    start_date=None,
    end_date=None,
    services=None,
    request_types=None,
    priorities=None,
    lines=None,
):
    result = df.copy()

    if start_date is not None:
        result = result[
            result["Дата регистрации"] >= pd.Timestamp(start_date)
        ]

    if end_date is not None:
        end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        result = result[
            result["Дата регистрации"] < end_exclusive
        ]

    if services:
        result = result[
            result["service_analytics"].isin(services)
        ]

    if request_types:
        result = result[
            result["Вид запроса"].isin(request_types)
        ]

    if priorities:
        result = result[
            result["Приоритет"].isin(priorities)
        ]

    if lines:
        unknown = set(lines) - set(LINE_COLUMNS)
        if unknown:
            raise ValueError(
                f"Неизвестные линии: {sorted(unknown)}"
            )

        selected_columns = [
            LINE_COLUMNS[line] for line in lines
        ]
        result = result[
            result[selected_columns].any(axis=1)
        ]

    return result.copy()
