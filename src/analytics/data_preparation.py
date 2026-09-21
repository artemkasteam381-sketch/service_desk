from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "original.xlsx"


def load_raw_data(path=None):
    """Читает исходник и проверяет идентификаторы обращений."""
    df = pd.read_excel(DATA_PATH if path is None else path)

    if "Номер запроса" not in df.columns:
        raise ValueError("Отсутствует столбец «Номер запроса»")

    if df["Номер запроса"].isna().any():
        raise ValueError("Есть обращения без номера")

    if df["Номер запроса"].duplicated().any():
        raise ValueError("Есть повторяющиеся номера обращений")

    return df

def prepare_data(path=None):
    df = load_raw_data(path).copy()

    date_columns = [
        "Дата регистрации",
        "Плановое время выполнения",
        "Крайний срок обработки",
        "Фактическое время выполнения",
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    df["sla_limit_td"] = pd.to_timedelta(
        df["Норматив ное время обработки (SLA)"],
        errors="coerce",
    )
    df["sla_actual_td"] = pd.to_timedelta(
        df["Фактическая длительность выполнения запроса (SLA)"],
        errors="coerce",
    )
    df["sla_actual_seconds"] = df["sla_actual_td"].dt.total_seconds()

    df["is_overdue"] = df["Просрочен?*"].eq("Просрочен")
    df["service_analytics"] = df["Услуга"].replace(
        {"---)": "Неопределённая услуга"}
    )

    line_flags = []

    for line_number in range(1, 5):
        source = f"Суммарное время работы {line_number} линии"
        time_column = f"line_{line_number}_time_td"
        flag_column = f"line_{line_number}_recorded"

        df[time_column] = pd.to_timedelta(df[source], errors="coerce")
        df[flag_column] = df[time_column] > pd.Timedelta(0)
        line_flags.append(flag_column)

    df["recorded_lines_count"] = df[line_flags].sum(axis=1)

    return df
