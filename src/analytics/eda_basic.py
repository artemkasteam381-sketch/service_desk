import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/original.xlsx")
OUTPUT_PATH = Path("data/processed/analytics")


def load_data():
    return pd.read_excel(DATA_PATH)


def basic_kpi(df):
    total = len(df)

    overdue = (
        df["Просрочен?*"]
        .eq("Просрочен")
        .sum()
    )

    return pd.DataFrame({
        "Показатель": [
            "Всего обращений",
            "Просрочено",
            "Доля просрочки (%)"
        ],
        "Значение": [
            total,
            overdue,
            round(overdue / total * 100, 2)
        ]
    })


def service_analysis(df):
    result = (
        df["Услуга"]
        .value_counts()
        .reset_index()
    )

    result.columns = [
        "Услуга",
        "Количество обращений"
    ]

    result["Доля (%)"] = (
        result["Количество обращений"]
        / len(df)
        * 100
    ).round(2)

    return result


def request_type_analysis(df):
    return (
        df["Вид запроса"]
        .value_counts()
        .reset_index(name="Количество обращений")
    )


def priority_analysis(df):
    result = (
        df.groupby("Приоритет")
        .agg(
            Обращения=("Номер запроса", "count"),
            Просрочки=("Просрочен?*", lambda x: (x == "Просрочен").sum())
        )
        .reset_index()
    )

    result["Доля просрочки (%)"] = (
        result["Просрочки"]
        / result["Обращения"]
        * 100
    ).round(2)

    return result


def processing_time_analysis(df):
    df = df.copy()

    df["Время обработки"] = pd.to_timedelta(
        df["Фактическая длительность выполнения запроса (SLA)"],
        errors="coerce"
    )

    result = (
        df.groupby("Услуга")
        .agg(
            Обращения=("Номер запроса", "count"),
            Среднее_время=("Время обработки", "mean"),
            Медиана_времени=("Время обработки", "median"),
            Максимальное_время=("Время обработки", "max")
        )
        .reset_index()
    )

    return result


if __name__ == "__main__":

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    df = load_data()

    tables = {
        "basic_kpi.csv": basic_kpi(df),
        "service_load.csv": service_analysis(df),
        "request_types.csv": request_type_analysis(df),
        "priority_sla.csv": priority_analysis(df),
        "processing_time.csv": processing_time_analysis(df)
    }

    for filename, table in tables.items():
        table.to_csv(
            OUTPUT_PATH / filename,
            index=False
        )

    print("Аналитические таблицы сохранены:")
    for file in tables:
        print("-", file)
