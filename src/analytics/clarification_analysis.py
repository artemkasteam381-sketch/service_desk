import pandas as pd

from src.analytics.data_preparation import PROJECT_ROOT, prepare_data


OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "analytics"


def clarification_analysis(df):
    df = df.copy()

    df["Группа уточнений"] = "2+"
    df.loc[df["clarifications_count"].eq(0), "Группа уточнений"] = "0"
    df.loc[df["clarifications_count"].eq(1), "Группа уточнений"] = "1"

    result = (
        df.groupby("Группа уточнений")
        .agg(
            Обращения=("Номер запроса", "nunique"),
            Обращения_с_временем=("sla_actual_seconds", "count"),
            Среднее_часов=("sla_actual_seconds", "mean"),
            Медиана_часов=("sla_actual_seconds", "median"),
            P90_часов=(
                "sla_actual_seconds",
                lambda values: values.quantile(0.9),
            ),
        )
        .reset_index()
    )

    result["Пропуски_времени"] = (
        result["Обращения"] - result["Обращения_с_временем"]
    )

    time_columns = [
        "Среднее_часов",
        "Медиана_часов",
        "P90_часов",
    ]
    result[time_columns] = (result[time_columns] / 3600).round(2)

    return result


def clarification_distribution(df):
    counts = df["clarifications_count"]

    groups = pd.Series("4+", index=df.index)
    groups.loc[counts.eq(0)] = "0"
    groups.loc[counts.eq(1)] = "1"
    groups.loc[counts.eq(2)] = "2"
    groups.loc[counts.eq(3)] = "3"
    groups.loc[counts.isna()] = "Нет данных"

    result = (
        groups.value_counts()
        .rename_axis("Количество уточнений")
        .reset_index(name="Количество обращений")
    )

    result["Доля (%)"] = (
        result["Количество обращений"] / len(df) * 100
    ).round(2)

    order = {"0": 0, "1": 1, "2": 2, "3": 3, "4+": 4, "Нет данных": 5}

    return (
        result.assign(
            order=result["Количество уточнений"].map(order)
        )
        .sort_values("order")
        .drop(columns="order")
    )

if __name__ == "__main__":
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    df = prepare_data()

    result = clarification_analysis(df)
    result.to_csv(
        OUTPUT_PATH / "clarification_analysis.csv",
        index=False,
    )

    distribution = clarification_distribution(df)
    distribution.to_csv(
        OUTPUT_PATH / "clarification_distribution.csv",
        index=False,
    )

    print(result)
    print(distribution)
