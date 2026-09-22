import pandas as pd

from src.analytics.data_preparation import PROJECT_ROOT, prepare_data


OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "analytics"


def monthly_dynamics(df):
    """Считает основные показатели по месяцу регистрации."""

    data = df.dropna(subset=["Дата регистрации"]).copy()

    data["Месяц"] = (
        data["Дата регистрации"]
        .dt.to_period("M")
        .astype(str)
    )

    result = (
        data.groupby("Месяц", sort=True)
        .agg(
            Обращения=("Номер запроса", "nunique"),
            Просрочки=("is_overdue", "sum"),
            Обращения_с_временем=("sla_actual_seconds", "count"),
            Медиана_часов=("sla_actual_seconds", "median"),
            P90_часов=(
                "sla_actual_seconds",
                lambda values: values.quantile(0.9),
            ),
        )
        .reset_index()
    )
    all_months = (
        pd.period_range(
            start=data["Дата регистрации"].min().to_period("M"),
            end=data["Дата регистрации"].max().to_period("M"),
            freq="M",
        )
        .astype(str)
    )

    result = (
        result.set_index("Месяц")
        .reindex(all_months)
        .rename_axis("Месяц")
        .reset_index()
    )

    count_columns = [
        "Обращения",
        "Просрочки",
        "Обращения_с_временем",
    ]

    result[count_columns] = (
        result[count_columns]
        .fillna(0)
        .astype(int)
    )

    result["Доля просрочки (%)"] = (
        result["Просрочки"]
        / result["Обращения"]
        * 100
    ).round(2)

    result["Пропуски_времени"] = (
        result["Обращения"]
        - result["Обращения_с_временем"]
    )

    result[["Медиана_часов", "P90_часов"]] = (
        result[["Медиана_часов", "P90_часов"]]
        / 3600
    ).round(2)
    result["Нет данных"] = result["Обращения"].eq(0)

    result["Малая выборка"] = (
        result["Обращения"].between(1, 29)
    )

    return result


if __name__ == "__main__":
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    data = prepare_data()
    result = monthly_dynamics(data)

    result.to_csv(
        OUTPUT_PATH / "monthly_dynamics.csv",
        index=False,
    )

    print(result.to_string(index=False))
    print()
    print("Создан monthly_dynamics.csv")
