import pandas as pd
from pathlib import Path
from src.analytics.data_preparation import prepare_data

DATA_PATH = Path("data/raw/original.xlsx")
OUTPUT_PATH = Path("data/processed/analytics")


def load_data():
    return prepare_data()

def add_wilson_interval(result):
    """Добавляет 95% интервал для исторической доли просрочек."""

    z = 1.96
    total = result["Обращения"].astype(float)
    overdue = result["Просрочки"].astype(float)
    probability = overdue / total

    denominator = 1 + z**2 / total

    center = (
        probability + z**2 / (2 * total)
    ) / denominator

    margin = (
        z
        * (
            probability * (1 - probability) / total
            + z**2 / (4 * total**2)
        ) ** 0.5
        / denominator
    )

    result["Нижняя граница риска 95%"] = (
        (center - margin) * 100
    ).round(2)

    result["Верхняя граница риска 95%"] = (
        (center + margin) * 100
    ).round(2)

    return result


def calculate_risk(df, group_column):

    result = (
        df.groupby(group_column)
        .agg(
            Обращения=("Номер запроса", "count"),
            Просрочки=("is_overdue", "sum")
        )
        .reset_index()
    )

    result["Риск SLA (%)"] = (
        result["Просрочки"]
        / result["Обращения"]
        * 100
    ).round(2)
    result = add_wilson_interval(result)

    overall_risk = df["is_overdue"].mean() * 100

    result["Общий риск SLA (%)"] = round(overall_risk, 2)

    result["Повышенный риск"] = (
        result["Обращения"].ge(30)
        & (
            result["Нижняя граница риска 95%"]
            > overall_risk
        )
    )

    result["Малая выборка"] = result["Обращения"] < 30
    
    return (
        result
        .sort_values(
            "Риск SLA (%)",
            ascending=False
        )
    )


def line_risk(df):

    df = df.copy()

    lines = {
        "1 линия": "line_1_recorded",
        "2 линия": "line_2_recorded",
        "3 линия": "line_3_recorded",
        "4 линия": "line_4_recorded",
    }

    result = []

    for line, column in lines.items():

        used = df[column]

        subset = df[used]

        result.append({
            "Линия": line,
            "Обращения": len(subset),
            "Просрочки": int(subset["is_overdue"].sum())
        })

    result = pd.DataFrame(result)

    result["Риск SLA (%)"] = (
        result["Просрочки"]
        / result["Обращения"]
        * 100
    ).round(2)
    result = add_wilson_interval(result)

    overall_risk = df["is_overdue"].mean() * 100

    result["Общий риск SLA (%)"] = round(overall_risk, 2)
    result["Повышенный риск"] = (
        result["Обращения"].ge(30)
        & (
            result["Нижняя граница риска 95%"]
            > overall_risk
        )
    )

    result["Малая выборка"] = result["Обращения"] < 30

    return result


if __name__ == "__main__":

    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()


    service = calculate_risk(
        df,
        "service_analytics"
    )
    service = service.rename(
        columns={"service_analytics": "Услуга"}
    )
    request_type = calculate_risk(
        df,
        "Вид запроса"
    )

    line = line_risk(df)


    service.to_csv(
        OUTPUT_PATH / "service_sla_risk.csv",
        index=False
    )

    request_type.to_csv(
        OUTPUT_PATH / "request_type_sla_risk.csv",
        index=False
    )

    line.to_csv(
        OUTPUT_PATH / "line_sla_risk.csv",
        index=False
    )


    print("Созданы:")
    print("- service_sla_risk.csv")
    print("- request_type_sla_risk.csv")
    print("- line_sla_risk.csv")

