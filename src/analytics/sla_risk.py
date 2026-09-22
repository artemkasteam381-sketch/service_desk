import pandas as pd
from pathlib import Path
from src.analytics.data_preparation import prepare_data

DATA_PATH = Path("data/raw/original.xlsx")
OUTPUT_PATH = Path("data/processed/analytics")


def load_data():
    return prepare_data()


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
            "Просрочки": (
                subset["Просрочен?*"]
                .eq("Просрочен")
                .sum()
            )
        })

    result = pd.DataFrame(result)

    result["Риск SLA (%)"] = (
        result["Просрочки"]
        / result["Обращения"]
        * 100
    ).round(2)

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

