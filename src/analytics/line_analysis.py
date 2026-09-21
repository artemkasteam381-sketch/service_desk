import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/original.xlsx")
OUTPUT_PATH = Path("data/processed/analytics")


def load_data():
    return pd.read_excel(DATA_PATH)


def analyze_lines(df):

    df = df.copy()

    line_columns = [
        "Суммарное время работы 1 линии",
        "Суммарное время работы 2 линии",
        "Суммарное время работы 3 линии",
        "Суммарное время работы 4 линии",
    ]

    for i, col in enumerate(line_columns, start=1):
        df[f"line_{i}_used"] = (
            pd.to_timedelta(df[col], errors="coerce")
            .fillna(pd.Timedelta(0))
            > pd.Timedelta(0)
        )

    df["Количество участвовавших линий"] = (
        df[
            [
                "line_1_used",
                "line_2_used",
                "line_3_used",
                "line_4_used",
            ]
        ]
        .sum(axis=1)
    )

    result = (
        df["Количество участвовавших линий"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    result.columns = [
        "Количество линий",
        "Количество обращений"
    ]

    return result


def line_work_time(df):

    result = []

    lines = {
        "1 линия": "Суммарное время работы 1 линии",
        "2 линия": "Суммарное время работы 2 линии",
        "3 линия": "Суммарное время работы 3 линии",
        "4 линия": "Суммарное время работы 4 линии",
    }

    for name, column in lines.items():

        time = pd.to_timedelta(
            df[column],
            errors="coerce"
        )

        result.append({
            "Линия": name,
            "Среднее время работы": time.mean(),
            "Медианное время работы": time.median()
        })

    return pd.DataFrame(result)


if __name__ == "__main__":

    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    participation = analyze_lines(df)
    work_time = line_work_time(df)

    participation.to_csv(
        OUTPUT_PATH / "line_participation.csv",
        index=False
    )

    work_time.to_csv(
        OUTPUT_PATH / "line_work_time.csv",
        index=False
    )

    print("Созданы:")
    print("- line_participation.csv")
    print("- line_work_time.csv")
