import pandas as pd
from pathlib import Path
from src.analytics.data_preparation import prepare_data


DATA_PATH = Path("data/raw/original.xlsx")
OUTPUT_PATH = Path("data/processed/analytics")


def load_data():
    return prepare_data()

def analyze_lines(df):
    result = (
        df["recorded_lines_count"]
        .value_counts()
        .sort_index()
        .rename_axis("Количество линий с зафиксированным временем")
        .reset_index(name="Количество обращений")
    )

    return result


def line_work_time(df):
    result = []
    total = len(df)

    for line_number in range(1, 5):
        flag_column = f"line_{line_number}_recorded"
        time_column = f"line_{line_number}_time_td"

        seconds = (
            df.loc[df[flag_column], time_column]
            .dt.total_seconds()
        )

        result.append({
            "Линия": f"{line_number} линия",
            "Всего обращений": total,
            "С зафиксированным временем": len(seconds),
            "Доля обращений (%)": round(len(seconds) / total * 100, 2),
            "Среднее часов": round(seconds.mean() / 3600, 2),
            "Медиана часов": round(seconds.median() / 3600, 2),
            "P90 часов": round(seconds.quantile(0.9) / 3600, 2),
            "Максимум часов": round(seconds.max() / 3600, 2),
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
