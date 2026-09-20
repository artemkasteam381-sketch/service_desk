import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/original.xlsx")
OUTPUT_PATH = Path("data/processed/analytics")


df = pd.read_excel(DATA_PATH)

df["Время SLA"] = pd.to_timedelta(
    df["Фактическая длительность выполнения запроса (SLA)"],
    errors="coerce"
)

df["Группа уточнений"] = pd.cut(
    df["Количество уточнений"],
    bins=[-1, 0, 1, 100],
    labels=["0", "1", "2+"]
)

result = (
    df.groupby(
        "Группа уточнений",
        observed=True
    )
    .agg(
        Обращения=("Номер запроса", "count"),
        Среднее_время=("Время SLA", "mean"),
        Медианное_время=("Время SLA", "median")
    )
    .reset_index()
)

result.to_csv(
    OUTPUT_PATH / "clarification_analysis.csv",
    index=False
)

print(result)
