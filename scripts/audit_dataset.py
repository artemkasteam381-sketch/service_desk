from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "original.xlsx"
OUTPUT_DIR = ROOT / "docs" / "data_audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_excel(DATA_PATH)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


profile = pd.DataFrame(
    {
        "column": df.columns,
        "dtype": [str(df[col].dtype) for col in df.columns],
        "non_null": [df[col].notna().sum() for col in df.columns],
        "missing": [df[col].isna().sum() for col in df.columns],
        "fill_pct": [
            round(df[col].notna().mean() * 100, 2)
            for col in df.columns
        ],
        "unique": [
            df[col].nunique(dropna=True)
            for col in df.columns
        ],
        "examples": [
            " | ".join(
                df[col]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )
            for col in df.columns
        ],
    }
)


profile.to_csv(
    OUTPUT_DIR / "data_dictionary.csv",
    index=False,
    encoding="utf-8-sig",
)


print(profile.to_string(index=False))
