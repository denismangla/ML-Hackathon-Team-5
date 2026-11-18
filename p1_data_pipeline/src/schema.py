import pandas as pd
from src.features import FEATURE_COLS

df = pd.read_parquet("data/processed/train_candidates.parquet")
schema = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(t) for t in df.dtypes],
    "role": ["feature" if c in FEATURE_COLS else "key/target" for c in df.columns],
    "min": [df[c].min() for c in df.columns],
    "max": [df[c].max() for c in df.columns],
    "missing": [int(df[c].isna().sum()) for c in df.columns],
})
print(df.shape)
print(schema.to_string(index=False))
schema.to_csv("results/feature_schema.csv", index=False)
print("saved results/feature_schema.csv")