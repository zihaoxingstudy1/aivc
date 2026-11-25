import pandas as pd
df = pd.read_parquet(r"D:\data\AIVC\data\tahoe_100m\metadata\pseudobulk_differential_expression\train-00000-of-01026.parquet")
print("shape:", df.shape)
print("cols:", list(df.columns))
print("head(2):", df.head(2).to_dict(orient="records"))
