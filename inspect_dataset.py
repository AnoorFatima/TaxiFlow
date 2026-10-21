import pandas as pd

file_path = r"C:\Users\anoor\OneDrive\Desktop\yellow_tripdata_2026-01.parquet"

df = pd.read_parquet(file_path)

print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nPickup date range:")
print(df["tpep_pickup_datetime"].min())
print(df["tpep_pickup_datetime"].max())

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique pickup zones:",
      df["PULocationID"].nunique())

print("\nUnique drop-off zones:",
      df["DOLocationID"].nunique())