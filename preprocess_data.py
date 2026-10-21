
import pandas as pd
from pathlib import Path

# 1. File paths
DATA_PATH = Path(r"C:\Users\anoor\OneDrive\Desktop\yellow_tripdata_2026-01.parquet")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Load only the columns we need
df = pd.read_parquet(
    DATA_PATH,
    columns=[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "trip_distance"
    ]
)

print("Original records:", len(df))

# 3. Keep January 2026 records only
start_date = pd.Timestamp("2026-01-01")
end_date = pd.Timestamp("2026-02-01")

df = df[
    (df["tpep_pickup_datetime"] >= start_date)
    & (df["tpep_pickup_datetime"] < end_date)
].copy()

# 4. Remove invalid records
df = df[
    df["tpep_pickup_datetime"].notna()
    & df["tpep_dropoff_datetime"].notna()
    & df["PULocationID"].notna()
    & (df["tpep_dropoff_datetime"]
       >= df["tpep_pickup_datetime"])
    & (df["trip_distance"] > 0)
    & (df["PULocationID"] > 0)
].copy()

print("Valid January records:", len(df))

# 5. Create 15-minute time intervals
df["time_interval"] = (
    df["tpep_pickup_datetime"].dt.floor("15min")
)

# 6. Calculate pickup demand per zone per interval
demand = (
    df.groupby(["time_interval", "PULocationID"])
      .size()
      .rename("pickup_count")
      .reset_index()
      .rename(columns={"PULocationID": "zone_id"})
)

# 7. Include intervals and zones with zero pickups
zones = sorted(df["PULocationID"].unique())

intervals = pd.date_range(
    start=start_date,
    end=end_date,
    freq="15min",
    inclusive="left"
)

full_index = pd.MultiIndex.from_product(
    [intervals, zones],
    names=["time_interval", "zone_id"]
)

demand = (
    demand.set_index(["time_interval", "zone_id"])
          .reindex(full_index, fill_value=0)
          .reset_index()
)

# 8. Add useful time features
demand["hour"] = demand["time_interval"].dt.hour
demand["day_of_week"] = demand["time_interval"].dt.dayofweek
demand["is_weekend"] = (
    demand["day_of_week"] >= 5
).astype(int)

# 9. Chronological split: no random shuffling
train = demand[
    demand["time_interval"] < "2026-01-22"
].copy()

validation = demand[
    (demand["time_interval"] >= "2026-01-22")
    & (demand["time_interval"] < "2026-01-27")
].copy()

test = demand[
    demand["time_interval"] >= "2026-01-27"
].copy()

# 10. Save outputs
demand.to_csv(
    OUTPUT_DIR / "demand_by_zone_15min.csv",
    index=False
)

train.to_csv(OUTPUT_DIR / "train_demand.csv", index=False)
validation.to_csv(
    OUTPUT_DIR / "validation_demand.csv", index=False
)
test.to_csv(OUTPUT_DIR / "test_demand.csv", index=False)

print("\nDemand dataset shape:", demand.shape)
print("Number of zones:", len(zones))
print("Number of time intervals:", len(intervals))
print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Testing rows:", len(test))

print("\nTop 10 zones by total January pickups:")
print(
    demand.groupby("zone_id")["pickup_count"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
)

print("\nPreprocessing completed successfully.")