
import pandas as pd
from pathlib import Path

# 1. Load the dataset
file_path = Path(r"C:\Users\anoor\OneDrive\Desktop\yellow_tripdata_2026-01.parquet")
df = pd.read_parquet(
    file_path,
    columns=[
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "trip_distance"
    ]
)
print("Original records:", len(df))

# 2. Keep valid January trips
df = df[
    (df["tpep_pickup_datetime"] >= "2026-01-01")
    & (df["tpep_pickup_datetime"] < "2026-02-01")
    & (df["tpep_dropoff_datetime"] >= df["tpep_pickup_datetime"])
    & (df["trip_distance"] > 0)
    & (df["PULocationID"] > 0)
].copy()

df = df.dropna(subset=[
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID"
])

print("Valid January records:", len(df))

# 3. Group pickups into 15-minute intervals
df["time_interval"] = df["tpep_pickup_datetime"].dt.floor("15min")

demand = df.groupby(
    ["time_interval", "PULocationID"]
).size().reset_index(name="pickup_count")

demand = demand.rename(columns={"PULocationID": "zone_id"})

# 4. Split data by date
train = demand[demand["time_interval"] < "2026-01-22"]
validation = demand[
    (demand["time_interval"] >= "2026-01-22")
    & (demand["time_interval"] < "2026-01-27")
]
test = demand[demand["time_interval"] >= "2026-01-27"]

# 5. Save the files
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

demand.to_csv(output_dir / "demand_by_zone_15min.csv", index=False)
train.to_csv(output_dir / "train_demand.csv", index=False)
validation.to_csv(output_dir / "validation_demand.csv", index=False)
test.to_csv(output_dir / "test_demand.csv", index=False)

print("Demand dataset shape:", demand.shape)
print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Testing rows:", len(test))
print("Preprocessing completed successfully.")