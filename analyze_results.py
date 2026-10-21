
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# TASK 8: TAXIFLOW PERFORMANCE ANALYSIS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "test_period_results.csv"
OUTPUT_PATH = BASE_DIR / "performance_analysis.csv"

# Load test-period evaluation results
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Results file not found: {INPUT_PATH}\n"
        "Run evaluate_test_period.py first."
    )

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "Policy",
    "Intervals",
    "Total Reward",
    "Estimated Pickups Served",
    "Estimated Unmet Demand",
    "Repositioning Cost"
]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

if df["Intervals"].le(0).any():
    raise ValueError("Intervals must be greater than zero.")

# --------------------------------------------------
# 1. Calculate performance metrics
# --------------------------------------------------

df["Reward Per Interval"] = (
    df["Total Reward"] / df["Intervals"]
)

df["Estimated Total Demand"] = (
    df["Estimated Pickups Served"]
    + df["Estimated Unmet Demand"]
)

df["Estimated Service Rate (%)"] = (
    df["Estimated Pickups Served"]
    / df["Estimated Total Demand"].replace(0, float("nan"))
    * 100
).fillna(0)

df["Estimated Unmet Demand Rate (%)"] = (
    df["Estimated Unmet Demand"]
    / df["Estimated Total Demand"].replace(0, float("nan"))
    * 100
).fillna(0)

df["Repositioning Cost Per Interval"] = (
    df["Repositioning Cost"] / df["Intervals"]
)

# --------------------------------------------------
# 2. Display the complete results table
# --------------------------------------------------

display_columns = [
    "Policy",
    "Intervals",
    "Total Reward",
    "Reward Per Interval",
    "Estimated Pickups Served",
    "Estimated Unmet Demand",
    "Estimated Service Rate (%)",
    "Estimated Unmet Demand Rate (%)",
    "Repositioning Cost",
    "Repositioning Cost Per Interval"
]

print("\n" + "=" * 90)
print("TAXIFLOW - TEST-PERIOD PERFORMANCE ANALYSIS")
print("=" * 90)

print(
    df[display_columns].round(3).to_string(index=False)
)

# --------------------------------------------------
# 3. Compare Q-learning with the other policies
# --------------------------------------------------

policy_lookup = {
    str(policy).strip().lower(): index
    for index, policy in enumerate(df["Policy"])
}

required_policies = [
    "no repositioning",
    "demand greedy",
    "q-learning"
]

missing_policies = [
    policy for policy in required_policies
    if policy not in policy_lookup
]

if missing_policies:
    raise ValueError(
        f"Missing policies in results: {missing_policies}"
    )

baseline = df.iloc[policy_lookup["no repositioning"]]
greedy = df.iloc[policy_lookup["demand greedy"]]
q_learning = df.iloc[policy_lookup["q-learning"]]


def percentage_change(new_value, old_value):
    """Return percentage change relative to old_value."""
    if old_value == 0:
        return float("nan")
    return (new_value - old_value) / abs(old_value) * 100


comparisons = []

for reference_name, reference in [
    ("No repositioning", baseline),
    ("Demand greedy", greedy)
]:
    comparisons.append({
        "Comparison": f"Q-learning vs {reference_name}",
        "Reward Difference": (
            q_learning["Total Reward"]
            - reference["Total Reward"]
        ),
        "Reward Improvement (%)": percentage_change(
            q_learning["Total Reward"],
            reference["Total Reward"]
        ),
        "Served Estimate Difference": (
            q_learning["Estimated Pickups Served"]
            - reference["Estimated Pickups Served"]
        ),
        "Served Estimate Improvement (%)": percentage_change(
            q_learning["Estimated Pickups Served"],
            reference["Estimated Pickups Served"]
        ),
        "Unmet Demand Difference": (
            q_learning["Estimated Unmet Demand"]
            - reference["Estimated Unmet Demand"]
        ),
        "Unmet Demand Reduction (%)": (
            percentage_change(
                reference["Estimated Unmet Demand"],
                q_learning["Estimated Unmet Demand"]
            )
        ),
        "Repositioning Cost Difference": (
            q_learning["Repositioning Cost"]
            - reference["Repositioning Cost"]
        )
    })

comparison_df = pd.DataFrame(comparisons)

print("\n" + "=" * 90)
print("Q-LEARNING COMPARISON")
print("=" * 90)
print(comparison_df.round(3).to_string(index=False))

# --------------------------------------------------
# 4. Save analysis results
# --------------------------------------------------

df[display_columns].to_csv(
    OUTPUT_PATH,
    index=False
)

COMPARISON_PATH = BASE_DIR / "policy_comparison.csv"

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False
)

print("\nAnalysis completed successfully.")
print(f"Performance metrics saved to: {OUTPUT_PATH}")
print(f"Policy comparisons saved to: {COMPARISON_PATH}")

print(
    "\nNote: Service rates, served pickups, and unmet demand "
    "are estimates from the simplified simulation, not "
    "measured real-world outcomes."
)