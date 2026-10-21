
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from taxi_environment import TaxiEnvironment


# ==========================================
# 1. File paths and configuration
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "demand_by_zone_15min.csv"
MODEL_PATH = BASE_DIR / "q_learning_model.npz"
OUTPUT_PATH = BASE_DIR / "test_period_results.csv"

TOTAL_TAXIS = 100

# January 27, 2026, at 00:00
TEST_START = pd.Timestamp("2026-01-27 00:00:00")

# February 1, 2026, at 00:00 (exclusive)
TEST_END = pd.Timestamp("2026-02-01 00:00:00")


# ==========================================
# 2. Load the trained Q-learning model
# ==========================================
model = np.load(MODEL_PATH)

saved_states = model["states"]
saved_q_values = model["q_values"]

q_table = defaultdict(lambda: np.zeros(3))

for saved_state, values in zip(saved_states, saved_q_values):
    state_key = tuple(int(value) for value in saved_state)
    q_table[state_key] = values

print("Q-learning model loaded successfully.")


# ==========================================
# 3. Find the held-out test period
# ==========================================
reference_env = TaxiEnvironment(
    data_path=DATA_PATH,
    total_taxis=TOTAL_TAXIS
)

timestamps = pd.to_datetime(reference_env.time_steps)

test_indices = np.where(
    (timestamps >= TEST_START)
    & (timestamps < TEST_END)
)[0]

if len(test_indices) == 0:
    raise ValueError(
        "No test intervals found. Check the dataset dates."
    )

test_start_index = int(test_indices[0])

print("Test start:", timestamps[test_indices[0]])
print("Test end:", timestamps[test_indices[-1]])
print("Available test intervals:", len(test_indices))


# ==========================================
# 4. Evaluate a policy
# ==========================================
def evaluate_policy(policy_name, choose_action):

    env = TaxiEnvironment(
        data_path=DATA_PATH,
        total_taxis=TOTAL_TAXIS
    )

    state = env.reset(start_step=test_start_index)

    total_reward = 0.0
    total_served = 0
    total_unmet = 0
    total_cost = 0.0
    steps = 0

    done = False

    while not done:
        action = choose_action(state)

        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_served += info["served_pickups_estimate"]
        total_unmet += info["unmet_demand_estimate"]
        total_cost += info["reposition_cost"]
        steps += 1

        state = next_state

    result = {
        "Policy": policy_name,
        "Intervals": steps,
        "Total Reward": total_reward,
        "Estimated Pickups Served": total_served,
        "Estimated Unmet Demand": total_unmet,
        "Repositioning Cost": total_cost
    }

    print(f"\n{policy_name}")
    print("-" * 42)
    for key, value in result.items():
        if key != "Policy":
            print(f"{key}: {value}")

    return result


# ==========================================
# 5. Define three policies
# ==========================================

# Policy A: Never reposition
def no_reposition_policy(state):
    return 0


# Policy B: Demand-greedy repositioning.
# Action 1 moves a taxi from a low-demand
# zone toward the highest-demand zone.
def demand_greedy_policy(state):
    return 1


# Policy C: Trained Q-learning policy
def q_learning_policy(state):
    return int(np.argmax(q_table[tuple(state)]))


# ==========================================
# 6. Run all policies on the same test dates
# ==========================================
results = []

results.append(
    evaluate_policy(
        "No repositioning",
        no_reposition_policy
    )
)

results.append(
    evaluate_policy(
        "Demand greedy",
        demand_greedy_policy
    )
)

results.append(
    evaluate_policy(
        "Q-learning",
        q_learning_policy
    )
)


# ==========================================
# 7. Display and save the comparison
# ==========================================
results_df = pd.DataFrame(results)

print("\n\nFINAL TEST-PERIOD COMPARISON")
print("=" * 75)
print(results_df.to_string(index=False))

results_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nResults saved to: {OUTPUT_PATH}")


# ==========================================
# 8. Compare Q-learning with the baseline
# ==========================================
baseline = results_df[
    results_df["Policy"] == "No repositioning"
].iloc[0]

trained = results_df[
    results_df["Policy"] == "Q-learning"
].iloc[0]

print("\nQ-LEARNING VS NO-REPOSITIONING")
print("=" * 45)

print(
    "Reward difference:",
    trained["Total Reward"] - baseline["Total Reward"]
)

print(
    "Served-pickup estimate difference:",
    trained["Estimated Pickups Served"]
    - baseline["Estimated Pickups Served"]
)

print(
    "Unmet-demand estimate difference:",
    trained["Estimated Unmet Demand"]
    - baseline["Estimated Unmet Demand"]
)

print("\nTest-period evaluation completed.")