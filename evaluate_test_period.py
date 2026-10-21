import numpy as np
import pandas as pd
from taxi_environment import TaxiEnvironment

# Load the trained model
model = np.load("q_learning_model.npz")
q_table = {
    tuple(state): values
    for state, values in zip(model["states"], model["q_values"])
}

# Find the start of the test period
env = TaxiEnvironment()
times = pd.to_datetime(env.times)
test_start = np.where(times >= "2026-01-27")[0][0]

# Evaluate one strategy
def evaluate(name):
    env = TaxiEnvironment()
    state = env.reset(start_step=test_start)

    reward_total = 0
    served_total = 0
    unmet_total = 0
    cost_total = 0
    done = False

    while not done:
        if name == "No repositioning":
            action = 0
        elif name == "Demand greedy":
            action = 1
        else:
            action = int(np.argmax(
                q_table.get(tuple(state), np.zeros(3))
            ))

        state, reward, done, info = env.step(action)

        reward_total += reward
        served_total += info["served_pickups_estimate"]
        unmet_total += info["unmet_demand_estimate"]
        cost_total += info["reposition_cost"]

    return {
        "Policy": name,
        "Total Reward": reward_total,
        "Estimated Pickups Served": served_total,
        "Estimated Unmet Demand": unmet_total,
        "Repositioning Cost": cost_total
    }

# Compare all three strategies
results = []

for policy in ["No repositioning", "Demand greedy", "Q-learning"]:
    results.append(evaluate(policy))

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))
results_df.to_csv("test_period_results.csv", index=False)

print("\nResults saved to test_period_results.csv")