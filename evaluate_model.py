
import numpy as np
from collections import defaultdict

from taxi_environment import TaxiEnvironment


# 1. Load the saved Q-table
model = np.load("q_learning_model.npz")

states = model["states"]
q_values = model["q_values"]

q_table = defaultdict(lambda: np.zeros(3))

for state, values in zip(states, q_values):
    q_table[tuple(int(x) for x in state)] = values


def evaluate_policy(policy_name, policy_action):
    env = TaxiEnvironment(
        data_path="data/demand_by_zone_15min.csv",
        total_taxis=100
    )

    state = env.reset(start_step=0)

    total_reward = 0.0
    total_served = 0
    total_unmet = 0
    total_reposition_cost = 0.0
    steps = 0

    done = False

    while not done:
        action = policy_action(state)

        state, reward, done, info = env.step(action)

        total_reward += reward
        total_served += info["served_pickups_estimate"]
        total_unmet += info["unmet_demand_estimate"]
        total_reposition_cost += info["reposition_cost"]
        steps += 1

    print(f"\n{policy_name}")
    print("-" * 40)
    print("Total reward:", round(total_reward, 2))
    print("Estimated pickups served:", total_served)
    print("Estimated unmet demand:", total_unmet)
    print("Repositioning cost:", round(total_reposition_cost, 2))
    print("Intervals evaluated:", steps)

    return {
        "reward": total_reward,
        "served": total_served,
        "unmet": total_unmet,
        "cost": total_reposition_cost
    }


# 2. Baseline: never reposition
baseline = evaluate_policy(
    "No-repositioning baseline",
    lambda state: 0
)


# 3. Trained Q-learning policy
trained = evaluate_policy(
    "Trained Q-learning policy",
    lambda state: int(np.argmax(q_table[state]))
)


# 4. Compare results
print("\nPOLICY COMPARISON")
print("=" * 40)

print(
    "Reward difference:",
    round(trained["reward"] - baseline["reward"], 2)
)

print(
    "Served pickup estimate difference:",
    trained["served"] - baseline["served"]
)

print(
    "Unmet demand estimate difference:",
    trained["unmet"] - baseline["unmet"]
)

print("\nEvaluation completed.")
