
import random
import numpy as np
from collections import defaultdict

from taxi_environment import TaxiEnvironment


# -----------------------------
# 1. Configuration
# -----------------------------
EPISODES = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.97

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# -----------------------------
# 2. Initialize environment
# -----------------------------
env = TaxiEnvironment(
    data_path="data/demand_by_zone_15min.csv",
    total_taxis=100
)

# Q-table:
# Each state maps to the estimated value of 3 actions.
q_table = defaultdict(lambda: np.zeros(env.num_actions))


# -----------------------------
# 3. Train Q-learning agent
# -----------------------------
episode_rewards = []

epsilon = EPSILON

print("Starting Q-learning training...")

for episode in range(EPISODES):

    # Start from the beginning of the dataset.
    state = env.reset(start_step=0)

    total_reward = 0.0
    done = False

    while not done:

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randrange(env.num_actions)
        else:
            action = int(np.argmax(q_table[state]))

        next_state, reward, done, info = env.step(action)

        # Q-learning update
        old_value = q_table[state][action]

        if done or next_state is None:
            next_max = 0.0
        else:
            next_max = float(np.max(q_table[next_state]))

        q_table[state][action] = old_value + LEARNING_RATE * (
            reward
            + DISCOUNT_FACTOR * next_max
            - old_value
        )

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

    epsilon = max(
        EPSILON_MIN,
        epsilon * EPSILON_DECAY
    )

    if (episode + 1) % 10 == 0:
        recent_average = np.mean(
            episode_rewards[-10:]
        )

        print(
            f"Episode {episode + 1}/{EPISODES} | "
            f"Reward: {total_reward:.2f} | "
            f"Last 10 average: {recent_average:.2f} | "
            f"Epsilon: {epsilon:.3f}"
        )


# -----------------------------
# 4. Save trained Q-table
# -----------------------------
states = list(q_table.keys())

q_values = np.array(
    [q_table[state] for state in states],
    dtype=np.float64
)

np.savez(
    "q_learning_model.npz",
    states=np.array(states, dtype=np.int32),
    q_values=q_values
)

print("\nTraining completed.")
print("Learned states:", len(states))
print("Model saved to q_learning_model.npz")


# -----------------------------
# 5. Save episode rewards
# -----------------------------
np.savetxt(
    "training_rewards.csv",
    np.array(episode_rewards),
    delimiter=",",
    header="total_reward",
    comments=""
)

print("Episode rewards saved to training_rewards.csv")
