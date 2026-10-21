
import random
import numpy as np
from collections import defaultdict
from taxi_environment import TaxiEnvironment

# Settings
EPISODES = 100
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0

random.seed(42)
np.random.seed(42)

# Create environment and Q-table
env = TaxiEnvironment()
q_table = defaultdict(lambda: np.zeros(3))
rewards = []

# Train the agent
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Choose an action
        if random.random() < EPSILON:
            action = random.randrange(3)
        else:
            action = int(np.argmax(q_table[state]))

        next_state, reward, done, info = env.step(action)

        # Update Q-value
        if done:
            future_reward = 0
        else:
            future_reward = np.max(q_table[next_state])

        q_table[state][action] += ALPHA * (
            reward + GAMMA * future_reward - q_table[state][action]
        )

        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    EPSILON = max(0.05, EPSILON * 0.97)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

# Save the trained model
states = list(q_table.keys())
q_values = np.array([q_table[s] for s in states])

np.savez(
    "q_learning_model.npz",
    states=np.array(states, dtype=np.int32),
    q_values=q_values
)

# Save training rewards
np.savetxt(
    "training_rewards.csv",
    rewards,
    delimiter=",",
    header="total_reward",
    comments=""
)

print("Training completed.")
print("Learned states:", len(states))
print("Model saved to q_learning_model.npz")