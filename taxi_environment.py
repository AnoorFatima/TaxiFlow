
import pandas as pd
import numpy as np
from pathlib import Path

class TaxiEnvironment:

    def __init__(self, data_path="data/demand_by_zone_15min.csv"):
        data_path = Path(__file__).resolve().parent / data_path
        self.df = pd.read_csv(data_path, parse_dates=["time_interval"])

        self.zones = sorted(self.df["zone_id"].unique())
        self.times = sorted(self.df["time_interval"].unique())

        self.demand = (
            self.df.pivot_table(
                index="time_interval",
                columns="zone_id",
                values="pickup_count",
                aggfunc="sum",
                fill_value=0
            )
            .reindex(index=self.times, columns=self.zones, fill_value=0)
            .to_numpy()
        )

        self.total_taxis = 100
        self.reposition_cost = 2
        self.unmet_penalty = 5
        self.num_actions = 3

    def reset(self, start_step=0):
        self.current_step = start_step

        # Distribute taxis equally among zones
        base = self.total_taxis // len(self.zones)
        self.taxis = np.full(len(self.zones), base, dtype=int)
        self.taxis[:self.total_taxis % len(self.zones)] += 1

        return self._get_state()

    def _get_state(self):
        demand = self.demand[self.current_step]
        total_demand = demand.sum()

        demand_level = int(np.digitize(total_demand, [250, 500, 1000]))

        busiest_zone = np.argmax(demand)
        average_taxis = self.total_taxis / len(self.zones)
        coverage = self.taxis[busiest_zone] / average_taxis
        coverage_level = int(np.digitize(coverage, [0.75, 1.5]))

        hour = pd.Timestamp(self.times[self.current_step]).hour
        time_period = hour // 6

        return demand_level, coverage_level, time_period

    def step(self, action):
        if action not in (0, 1, 2):
            raise ValueError("Action must be 0, 1, or 2.")

        demand = self.demand[self.current_step]
        movement_cost = 0

        # Action 1 or 2 moves one taxi
        if action in (1, 2):
            donors = np.where(self.taxis > 0)[0]

            if action == 1:
                donor = min(donors, key=lambda z: demand[z])
                target = np.argmax(demand)
            else:
                demand_per_taxi = demand / np.maximum(self.taxis, 1)
                donor = min(donors, key=lambda z: demand_per_taxi[z])
                target = np.argmax(demand_per_taxi)

            if donor != target:
                self.taxis[donor] -= 1
                self.taxis[target] += 1
                movement_cost = self.reposition_cost

        # Calculate served and unmet pickups
        served = np.minimum(demand, self.taxis).sum()
        unmet = np.maximum(demand - self.taxis, 0).sum()

        # Calculate reward
        reward = served-self.unmet_penalty * unmet - movement_cost

        info = {
            "time_interval": str(self.times[self.current_step]),
            "served_pickups_estimate": int(served),
            "unmet_demand_estimate": int(unmet),
            "reposition_cost": movement_cost,
            "taxis_allocated": int(self.taxis.sum())
        }

        self.current_step += 1
        done = self.current_step >= len(self.times)

        next_state = None if done else self._get_state()

        return next_state, float(reward), done, info


if __name__ == "__main__":
    env = TaxiEnvironment()
    state = env.reset()

    print("Environment initialized successfully.")
    print("Zones:", len(env.zones))
    print("Time intervals:", len(env.times))
    print("Initial state:", state)

    next_state, reward, done, info = env.step(0)

    print("Reward:", reward)
    print("Result:", info)
