
import pandas as pd
import numpy as np
from pathlib import Path


class TaxiEnvironment:
    """
    TaxiFlow fleet-level repositioning environment.

    Actions:
        0 = Keep the current allocation
        1 = Move one taxi from a low-demand zone
            to the highest-demand zone
        2 = Move one taxi from the lowest-demand-per-taxi
            zone to the highest-demand-per-taxi zone

    One action represents one 15-minute interval.

    IMPORTANT:
    This is a simplified simulation, not a real-time
    taxi dispatch system.
    """

    def __init__(
        self,
        data_path="data/demand_by_zone_15min.csv",
        total_taxis=100,
        reposition_cost=2.0,
        unmet_demand_penalty=5.0,
        seed=42
    ):
        self.data_path = Path(data_path)

        if not self.data_path.is_absolute():
            self.data_path = (
                Path(__file__).resolve().parent
                / self.data_path
            )

        self.total_taxis = total_taxis
        self.reposition_cost = reposition_cost
        self.unmet_demand_penalty = unmet_demand_penalty
        self.rng = np.random.default_rng(seed)

        self.df = pd.read_csv(
            self.data_path,
            parse_dates=["time_interval"]
        )

        self.df = self.df.sort_values(
            ["time_interval", "zone_id"]
        ).reset_index(drop=True)

        self.zones = sorted(
            self.df["zone_id"].unique()
        )

        self.num_zones = len(self.zones)

        self.time_steps = sorted(
            self.df["time_interval"].unique()
        )

        self.demand_matrix = (
            self.df.pivot(
                index="time_interval",
                columns="zone_id",
                values="pickup_count"
            )
            .reindex(
                index=self.time_steps,
                columns=self.zones,
                fill_value=0
            )
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )

        self.num_actions = 3
        self.current_step = 0
        self.taxi_allocation = None

    def reset(self, start_step=0):
        """Reset the simulation and distribute the fleet."""

        if not 0 <= start_step < len(self.time_steps):
            raise ValueError(
                "start_step must be within the available "
                "time intervals."
            )

        self.current_step = start_step

        # Distribute taxis as evenly as possible.
        base = self.total_taxis // self.num_zones
        remainder = self.total_taxis % self.num_zones

        self.taxi_allocation = np.full(
            self.num_zones,
            base,
            dtype=int
        )

        self.taxi_allocation[:remainder] += 1

        return self._get_state()

    def _get_state(self):
        """
        State:
        (overall_demand_level, coverage_level, time_period)

        Overall demand has four levels.
        Coverage has three levels.
        Time has four periods.
        """

        demand = self.demand_matrix[self.current_step]
        total_demand = float(demand.sum())

        # Categorize total demand during this interval.
        demand_level = int(
            np.digitize(
                total_demand,
                [250, 500, 1000]
            )
        )

        # Find the zone with the highest current demand.
        highest_demand_zone = int(np.argmax(demand))

        # Compare its taxi allocation with average allocation.
        average_allocation = (
            self.total_taxis / self.num_zones
        )

        coverage_ratio = (
            self.taxi_allocation[highest_demand_zone]
            / max(average_allocation, 1e-6)
        )

        coverage_level = int(
            np.digitize(
                coverage_ratio,
                [0.75, 1.5]
            )
        )

        hour = pd.Timestamp(
            self.time_steps[self.current_step]
        ).hour

        if hour < 6:
            time_period = 0
        elif hour < 12:
            time_period = 1
        elif hour < 18:
            time_period = 2
        else:
            time_period = 3

        return (
            demand_level,
            coverage_level,
            time_period
        )

    def _calculate_performance(self):
        """
        Estimate pickups served and unmet demand.

        Assumption: each allocated taxi can serve at most
        one pickup during a 15-minute interval.
        """

        demand = self.demand_matrix[self.current_step]

        served = np.minimum(
            demand,
            self.taxi_allocation
        )

        unmet = np.maximum(
            demand - self.taxi_allocation,
            0
        )

        return int(served.sum()), int(unmet.sum())

    def _reposition_taxi(self, action, demand):
        """Apply one of the two repositioning strategies."""

        donor_candidates = np.where(
            self.taxi_allocation > 0
        )[0]

        if len(donor_candidates) == 0:
            return 0.0, False

        if action == 1:
            # Choose a low-demand donor zone.
            donor_zone = min(
                donor_candidates,
                key=lambda z: demand[z]
            )

            # Target the zone with the highest demand.
            target_zone = int(np.argmax(demand))

        else:
            # Compare demand with the number of taxis
            # allocated to each zone.
            demand_per_taxi = demand / np.maximum(
                self.taxi_allocation,
                1
            )

            # Choose a donor with low demand per taxi.
            donor_zone = min(
                donor_candidates,
                key=lambda z: demand_per_taxi[z]
            )

            # Choose the zone with the highest demand
            # relative to its current taxi allocation.
            target_zone = int(np.argmax(demand_per_taxi))

        # Do not move a taxi if the donor and target
        # are the same zone.
        if donor_zone == target_zone:
            return 0.0, False

        self.taxi_allocation[donor_zone] -= 1
        self.taxi_allocation[target_zone] += 1

        return self.reposition_cost, True

    def step(self, action):
        """
        Execute one action and advance one interval.

        Returns:
            next_state, reward, done, info
        """

        if self.taxi_allocation is None:
            raise RuntimeError(
                "Call reset() before step()."
            )

        if action not in (0, 1, 2):
            raise ValueError(
                "Action must be 0, 1, or 2."
            )

        if self.current_step >= len(self.time_steps):
            raise RuntimeError(
                "Episode is finished. Call reset()."
            )

        demand = self.demand_matrix[self.current_step]

        movement_cost = 0.0
        moved_taxi = False

        if action in (1, 2):
            movement_cost, moved_taxi = (
                self._reposition_taxi(action, demand)
            )

        served, unmet = self._calculate_performance()

        reward = (
            served
            - self.unmet_demand_penalty * unmet
            - movement_cost
        )

        # Ensure repositioning never changes fleet size.
        assert int(self.taxi_allocation.sum()) == self.total_taxis

        evaluated_interval = self.time_steps[self.current_step]

        self.current_step += 1

        # Process all available intervals, including the last.
        done = self.current_step >= len(self.time_steps)

        next_state = (
            None if done else self._get_state()
        )

        info = {
            "time_interval": str(evaluated_interval),
            "served_pickups_estimate": served,
            "unmet_demand_estimate": unmet,
            "reposition_cost": movement_cost,
            "moved_taxi": moved_taxi,
            "taxis_allocated": int(
                self.taxi_allocation.sum()
            )
        }

        return next_state, float(reward), done, info


if __name__ == "__main__":
    env = TaxiEnvironment()

    state = env.reset()

    print("Environment initialized successfully.")
    print("Number of zones:", env.num_zones)
    print("Number of time steps:", len(env.time_steps))
    print("Total taxis:", env.total_taxis)
    print("Number of actions:", env.num_actions)
    print("Initial state:", state)

    initial_step = env.current_step

    next_state, reward, done, info = env.step(0)

    print("\nOne-step test")
    print("Reward:", reward)
    print("Info:", info)
    print("Time steps advanced:", env.current_step - initial_step)
    print("Next state:", next_state)