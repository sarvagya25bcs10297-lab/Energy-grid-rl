import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class EnergyGridEnv(gym.Env):
    
    def __init__(self):
        super(EnergyGridEnv, self).__init__()

        # Action Space:
        # 0: Charge Battery (from renewables/grid)
        # 1: Discharge Battery (to meet demand)
        # 2: Sell Excess to Grid
        # 3: Do Nothing/Idle
        self.action_space = spaces.Discrete(4)

        # Observation Space:
        # 0: Hour of day (sin)
        # 1: Hour of day (cos)
        # 2: Solar Generation (kW)
        # 3: Wind Generation (kW)
        # 4: Current Demand (kW)
        # 5: Battery State of Charge (kWh)
        # 6: Grid Price ($/kWh)
        # 7: Price Delta (Current - Previous)
        self.observation_space = spaces.Box(
            low=-1, high=100, shape=(8,), dtype=np.float32
        )

        # Constants
        self.battery_capacity = 50.0  # kWh
        self.max_charge_rate = 10.0   # kW
        self.max_discharge_rate = 10.0 # kW
        self.efficiency = 0.9         # charging/discharging efficiency

    def _get_environmental_data(self, hour):
       
        # Solar: Bell curve peaking at 12pm
        solar = max(0, 15 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0

        # Wind: More volatile, slightly higher at night
        wind = 5 + 3 * np.sin(np.pi * hour / 12) + np.random.normal(0, 1)
        wind = max(0, wind)

        # Demand: Peak in morning (8am) and evening (7pm)
        demand = 8 + 4 * np.sin(np.pi * (hour - 8) / 12) + 4 * np.sin(np.pi * (hour - 19) / 12)
        demand = max(0, demand + np.random.normal(0, 0.5))

        # Price: Peak hours are more expensive
        price = 0.15 + 0.10 * (1 if (8 <= hour <= 10 or 18 <= hour <= 22) else 0)

        return solar, wind, demand, price

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_hour = 0
        self.battery_soc = self.battery_capacity * 0.5  # Start at 50%
        self.prev_price = 0.15

        solar, wind, demand, price = self._get_environmental_data(self.current_hour)

        obs = np.array([
            np.sin(2 * np.pi * self.current_hour / 24),
            np.cos(2 * np.pi * self.current_hour / 24),
            solar,
            wind,
            demand,
            self.battery_soc,
            price,
            price - self.prev_price
        ], dtype=np.float32)

        return obs, {}

    def step(self, action):
        solar, wind, demand, price = self._get_environmental_data(self.current_hour)
        self.prev_price = price # Update for next delta

        # Net renewable energy available
        net_renewable = solar + wind - demand

        cost = 0
        penalty = 0

        # Logic for actions
        if action == 0: # Charge
            charge_amount = min(self.max_charge_rate, self.battery_capacity - self.battery_soc)
            # If net_renewable is negative, we buy from grid to charge
            energy_needed = charge_amount - max(0, net_renewable)
            if energy_needed > 0:
                cost += energy_needed * price

            self.battery_soc += charge_amount * self.efficiency

            # Professional Reward Shaping: Reward charging during high renewables
            if net_renewable > 0:
                penalty -= 0.5 # Bonus for utilizing free energy

        elif action == 1: # Discharge
            discharge_amount = min(self.max_discharge_rate, self.battery_soc)
            self.battery_soc -= discharge_amount / self.efficiency
            # Discharge helps satisfy demand
            net_renewable += discharge_amount

        elif action == 2: # Sell Excess
            if net_renewable > 0:
                cost -= net_renewable * (price * 0.8) # Sell at a discount
            else:
                # Trying to sell when there is a deficit is heavily penalized
                penalty += abs(net_renewable) * price * 2

        elif action == 3: # Idle
            pass

        # Handle final balance after action
        # If net_renewable is still negative, we MUST buy from grid to prevent blackout
        if net_renewable < 0:
            # Professional Reward Shaping: Huge penalty for "blackouts" (unmet demand)
            penalty += abs(net_renewable) * price * 5
            cost += abs(net_renewable) * price

        # Battery constraints
        self.battery_soc = np.clip(self.battery_soc, 0, self.battery_capacity)

        # Reward function: minimize cost and penalties
        reward = -(cost + penalty)

        # Penalty for extreme battery levels (Battery Health)
        if self.battery_soc < (self.battery_capacity * 0.1) or self.battery_soc > (self.battery_capacity * 0.9):
            reward -= 2.0

        self.current_hour += 1
        done = self.current_hour >= 24
        truncated = False

        # Prepare next observation (must match the 8-dim space)
        next_hour = self.current_hour % 24
        next_solar, next_wind, next_demand, next_price = self._get_environmental_data(next_hour)

        obs = np.array([
            np.sin(2 * np.pi * next_hour / 24),
            np.cos(2 * np.pi * next_hour / 24),
            next_solar,
            next_wind,
            next_demand,
            self.battery_soc,
            next_price,
            next_price - price
        ], dtype=np.float32)

        return obs, reward, done, truncated, {}
