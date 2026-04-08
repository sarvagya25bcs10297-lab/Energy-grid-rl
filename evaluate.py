import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import EnergyGridEnv

def evaluate():
    # Initialize environment and load model
    env = EnergyGridEnv()
    model = PPO.load("energy_grid_agent")

    obs, _ = env.reset()
    done = False

    history = {
        "hour": [],
        "solar": [],
        "wind": [],
        "demand": [],
        "soc": [],
        "price": [],
        "action": [],
        "cost": []
    }

    total_cost = 0

    print("Running evaluation for 24 hours...")

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # Store current state before step
        hour = obs[0]
        solar = obs[1]
        wind = obs[2]
        demand = obs[3]
        soc = obs[4]
        price = obs[5]

        obs, reward, done, truncated, _ = env.step(action)

        # Cost is -reward (since reward = -cost)
        step_cost = -reward
        total_cost += step_cost

        history["hour"].append(hour)
        history["solar"].append(solar)
        history["wind"].append(wind)
        history["demand"].append(demand)
        history["soc"].append(soc)
        history["price"].append(price)
        history["action"].append(action)
        history["cost"].append(step_cost)

    df = pd.DataFrame(history)

    # Visualizations
    plt.figure(figsize=(15, 10))

    # 1. Power Balance
    plt.subplot(3, 1, 1)
    plt.plot(df["hour"], df["solar"] + df["wind"], label="Renewables", color="green")
    plt.plot(df["hour"], df["demand"], label="Demand", color="red", linestyle="--")
    plt.title("Energy Balance (Production vs Demand)")
    plt.ylabel("kW")
    plt.legend()
    plt.grid(True)

    # 2. Battery State of Charge
    plt.subplot(3, 1, 2)
    plt.plot(df["hour"], df["soc"], label="Battery SoC", color="blue")
    plt.title("Battery State of Charge")
    plt.ylabel("kWh")
    plt.legend()
    plt.grid(True)

    # 3. Actions and Prices
    plt.subplot(3, 1, 3)
    plt.plot(df["hour"], df["price"], label="Price", color="orange")
    plt.step(df["hour"], df["action"], label="Action", color="purple", where="post")
    plt.title("Market Price and Agent Actions")
    plt.xlabel("Hour of Day")
    plt.ylabel("Price / Action ID")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    print("Evaluation complete. Results saved to evaluation_results.png")
    print(f"Total cost for the day: ${total_cost:.2f}")

if __name__ == "__main__":
    evaluate()
