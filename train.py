import gymnasium as gym
from stable_baselines3 import PPO
from env import EnergyGridEnv
import os

def train():
    # Initialize environment
    env = EnergyGridEnv()

    # Initialize PPO agent
    # MlpPolicy is used because the observation space is a simple vector
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0001,
        gamma=0.999,
        n_steps=2048,
        batch_size=64
    )

    print("Starting training...")
    # Train for a reasonable number of timesteps for a hackathon project
    # 24 hours * 2000 episodes = 48,000 steps. Let's do 100k for better convergence.
    model.learn(total_timesteps=100000)

    # Save the model
    model_path = "energy_grid_model"
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
