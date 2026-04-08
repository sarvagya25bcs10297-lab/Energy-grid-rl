"""
inference.py – OpenEnv-compatible inference entry point.

Loads the trained PPO model, runs it on the EnergyGridEnv for one
24-hour episode, and emits the required [START]/[STEP]/[END] structured
output to stdout so the OpenEnv validator can parse it.
"""

import sys
import numpy as np
from stable_baselines3 import PPO
from env import EnergyGridEnv

TASK_NAME = "energy_grid_balancing"


def main():
    # ── Load environment & trained agent ──────────────────────────────
    env = EnergyGridEnv()
    try:
        model = PPO.load("energy_grid_model")
    except Exception as exc:
        print(f"[ERROR] Could not load model: {exc}", flush=True)
        sys.exit(1)

    # ── Episode loop ──────────────────────────────────────────────────
    obs, _ = env.reset()
    done = False
    step_num = 0
    total_reward = 0.0
    action_names = ["charge", "discharge", "sell", "idle"]

    print(f"[START] task={TASK_NAME}", flush=True)

    while not done:
        step_num += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward

        action_label = action_names[int(action)]
        print(
            f"[STEP] step={step_num} action={action_label} reward={reward:.4f}",
            flush=True,
        )

        if truncated:
            break

    # ── Final score (normalised to 0-1 range) ────────────────────────
    # Reward is always negative (cost + penalty); a perfect score ≈ 0.
    # We clamp the normalised value into [0, 1].
    max_possible_penalty = 24 * 50  # rough upper bound per episode
    score = float(np.clip(1.0 + total_reward / max_possible_penalty, 0.0, 1.0))

    print(
        f"[END] task={TASK_NAME} score={score:.4f} steps={step_num}",
        flush=True,
    )


if __name__ == "__main__":
    main()
