import sys
import os
import traceback
# v1.0.3 - Fixed dependency issues and robustified environment handling
import json
import urllib.request
import urllib.error
import numpy as np
from stable_baselines3 import PPO

# ── Configuration ──────────────────────────────────────────────────
TASK_NAME = "energy_grid_balancing"
# Support multiple possible env var names for the environment endpoint
ENV_API_URL = os.getenv("ENV_API_URL") or os.getenv("ENVIRONMENT_URL") or "http://localhost:7860"

# Ensure current directory is in path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Remote Environment Wrapper ─────────────────────────────────────
class RemoteEnergyGridEnv:
    """A wrapper that communicates with the EnergyGridEnv via HTTP API using urllib."""
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def _post(self, endpoint, data):
        url = f"{self.base_url}/{endpoint}"
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def reset(self, seed=None):
        try:
            data = self._post("reset", {"seed": seed})
            return np.array(data["observation"], dtype=np.float32), data.get("info", {})
        except Exception as e:
            raise RuntimeError(f"Failed to reset remote environment at {self.base_url}: {e}")

    def step(self, action):
        try:
            data = self._post("step", {"action": int(action)})
            # Handle both 'done' (old gym) and 'terminated'/'truncated' (gymnasium)
            terminated = data.get("terminated", data.get("done", False))
            truncated = data.get("truncated", False)
            return (
                np.array(data["observation"], dtype=np.float32),
                data["reward"],
                terminated,
                truncated,
                data.get("info", {})
            )
        except Exception as e:
            raise RuntimeError(f"Failed to step remote environment at {self.base_url}: {e}")


def main():
    # Emit [START] at the very beginning of the evaluation process
    print(f"[START] task={TASK_NAME}", flush=True)
    
    # ── 1. Determine Environment ─────────────────────────────────────
    print(f"[INFO] Initializing inference for task: {TASK_NAME}", file=sys.stderr)
    
    # Check if a remote environment is available/requested
    use_remote = False
    try:
        url = f"{ENV_API_URL}/info"
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                print(f"[INFO] Using remote environment at {ENV_API_URL}", file=sys.stderr)
                env = RemoteEnergyGridEnv(ENV_API_URL)
                use_remote = True
    except:
        pass

    if not use_remote:
        print("[INFO] Falling back to local EnergyGridEnv.", file=sys.stderr)
        try:
            from env import EnergyGridEnv
            env = EnergyGridEnv()
        except ImportError:
            try:
                from energy_grid_rl.env import EnergyGridEnv
                env = EnergyGridEnv()
            except ImportError as e:
                print(f"[ERROR] Could not import local EnergyGridEnv: {e}", file=sys.stderr)
                sys.exit(1)

    # ── 2. Load Model ────────────────────────────────────────────────
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "energy_grid_model")
    try:
        model = PPO.load(model_path)
        print(f"[INFO] Model loaded from {model_path}", file=sys.stderr)
    except Exception as exc:
        print(f"[ERROR] Could not load PPO model: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # ── 3. Episode Loop ──────────────────────────────────────────────
    try:
        obs, _ = env.reset()
        done = False
        step_num = 0
        total_reward = 0.0
        action_names = ["charge", "discharge", "sell", "idle"]

        while not done:
            step_num += 1
            action, _ = model.predict(obs, deterministic=True)
            
            # Extract scalar action
            if isinstance(action, (np.ndarray, list)):
                action_val = int(action.flatten()[0])
            else:
                action_val = int(action)

            obs, reward, terminated, truncated, info = env.step(action_val)
            done = terminated or truncated
            total_reward += reward

            action_label = action_names[action_val]
            # Emit [STEP] tag
            print(
                f"[STEP] step={step_num} action={action_label} reward={reward:.4f}",
                flush=True,
            )

        # ── 4. Final Score ───────────────────────────────────────────
        max_possible_penalty = 24 * 50
        score = float(np.clip(1.0 + total_reward / max_possible_penalty, 0.0, 1.0))

        # Emit [END] tag
        print(
            f"[END] task={TASK_NAME} score={score:.4f} steps={step_num}",
            flush=True,
        )

    except Exception:
        print("[ERROR] Unhandled exception during episode execution:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)




if __name__ == "__main__":
    main()


