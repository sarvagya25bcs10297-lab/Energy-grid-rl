import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import sys

# Add the energy_grid_rl directory to path so we can import env
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "energy_grid_rl"))

from env import EnergyGridEnv

app = FastAPI(title="AI Energy Grid Balancer - OpenEnv API")

# Global environment instance
env = EnergyGridEnv()


class ResetRequest(BaseModel):
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: int


class ResetResponse(BaseModel):
    observation: list
    info: dict


class StepResponse(BaseModel):
    observation: list
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@app.get("/")
def health():
    return {"status": "ok", "environment": "EnergyGridEnv"}


@app.get("/info")
def env_info():
    return {
        "action_space": {
            "type": "Discrete",
            "n": env.action_space.n
        },
        "observation_space": {
            "type": "Box",
            "shape": list(env.observation_space.shape),
            "low": env.observation_space.low.tolist(),
            "high": env.observation_space.high.tolist()
        }
    }


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    observation, info = env.reset(seed=request.seed)
    return ResetResponse(
        observation=observation.tolist(),
        info=info
    )


@app.post("/step")
def step(request: StepRequest):
    observation, reward, terminated, truncated, info = env.step(request.action)
    return StepResponse(
        observation=observation.tolist(),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=info
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
