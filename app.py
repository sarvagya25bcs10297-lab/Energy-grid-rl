import gradio as gr
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import EnergyGridEnv
from huggingface_hub import InferenceClient
import os

# Initialize Hugging Face Client
# HF_TOKEN should be set in Hugging Face Space Secrets
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def run_simulation(battery_cap, demand_mult):
    # 1. Setup Env with Custom User Parameters
    env = EnergyGridEnv()
    env.battery_capacity = float(battery_cap)

    # Load the pre-trained model
    # In a real HF Space, ensure the model file is uploaded to the repo
    try:
        model = PPO.load("energy_grid_model")
    except:
        return "Error: Model file 'energy_grid_model.zip' not found. Please ensure it is uploaded with the app.", None, ""

    obs, _ = env.reset()
    done = False
    history = {"hour": [], "solar": [], "wind": [], "demand": [], "soc": [], "price": [], "action": []}
    total_cost = 0

    # Run 24 hour simulation
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # Store state
        hour, solar, wind, demand, soc, price = obs[0], obs[2], obs[3], obs[4], obs[5], obs[6]
        # Apply demand multiplier from user
        demand *= demand_mult

        obs, reward, done, truncated, _ = env.step(action)

        total_cost += (-reward)
        history["hour"].append(hour)
        history["solar"].append(solar)
        history["wind"].append(wind)
        history["demand"].append(demand)
        history["soc"].append(soc)
        history["price"].append(price)
        history["action"].append(action)

    df = pd.DataFrame(history)

    # 2. Generate Dashboard Graph
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df["hour"], df["solar"] + df["wind"], label="Renewables", color="green")
    plt.plot(df["hour"], df["demand"], label="Demand", color="red", linestyle="--")
    plt.title("Energy Balance")
    plt.legend(); plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(df["hour"], df["soc"], label="Battery SoC", color="blue")
    plt.title("Battery State")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plot_path = "sim_results.png"
    plt.savefig(plot_path)
    plt.close()

    # 3. AI Analysis via Hugging Face API
    analysis_prompt = (
        f"Act as a Smart Grid Expert. Analyze this RL agent performance: "
        f"Total Cost: ${total_cost:.2f}, Final Battery SoC: {df['soc'].iloc[-1]:.2f}kWh. "
        f"The agent managed a grid with {battery_cap}kWh capacity. "
        f"Provide a professional 2-sentence summary of the efficiency and a tip for improvement."
    )

    try:
        # Using a powerful open model like Mistral for analysis
        response = client.text_generation(analysis_prompt, model="mistralai/Mistral-7B-Instruct-v0.2", max_new_tokens=100)
        ai_report = response
    except Exception as e:
        ai_report = f"AI Analysis unavailable: {str(e)}. Total Cost: ${total_cost:.2f}"

    return f"Total Daily Cost: ${total_cost:.2f}", plot_path, ai_report

# Gradio UI Layout
with gr.Blocks(title="AI Energy Grid Balancer") as demo:
    gr.Markdown("# ⚡ AI Energy Grid Balancer")
    gr.Markdown("This app uses a **Deep Reinforcement Learning (PPO)** agent to balance renewable energy and demand. It utilizes the **Hugging Face Inference API** for expert analysis.")

    with gr.Row():
        with gr.Column():
            batt_input = gr.Slider(10, 100, value=50, label="Battery Capacity (kWh)")
            demand_input = gr.Slider(0.5, 2.0, value=1.0, label="Demand Multiplier")
            btn = gr.Button("Simulate & Analyze", variant="primary")

        with gr.Column():
            cost_out = gr.Textbox(label="Financial Result")
            ai_out = gr.Textbox(label="Hugging Face AI Expert Analysis")
            plot_out = gr.Image(label="Performance Dashboard")

    btn.click(run_simulation, inputs=[batt_input, demand_input], outputs=[cost_out, plot_out, ai_out])

# ─── OpenEnv API Endpoints ───────────────────────────────────────────
# Required by the hackathon submission checker.
# Gradio exposes its FastAPI app, so we mount /reset and /step on it.

from fastapi import Request
from fastapi.responses import JSONResponse

fastapi_app = demo.app

# Shared environment instance for OpenEnv API
openenv_env = EnergyGridEnv()


@fastapi_app.post("/reset")
async def openenv_reset(request: Request):
    try:
        body = await request.json()
        seed = body.get("seed", None) if body else None
    except Exception:
        seed = None
    observation, info = openenv_env.reset(seed=seed)
    return JSONResponse(content={
        "observation": observation.tolist(),
        "info": info
    })


@fastapi_app.post("/step")
async def openenv_step(request: Request):
    body = await request.json()
    action = int(body.get("action", 0))
    observation, reward, terminated, truncated, info = openenv_env.step(action)
    return JSONResponse(content={
        "observation": observation.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    })


@fastapi_app.get("/info")
async def openenv_info():
    return JSONResponse(content={
        "action_space": {"type": "Discrete", "n": openenv_env.action_space.n},
        "observation_space": {
            "type": "Box",
            "shape": list(openenv_env.observation_space.shape),
            "low": openenv_env.observation_space.low.tolist(),
            "high": openenv_env.observation_space.high.tolist()
        }
    })


if __name__ == "__main__":
    demo.launch()
