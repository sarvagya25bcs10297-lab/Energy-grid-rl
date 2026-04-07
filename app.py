import os
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import EnergyGridEnv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from huggingface_hub import InferenceClient

# Initialize Environment
# We use separate instances for UI and API to avoid state interference
api_env = EnergyGridEnv()

# Initialize Hugging Face Client
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def run_simulation(battery_cap, demand_mult):
    # Setup Env with Custom User Parameters
    sim_env = EnergyGridEnv()
    sim_env.battery_capacity = float(battery_cap)

    # Load the pre-trained model
    try:
        model = PPO.load("energy_grid_model")
    except Exception as e:
        return f"Error: Model loading failed ({str(e)}). Please ensure energy_grid_model.zip exists.", None, ""

    obs, _ = sim_env.reset()
    done = False
    history = {
        "hour": [], "solar": [], "wind": [], "demand": [], 
        "soc": [], "price": [], "action": []
    }
    total_cost = 0

    # Run 24 hour simulation
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # Correct Observation Mapping:
        # [0:sin_h, 1:cos_h, 2:solar, 3:wind, 4:demand, 5:soc, 6:price, 7:price_delta]
        hour = sim_env.current_hour
        solar, wind, demand, soc, price = obs[2], obs[3], obs[4], obs[5], obs[6]
        
        # Apply demand multiplier from user
        demand *= demand_mult

        obs, reward, done, truncated, _ = sim_env.step(action)

        total_cost += (-reward)
        history["hour"].append(hour)
        history["solar"].append(solar)
        history["wind"].append(wind)
        history["demand"].append(demand)
        history["soc"].append(soc)
        history["price"].append(price)
        history["action"].append(action)

    df = pd.DataFrame(history)

    # Generate Graphs
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

    # AI Analysis
    analysis_prompt = (
        f"Act as a Smart Grid Expert. Analyze this RL agent performance: "
        f"Total Cost: ${total_cost:.2f}, Final Battery SoC: {df['soc'].iloc[-1]:.2f}kWh. "
        f"The agent managed a grid with {battery_cap}kWh capacity. "
        f"Provide a 2-sentence professional insight."
    )

    try:
        response = client.chat_completion(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=150
        )
        ai_report = response.choices[0].message.content
    except Exception as e:
        ai_report = f"AI Analysis Result: Agent completed with total cost ${total_cost:.2f}."

    return f"Total Daily Cost: ${total_cost:.2f}", plot_path, ai_report

# --- Gradio UI ---
with gr.Blocks(title="AI Energy Grid Balancer") as demo:
    gr.Markdown("# ⚡ AI Energy Grid Balancer")
    gr.Markdown("Deep Reinforcement Learning (PPO) agent balancing renewable energy and demand.")

    with gr.Row():
        with gr.Column():
            batt_input = gr.Slider(10, 100, value=50, label="Battery Capacity (kWh)")
            demand_input = gr.Slider(0.5, 2.0, value=1.0, label="Demand Multiplier")
            btn = gr.Button("Simulate & Analyze", variant="primary")

        with gr.Column():
            cost_out = gr.Textbox(label="Financial Result")
            ai_out = gr.Textbox(label="AI Expert Analysis")
            plot_out = gr.Image(label="Performance Dashboard")

    btn.click(run_simulation, inputs=[batt_input, demand_input], outputs=[cost_out, plot_out, ai_out])

# --- OpenEnv API (FastAPI) ---
app = FastAPI()

@app.post("/reset")
async def openenv_reset(request: Request):
    try:
        body = await request.json()
        seed = body.get("seed")
    except:
        seed = None
    observation, info = api_env.reset(seed=seed)
    return JSONResponse(content={
        "observation": observation.tolist(),
        "info": info
    })

@app.post("/step")
async def openenv_step(request: Request):
    try:
        body = await request.json()
        action = int(body.get("action", 0))
    except:
        action = 0
    observation, reward, terminated, truncated, info = api_env.step(action)
    return JSONResponse(content={
        "observation": observation.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    })

@app.get("/info")
async def openenv_info():
    return JSONResponse(content={
        "action_space": {"type": "Discrete", "n": api_env.action_space.n},
        "observation_space": {
            "type": "Box",
            "shape": list(api_env.observation_space.shape),
            "low": api_env.observation_space.low.tolist(),
            "high": api_env.observation_space.high.tolist()
        }
    })

# Mount Gradio to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
