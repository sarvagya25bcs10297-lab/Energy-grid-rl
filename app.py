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

# Initialize Environment for API
api_env = EnergyGridEnv()

# Initialize Hugging Face Client
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def run_simulation(battery_cap, demand_mult):
    sim_env = EnergyGridEnv()
    sim_env.battery_capacity = float(battery_cap)
    
    try:
        model = PPO.load("energy_grid_agent")
    except Exception as e:
        return f"Error: Model loading failed ({str(e)})", None, "Ensure energy_grid_agent.zip is in root."

    obs, _ = sim_env.reset()
    done = False
    history = {"hour": [], "solar": [], "wind": [], "demand": [], "soc": [], "price": [], "action": []}
    total_cost = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Fix indexing: [sin_h, cos_h, solar, wind, demand, soc, price, delta]
        hour = sim_env.current_hour
        solar, wind, demand, soc, price = obs[2], obs[3], obs[4], obs[5], obs[6]
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

    prompt = f"Expert analysis: Total Cost ${total_cost:.2f}, Battery {battery_cap}kWh. Provide 2 sentences."
    try:
        response = client.chat_completion(model="meta-llama/Llama-3.2-1B-Instruct", 
                                        messages=[{"role": "user", "content": prompt}], max_tokens=100)
        ai_report = response.choices[0].message.content
    except:
        ai_report = f"Agent finished with cost ${total_cost:.2f}."

    return f"Total Daily Cost: ${total_cost:.2f}", plot_path, ai_report

# --- UI ---
with gr.Blocks(title="AI Energy Grid Balancer") as demo:
    gr.Markdown("# ⚡ AI Energy Grid Balancer")
    with gr.Row():
        with gr.Column():
            batt_input = gr.Slider(10, 100, value=50, label="Battery Capacity (kWh)")
            demand_input = gr.Slider(0.5, 2.0, value=1.0, label="Demand Multiplier")
            btn = gr.Button("Simulate & Analyze", variant="primary")
        with gr.Column():
            cost_out = gr.Textbox(label="Result")
            ai_out = gr.Textbox(label="AI Insight")
            plot_out = gr.Image(label="Chart")
    btn.click(run_simulation, inputs=[batt_input, demand_input], outputs=[cost_out, plot_out, ai_out])

# --- API ---
main_app = FastAPI()

@main_app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
        seed = body.get("seed")
    except:
        seed = None
    observation, info = api_env.reset(seed=seed)
    return JSONResponse({"observation": observation.tolist(), "info": info})

@main_app.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
        action = int(body.get("action", 0))
    except:
        action = 0
    observation, reward, terminated, truncated, info = api_env.step(action)
    return JSONResponse({
        "observation": observation.tolist(), "reward": float(reward),
        "terminated": bool(terminated), "truncated": bool(truncated), "info": info
    })

@main_app.get("/info")
async def info():
    return {"action_space": api_env.action_space.n, "observation_space": api_env.observation_space.shape}

# Mount Gradio at root
# IMPORTANT: This must be the last step
app = gr.mount_gradio_app(main_app, demo, path="/")

def main():
    import uvicorn
    # The 'app' object is the FastAPI instance with Gradio mounted
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
