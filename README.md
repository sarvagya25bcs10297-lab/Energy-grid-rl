---
title: AI Energy Grid Balancer
emoji: ⚡
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: false
---

# ⚡ AI Energy Grid Balancer

> A Deep Reinforcement Learning system that intelligently manages battery storage to balance renewable energy production and consumer demand — minimizing costs from the main grid in real time.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Parzival7498/Energy-Grid-Rl)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-PPO-orange)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Custom%20Env-brightgreen)](https://gymnasium.farama.org/)

---

## 🎯 Problem Statement

Modern power grids face a critical challenge: **renewable energy sources (solar & wind) are intermittent**, while consumer demand fluctuates throughout the day. Without intelligent management, excess renewable energy is wasted, and peak demand forces expensive grid purchases.

This project trains a **PPO (Proximal Policy Optimization)** agent to operate a battery storage system that:
- ☀️ **Charges** when renewables are abundant and prices are low
- 🔋 **Discharges** to meet demand during expensive peak hours
- 💰 **Sells excess** energy back to the grid when profitable
- ⏸️ **Idles** when no action yields better returns

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  Gradio Web UI                  │
│         (Interactive Dashboard + Controls)       │
├─────────────────────────────────────────────────┤
│               PPO Agent (SB3)                   │
│          Trained Policy Network (MLP)            │
├─────────────────────────────────────────────────┤
│           EnergyGridEnv (Gymnasium)              │
│  ┌───────────┬───────────┬──────────┬─────────┐ │
│  │   Solar   │   Wind    │  Demand  │  Price  │ │
│  │Generation │Generation │  Curve   │ Signal  │ │
│  └───────────┴───────────┴──────────┴─────────┘ │
├─────────────────────────────────────────────────┤
│         HF Inference API (LLM Analysis)          │
│        Meta Llama 3.2 / Mistral 7B Expert        │
└─────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
energy_grid_rl/
├── energy_grid_rl/
│   ├── env.py                    # Custom Gymnasium environment
│   ├── train.py                  # PPO training script
│   ├── evaluate.py               # Evaluation & visualization
│   ├── app.py                    # Gradio web application (HF Space)
│   ├── energy_grid_model.zip     # Pre-trained PPO model
│   ├── evaluation_results.png    # Sample evaluation output
│   └── requirements.txt         # Python dependencies
├── inference.py                  # LLM-based inference entry point
├── .gitattributes                # Git LFS tracking config
└── README.md
```

---

## 🔬 Environment Details

### Observation Space (8-dimensional)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `sin(hour)` | [-1, 1] | Cyclical time encoding (sine) |
| 1 | `cos(hour)` | [-1, 1] | Cyclical time encoding (cosine) |
| 2 | Solar Generation | [0, 15] kW | Peaks at noon, zero at night |
| 3 | Wind Generation | [0, ~10] kW | Stochastic with sinusoidal base |
| 4 | Consumer Demand | [0, ~16] kW | Dual peaks: morning & evening |
| 5 | Battery SoC | [0, 50] kWh | Current state of charge |
| 6 | Grid Price | [0.15, 0.25] $/kWh | Time-of-use pricing |
| 7 | Price Delta | [-0.10, 0.10] | Price change signal |

### Action Space (Discrete, 4 actions)

| Action | Name | Effect |
|--------|------|--------|
| 0 | **Charge** | Store energy in battery from renewables/grid |
| 1 | **Discharge** | Release stored energy to meet demand |
| 2 | **Sell Excess** | Sell surplus renewable energy back to grid |
| 3 | **Idle** | Take no action |

### Reward Shaping

- **Negative cost** from grid purchases
- **Bonus** for charging during cheap hours
- **Bonus** for discharging during expensive hours
- **Penalty** for unmet demand
- **Penalty** for extreme battery SoC (< 10% or > 90%)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/sarvagya25bcs10297-lab/Energy-grid-rl.git
cd Energy-grid-rl

# Install dependencies
pip install -r energy_grid_rl/requirements.txt
```

### Training the Agent

```bash
cd energy_grid_rl
python train.py
```

This trains a PPO agent for **100,000 timesteps** (~2,000 simulated days). The trained model is saved as `energy_grid_model.zip`.

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0001 |
| Gamma (Discount) | 0.999 |
| Steps per Update | 2048 |
| Batch Size | 64 |
| Policy | MlpPolicy |

### Evaluating the Agent

```bash
cd energy_grid_rl
python evaluate.py
```

Generates a 3-panel visualization saved to `evaluation_results.png`:
1. **Energy Balance** — Renewable production vs. consumer demand
2. **Battery SoC** — Battery state of charge over 24 hours
3. **Actions & Prices** — Agent decisions overlaid on market prices

### Running the Web App (Locally)

```bash
cd energy_grid_rl
python app.py
```

Opens a **Gradio** dashboard at `http://localhost:7860` with:
- 🎚️ Adjustable battery capacity (10–100 kWh)
- 🎚️ Demand multiplier (0.5x–2.0x)
- 📊 Real-time performance dashboard
- 🤖 AI-powered expert analysis via Hugging Face Inference API

### Running the Inference Script

```bash
# Set required environment variable
export HF_TOKEN=your_huggingface_token

# Optional: override defaults
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct

python inference.py
```

---

## 🌐 Live Demo

The app is deployed as a **Hugging Face Space**:

🔗 [**Launch AI Energy Grid Balancer →**](https://huggingface.co/spaces/Parzival7498/Energy-Grid-Rl)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| RL Framework | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO) |
| Environment | [Gymnasium](https://gymnasium.farama.org/) (Custom) |
| Web Interface | [Gradio](https://gradio.app/) |
| AI Analysis | [Hugging Face Inference API](https://huggingface.co/inference-api) |
| Visualization | [Matplotlib](https://matplotlib.org/) |
| Data Processing | [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/) |

---

## 📊 Sample Results

The trained agent learns to:
- **Charge** the battery during midday solar peaks (cheap energy)
- **Discharge** during evening demand peaks (expensive hours)
- **Sell excess** renewables when grid prices are high
- Maintain battery SoC within healthy bounds (10%–90%)

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

<p align="center">
  Built with ❤️ using Reinforcement Learning & Hugging Face
</p>
