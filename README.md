---
title: Multi-Agent MarketForge
emoji: 🏪
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - multi-agent
  - market-simulation
  - grpo
  - reinforcement-learning
---

# Multi-Agent MarketForge

A **multi-commodity market simulation environment** purpose-built for training LLM agents in **cooperation**, **competition**, **negotiation**, and **coalition formation**. Designed as an OpenEnv v0.2.1 environment for the OpenEnv Hackathon.

## The Four Pillars of Multi-Agent Interaction

| Pillar | Mechanism | Agent Behavior |
|--------|-----------|---------------|
| **Cooperation** | Supply chain formation | Agents combine raw commodities into compound goods (Bread = Wheat + Oil) |
| **Competition** | Continuous Double Auction | Agents bid/ask on limit order book for scarce resources |
| **Negotiation** | Natural language bilateral deals | Agents propose and counter-propose trade terms |
| **Coalition** | Dynamic alliance formation | Agents form buying/selling groups with reputation tracking |

## Environment Features

- **4 raw commodities** (Wheat, Iron, Timber, Oil) and **3 compound goods** (Bread, Tools, Furniture)
- **8 heterogeneous agents** across 4 roles (Producer, Consumer, Trader, Speculator)
- **12 stochastic market events** (droughts, embargoes, surpluses, festivals)
- **Partial observability** -- agents only see own inventory + top-of-book prices
- **Mixed-motive payoffs** with configurable self-interest parameter (alpha)
- **Theory-of-mind** progression: Level 0 (reactive) -> Level 1 (predictive) -> Level 2 (recursive)
- **Multi-level reward signals**: immediate (trades), intermediate (contracts), episode-level (wealth)

## Architecture

```
[OpenEnv Client]  <--HTTP-->  [FastAPI Server]  <-->  [Market Engine]
     |                              |                      |
[GRPO Trainer]              [Gradio Dashboard]    [4 Commodity CDAs]
     |                              |              [Coalition Tracker]
[Reward Functions]          [Price Charts]         [Deal Negotiator]
                            [Wealth Tracking]      [Event Generator]
```

## Deployment to HuggingFace Spaces

Space: https://huggingface.co/spaces/kenmandal/market-forge-env

### Step 1 — Authenticate
Only needed once (or when the token expires). Token name: `openenv-token`.
```bash
huggingface-cli login
# Paste your write-access token from: https://huggingface.co/settings/tokens
```

### Step 2 — Deploy
Run from inside the `market-forge-openenv/` directory:
```python
from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path=".",
    repo_id="kenmandal/market-forge-env",
    repo_type="space",
    ignore_patterns=[
        "__pycache__", "*.pyc", "*.pptx", "*.docx", "*.zip",
        "tests/", "training_results.png", "deploy_to_hf.sh",
        "*_1.py", "~$*",
    ],
    commit_message="Deploy Multi-Agent MarketForge",
)
```

Files deployed: `README.md`, `Dockerfile`, `requirements.txt`, `app_visual.py`,
`client.py`, `models.py`, `rewards.py`, `__init__.py`, `server/__init__.py`,
`server/app.py`, `server/market_environment.py`, `train_market_forge.py`,
`train_market_forge_notebook.py`

---

## Quick Start

### Connect to the Environment

```python
from client import MarketForgeEnv
from models import MarketAction

env = MarketForgeEnv(base_url="https://YOUR-SPACE.hf.space")
result = env.reset()
print(result.observation.prompt)

# Place a buy order
result = env.step(MarketAction(
    agent_id="trader_1",
    action_type="buy",
    commodity="wheat",
    price=12.0,
    quantity=5,
))
print(f"Reward: {result.reward}, Cash: {result.observation.cash}")
```

### Run the Visual Dashboard

```bash
python app_visual.py
# Open http://localhost:7860
```

### Train with GRPO

```bash
pip install trl transformers datasets accelerate
python train_market_forge.py
```

## Reward Structure

The reward system implements the formal mixed-motive payoff:

`u_i(a) = alpha_i * v_self(a) + (1 - alpha_i) * v_collective(a)`

| Reward Function | Signal | Weight |
|----------------|--------|--------|
| `reward_from_env` | Environment step reward (profit/loss) | 40% |
| `reward_valid_json` | Valid JSON action format | 20% |
| `reward_strategic_depth` | Theory-of-mind actions | 20% |
| `reward_event_response` | Adapting to market events | 10% |
| `reward_wealth_growth` | Episode-level wealth change | 10% |

## Action Types

| Type | Description | Pillar |
|------|-------------|--------|
| `buy` / `sell` | Submit limit order to CDA | Competition |
| `produce` | Create compound good from recipe | Cooperation |
| `negotiate` | Send NL message with deal proposal | Negotiation |
| `accept_deal` / `reject_deal` | Respond to deal offer | Negotiation |
| `propose_coalition` | Create new alliance | Coalition |
| `join_coalition` / `leave_coalition` | Manage coalition membership | Coalition |
| `pass` | Do nothing | -- |

## Judging Criteria Alignment

- **Environment Innovation (40%)**: Novel multi-commodity CDA + NL negotiation + coalition dynamics
- **Storytelling (30%)**: Interactive Gradio dashboard with real-time charts and event timeline
- **Training Improvement (20%)**: GRPO training script with observable reward curves
- **Reward Pipeline (10%)**: 5-level reward system with theory-of-mind bonuses

## Built With

- [OpenEnv v0.2.1](https://github.com/meta-pytorch/OpenEnv) - Environment framework
- [TRL (GRPOTrainer)](https://github.com/huggingface/trl) - Training pipeline
- [Gradio](https://gradio.app/) - Visual dashboard
- [FastAPI](https://fastapi.tiangolo.com/) - HTTP API server
