---
title: Multi-Agent MarketForge
emoji: 🏪
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
suggested_hardware: t4-small
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

## Deployment to HuggingFace Spaces (GPU)

Space: https://huggingface.co/spaces/kenmandal/market-forge-env

> **Hardware:** This Space uses a **T4 Small GPU** (`suggested_hardware: t4-small`)
> so the trained model can run inference during visual simulations. The
> Dockerfile is based on `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime`
> and downloads the model at build time.

### Step 0 — Set GPU Hardware

On the HuggingFace Space settings page, select **T4 Small** (or higher) under
**Space hardware**. The `suggested_hardware: t4-small` field in the README
frontmatter signals this to HuggingFace, but you should verify it is active.

Available GPU tiers:
| Tier | GPU | VRAM | Cost |
|------|-----|------|------|
| `t4-small` | NVIDIA T4 | 16 GB | ~$0.60/hr |
| `t4-medium` | NVIDIA T4 | 16 GB | ~$0.90/hr |
| `a10g-small` | NVIDIA A10G | 24 GB | ~$1.05/hr |

The 0.5B model fits easily on a T4. For 1.7B+ models use `a10g-small`.

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
    commit_message="Deploy Multi-Agent MarketForge with GPU + trained model",
)
```

### Step 3 — Use a Custom Trained Model

To deploy your own fine-tuned model instead of the base Qwen:

**Option A — Set `MODEL_REPO` env var in HF Space Secrets:**
```
MODEL_REPO=kenmandal/market-forge-agent
```
The Dockerfile downloads this model at build time.

**Option B — Upload model weights directly:**
Push your `./market-forge-agent` directory to a HuggingFace model repo,
then set `MODEL_REPO` to that repo ID.

Files deployed: `README.md`, `Dockerfile`, `requirements.txt`, `app_visual.py`,
`run_simulation.py`, `evaluate_and_report.py`,
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

## Northflank Deployment

Project: https://app.northflank.com/t/openenv-hack-67/project/hackathon/
Dockerfile: `Dockerfile.northflank`
Node: 128 vCPU | 2 TB RAM | 1x NVIDIA H100 (80 GB)

### Northflank Service Configuration

| Setting | Value |
|---------|-------|
| Build source | GitHub repo `kaniska/MarketForge`, branch `main` |
| Dockerfile path | `Dockerfile.northflank` |
| Compute plan | H100 GPU (1x), 16+ vCPU, 64+ GB RAM |
| Ephemeral disk | 30 GB (under Advanced Resource Options) |
| Persistent volume | `/workspace/checkpoints` |
| Ports | 8000 (FastAPI env server), 8888 (JupyterLab) |
| Secrets | `HUGGING_FACE_HUB_TOKEN`, `WANDB_API_KEY` |

### Troubleshooting: "This repository is public and does not have Northflank installed"

This error appears when creating a CI/CD service pointing to `kaniska/MarketForge`.
It means the Northflank GitHub App is not installed on the repo, so auto-builds
on git push won't trigger. Three ways to resolve it:

#### Fix A — Install the Northflank GitHub App (enables auto-build on push)

1. Go to your team's Git integration page:
   `https://app.northflank.com/t/openenv-hack-67/integrations/vcs`
2. Click **Link provider → GitHub → Authorize**
3. When prompted, select **"Only select repositories"** and choose `kaniska/MarketForge`
4. Return to Northflank and re-create the service — the warning will be gone

#### Fix B — Create the service anyway and trigger builds manually

1. On the "Create service" screen, **ignore the warning and click Create**
2. Go to the service → **Builds** tab
3. Click **"Trigger build"** to start a build manually

Builds work fine — auto-trigger on push is just disabled without the GitHub App.
This is sufficient for hackathon use.

#### Fix C — Deploy from a pre-built Docker image (no GitHub integration needed)

Build and push the image locally, then point Northflank at the registry:

```bash
# Build the Northflank-specific image
cd market-forge-openenv/
docker build -f Dockerfile.northflank -t kenmandal/market-forge-training:latest .

# Push to Docker Hub
docker login
docker push kenmandal/market-forge-training:latest
```

In Northflank → Create service → **"Deploy a Docker image"** → enter:
```
kenmandal/market-forge-training:latest
```

No GitHub App required.

### Launch Model Training on Northflank

Once the service is running, override the CMD in Northflank UI
(**Service → Settings → CMD Override**) to start training:

```
/bin/bash -c "cd /workspace/market-forge-openenv && python train_market_forge.py"
```

Or connect via JupyterLab (port 8888) and run the training notebook interactively:
```
MarketForge_Colab_Training.ipynb
```

### Monitor Training on Northflank

| What | Where |
|------|-------|
| Container logs | Service → **Logs** tab |
| GPU / CPU metrics | Service → **Metrics** tab |
| Shell access | Service → **Shell** button (terminal in browser) |
| SSH access | Follow Northflank SSH guide, then `northflank login` |
| Checkpoints | Persistent volume at `/workspace/checkpoints` |
| Model on HF Hub | `https://huggingface.co/kenmandal/market-forge-agent` |

---

## Built With

- [OpenEnv v0.2.1](https://github.com/meta-pytorch/OpenEnv) - Environment framework
- [TRL (GRPOTrainer)](https://github.com/huggingface/trl) - Training pipeline
- [Gradio](https://gradio.app/) - Visual dashboard
- [FastAPI](https://fastapi.tiangolo.com/) - HTTP API server
