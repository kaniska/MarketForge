# Multi-Agent MarketForge — OpenEnv Environment

## Project Overview

Multi-commodity market simulation environment for training LLM agents in cooperation, competition, negotiation, and coalition formation. Built as an OpenEnv v0.2.1 environment.

## Architecture

```
[OpenEnv Client]  <--HTTP-->  [FastAPI Server]  <-->  [Market Engine]
     |                              |                      |
[GRPO Trainer]              [Gradio Dashboard]    [4 Commodity CDAs]
     |                              |              [Coalition Tracker]
[Reward Functions]          [Price Charts]         [Deal Negotiator]
                            [Wealth Tracking]      [Event Generator]
```

## Key Files

| File | Purpose |
|------|---------|
| `server/app.py` | FastAPI server — `/reset`, `/step`, `/state` endpoints |
| `server/market_environment.py` | Core market engine: CDA order book, coalitions, events, negotiation |
| `models.py` | Pydantic models: `MarketAction`, `MarketObservation`, `MarketState` |
| `client.py` | OpenEnv-compatible Python client for the environment |
| `rewards.py` | 5-level reward functions for GRPO training |
| `train_market_forge.py` | GRPO training script using TRL |
| `run_simulation.py` | Game loop: runs trained LLM agents vs baselines in a full episode |
| `evaluate_and_report.py` | Multi-episode evaluation: compares trained LLM vs random/rule-based, generates report |
| `app_visual.py` | Gradio visual dashboard (port 7860) |
| `Dockerfile` | HuggingFace Spaces deployment |
| `Dockerfile.northflank` | Northflank GPU deployment |

## Tech Stack

- **Python 3.10+**
- **FastAPI** — HTTP API server
- **Gradio** — Visual dashboard
- **TRL (GRPOTrainer)** — RL training pipeline
- **Transformers / Datasets** — Model & data loading
- **Matplotlib / NumPy** — Charting & numerics

## Market Simulation

- 4 raw commodities: Wheat, Iron, Timber, Oil
- 3 compound goods: Bread (Wheat+Oil), Tools (Iron+Timber), Furniture (Timber+Oil)
- 8 agents across 4 roles: Producer, Consumer, Trader, Speculator
- 12 stochastic market events
- Continuous Double Auction (CDA) for raw commodities
- Natural language negotiation for bilateral deals

## Running Locally

```bash
# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Start the visual dashboard
python app_visual.py  # http://localhost:7860

# Run a simulation with trained model vs baselines
python run_simulation.py --model ./market-forge-agent --rounds 30

# Run baseline-only (no LLM) for comparison
python run_simulation.py --baseline-only --rounds 30

# Full evaluation with report generation
python evaluate_and_report.py --model ./market-forge-agent --episodes 5 --rounds 30
```

## Training → Simulation → Evaluation Pipeline

```
1. Train:    python train_market_forge.py        → saves model to ./market-forge-agent
2. Simulate: python run_simulation.py --model ./market-forge-agent
3. Evaluate: python evaluate_and_report.py --model ./market-forge-agent
                                                  → evaluation_report.md + evaluation_data.json
```

## Deployment

- **HuggingFace Spaces**: `huggingface-cli login` then run the upload script in README
- **Northflank**: Use `Dockerfile.northflank` with H100 GPU node

## Conventions

- Use Pydantic models for all API request/response schemas
- Reward functions return float in [-1, 1] range
- Agent IDs follow `{role}_{number}` pattern (e.g., `trader_1`, `producer_2`)
- All prices are floats in dollar units
- Environment state is reset between episodes via `/reset`
