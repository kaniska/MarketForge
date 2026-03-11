# MarketForge OpenEnv — Skills & Capabilities

## Environment Management

- **Reset environment** — `POST /reset` clears all state, reinitializes agents, order books, and coalitions
- **Step environment** — `POST /step` with `MarketAction` JSON to advance simulation
- **Query state** — `GET /state` returns full `MarketState` snapshot

## Agent Actions

| Skill | Endpoint | Description |
|-------|----------|-------------|
| Buy/Sell | `POST /step` | Submit limit orders to the Continuous Double Auction |
| Produce | `POST /step` | Combine raw commodities into compound goods via recipes |
| Negotiate | `POST /step` | Send natural language deal proposals to other agents |
| Accept/Reject Deal | `POST /step` | Respond to incoming negotiation offers |
| Propose Coalition | `POST /step` | Create a new alliance with specified members |
| Join/Leave Coalition | `POST /step` | Manage coalition membership |
| Pass | `POST /step` | Skip turn |

## Training Pipeline

- **GRPO Training** — `train_market_forge.py` runs GRPOTrainer from TRL
- **Reward Functions** — 5 reward signals in `rewards.py`:
  - `reward_from_env` (40%) — environment step reward
  - `reward_valid_json` (20%) — valid action format
  - `reward_strategic_depth` (20%) — theory-of-mind actions
  - `reward_event_response` (10%) — market event adaptation
  - `reward_wealth_growth` (10%) — episode-level wealth change

## Visualization

- **Gradio Dashboard** — `app_visual.py` provides real-time:
  - Price charts for all commodities
  - Agent wealth tracking over time
  - Trade activity feed
  - Market event timeline

## Simulation & Evaluation

- **Run Simulation** — `run_simulation.py` drives a full game episode:
  - Trained LLM model generates JSON actions from observation prompts
  - Baseline agents (random or rule-based heuristics) compete alongside
  - Awards and leaderboard computed at game end
  - Supports: `--model`, `--llm-agents`, `--baseline-only`, `--rounds`, `--seed`

- **Agent Strategies** — Three built-in agent types in `run_simulation.py`:
  - `RandomAgent` — picks random legal actions with random parameters
  - `RuleBasedAgent` — role-aware heuristics (producers sell specialty, consumers buy ingredients, traders exploit spreads, speculators react to events)
  - `TrainedLLMAgent` — loads a fine-tuned HuggingFace model, feeds observation prompt, parses JSON action output

- **Evaluation & Reporting** — `evaluate_and_report.py` runs multi-episode comparison:
  - 3 configurations: random-baseline, rule-based-baseline, trained-LLM
  - Identical seeds across configs for fair comparison
  - Outputs `evaluation_report.md` (Markdown) and `evaluation_data.json` (raw data)
  - Report includes: executive summary, per-agent breakdown, episode-by-episode results

- **Winner Determination** — computed by `MarketEnvironment` at game end:
  - Award 1 (Market Champion): highest total_wealth = cash + inventory_value + compound_value
  - Award 2 (Master Strategist): 0.35*trade_efficiency + 0.25*negotiation_mastery + 0.20*cooperation_index + 0.20*event_adaptability

## Deployment Skills

- **HuggingFace Spaces** — Docker-based deployment via `Dockerfile`
- **Northflank GPU** — `Dockerfile.northflank` for H100 training nodes
- **Local dev** — `uvicorn server.app:app` + `python app_visual.py`

## Client Integration

```python
from client import MarketForgeEnv
from models import MarketAction

env = MarketForgeEnv(base_url="http://localhost:8000")
result = env.reset()
result = env.step(MarketAction(
    agent_id="trader_1",
    action_type="buy",
    commodity="wheat",
    price=12.0,
    quantity=5,
))
```
