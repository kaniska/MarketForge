"""
Multi-Agent MarketForge - Visual Simulation Dashboard
=======================================================
Gradio-based interactive dashboard for visualizing the market simulation.
Covers the Storytelling criterion (30% of judging) with rich visualization.

Supports two simulation modes:
  - Heuristic-only: all agents use hand-crafted strategies (works everywhere)
  - Trained LLM:    selected agents use a GRPO-trained model for decisions
                    (requires torch + a downloaded model checkpoint)
"""
import gradio as gr
import json
import os
import random
import re
import time
import copy
from dataclasses import asdict
from typing import Dict, List, Any, Optional

from models import MarketAction, MarketObservation
from server.market_environment import (
    MarketEnvironment, COMMODITIES, COMPOUND_GOODS,
    BASE_PRICES, COMPOUND_PRICES, AGENT_ROLES,
)
from rewards import extract_action

# ===========================================================================
# Trained Model Loader (GPU-aware, graceful fallback)
# ===========================================================================
_loaded_model = None
_loaded_tokenizer = None
_model_device = None
_model_load_error: Optional[str] = None

# Resolve model path: env var > local dir > HuggingFace hub ID
DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_DIR",
    os.environ.get(
        "MODEL_REPO",
        "./trained-model"
        if os.path.isdir("./trained-model")
        else "Qwen/Qwen2.5-0.5B-Instruct",
    ),
)

LLM_SYSTEM_PROMPT = (
    "You are an autonomous trading agent in a multi-commodity MarketForge.\n"
    "You trade wheat, iron, timber, and oil. You can produce compound goods.\n\n"
    "RESPOND WITH EXACTLY ONE JSON OBJECT choosing your action. Valid action_types:\n"
    "  buy, sell, produce, negotiate, propose_coalition, join_coalition, "
    "accept_deal, pass\n\n"
    "EXAMPLES:\n"
    '  {"action_type":"buy","commodity":"wheat","price":10,"quantity":5}\n'
    '  {"action_type":"sell","commodity":"iron","price":18,"quantity":3}\n'
    '  {"action_type":"produce","compound_good":"bread"}\n'
    '  {"action_type":"negotiate","target_agent":"producer_wheat",'
    '"commodity":"wheat","price":9,"quantity":10,'
    '"message":"Bulk discount for wheat?"}\n'
    '  {"action_type":"pass"}\n\n'
    "STRATEGY: Buy low, sell high. React to events. Negotiate deals. "
    "Form coalitions. Produce compound goods when profitable."
)


def load_trained_model(model_path: str = None):
    """Load the trained model + tokenizer. Returns (model, tokenizer, device, error)."""
    global _loaded_model, _loaded_tokenizer, _model_device, _model_load_error

    if _loaded_model is not None:
        return _loaded_model, _loaded_tokenizer, _model_device, None

    path = model_path or DEFAULT_MODEL_PATH
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[app_visual] Loading trained model from {path} ...")
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, device_map=device_map,
        )
        model.eval()
        device = next(model.parameters()).device

        _loaded_model = model
        _loaded_tokenizer = tokenizer
        _model_device = device
        _model_load_error = None
        print(f"[app_visual] Model loaded on {device} (dtype={dtype})")
        return model, tokenizer, device, None

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        _model_load_error = err
        print(f"[app_visual] Model load failed: {err}")
        return None, None, None, err


def llm_decide(obs: MarketObservation, agent_id: str) -> MarketAction:
    """Generate an action using the loaded trained model."""
    model, tokenizer, device, err = load_trained_model()
    if model is None:
        # Fallback to pass if model unavailable
        return MarketAction(agent_id=agent_id, action_type="pass")

    import torch

    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": obs.prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    parsed = extract_action(raw_output)
    if parsed and "action_type" in parsed:
        parsed["agent_id"] = agent_id
        valid_fields = set(MarketAction.__dataclass_fields__.keys())
        filtered = {k: v for k, v in parsed.items() if k in valid_fields}
        return MarketAction(**filtered)

    return MarketAction(agent_id=agent_id, action_type="pass")


# Try to pre-load the model at startup (non-blocking)
try:
    load_trained_model()
except Exception:
    pass

# ===========================================================================
# Simulation Engine (local, no server needed)
# ===========================================================================
env = MarketEnvironment()
sim_history: List[Dict[str, Any]] = []
agent_wealth_history: Dict[str, List[float]] = {}
price_chart_data: Dict[str, List[float]] = {}
reward_history: List[float] = []
trade_log: List[str] = []
coalition_log: List[str] = []
negotiation_log: List[str] = []
_state_last_modified: float = time.time()  # only updated on actual state changes


# ---------------------------------------------------------------------------
# Heuristic AI Agents (different strategies for simulation)
# ---------------------------------------------------------------------------
class HeuristicAgent:
    """Base heuristic agent that makes plausible market decisions."""

    def __init__(self, agent_id: str, role: str, specialty: str = ""):
        self.agent_id = agent_id
        self.role = role
        self.specialty = specialty
        self.history: List[float] = []

    def decide(self, obs: MarketObservation) -> MarketAction:
        if self.role == "producer":
            return self._producer_strategy(obs)
        elif self.role == "consumer":
            return self._consumer_strategy(obs)
        elif self.role == "trader":
            return self._trader_strategy(obs)
        elif self.role == "speculator":
            return self._speculator_strategy(obs)
        else:
            return MarketAction(agent_id=self.agent_id, action_type="pass")

    def _producer_strategy(self, obs: MarketObservation) -> MarketAction:
        """Producers sell their specialty commodity."""
        inv = obs.inventory.get(self.specialty, 0)
        if inv > 5:
            # Sell at slightly above market price
            market_price = obs.last_trade_prices.get(self.specialty, BASE_PRICES.get(self.specialty, 10))
            if market_price <= 0:
                market_price = BASE_PRICES.get(self.specialty, 10)
            sell_price = market_price * random.uniform(0.9, 1.15)
            qty = random.randint(2, min(8, inv))
            return MarketAction(
                agent_id=self.agent_id,
                action_type="sell",
                commodity=self.specialty,
                price=round(sell_price, 1),
                quantity=qty,
            )

        # Occasionally propose coalition
        if random.random() < 0.1:
            return MarketAction(
                agent_id=self.agent_id,
                action_type="propose_coalition",
                message=f"Producer coalition for {self.specialty} price stability",
            )
        return MarketAction(agent_id=self.agent_id, action_type="pass")

    def _consumer_strategy(self, obs: MarketObservation) -> MarketAction:
        """Consumers buy ingredients and produce compound goods."""
        target_good = self.specialty  # e.g. "bread"
        if target_good in COMPOUND_GOODS:
            recipe = COMPOUND_GOODS[target_good]
            # Check if we can produce
            can_produce = all(
                obs.inventory.get(ing, 0) >= qty
                for ing, qty in recipe.items()
            )
            if can_produce:
                return MarketAction(
                    agent_id=self.agent_id,
                    action_type="produce",
                    compound_good=target_good,
                )

            # Buy missing ingredients
            for ingredient, qty_needed in recipe.items():
                have = obs.inventory.get(ingredient, 0)
                if have < qty_needed:
                    market_price = obs.last_trade_prices.get(ingredient, BASE_PRICES.get(ingredient, 10))
                    if market_price <= 0:
                        market_price = BASE_PRICES.get(ingredient, 10)
                    buy_price = market_price * random.uniform(1.0, 1.2)
                    buy_qty = qty_needed - have + random.randint(0, 2)

                    # Sometimes negotiate instead of buying on open market
                    if random.random() < 0.3:
                        producer = f"producer_{ingredient}"
                        return MarketAction(
                            agent_id=self.agent_id,
                            action_type="negotiate",
                            target_agent=producer,
                            commodity=ingredient,
                            price=round(buy_price * 0.9, 1),
                            quantity=buy_qty,
                            message=f"I need {buy_qty} units of {ingredient} for {target_good} production. Can we agree on ${buy_price * 0.9:.1f}/unit?",
                        )

                    return MarketAction(
                        agent_id=self.agent_id,
                        action_type="buy",
                        commodity=ingredient,
                        price=round(buy_price, 1),
                        quantity=buy_qty,
                    )

        return MarketAction(agent_id=self.agent_id, action_type="pass")

    def _trader_strategy(self, obs: MarketObservation) -> MarketAction:
        """Traders do arbitrage - buy low, sell high."""
        # Look for arbitrage opportunities
        best_buy = None
        best_sell = None

        for c in COMMODITIES:
            tb = obs.top_of_book.get(c, {})
            best_ask = tb.get("best_ask", 0)
            best_bid = tb.get("best_bid", 0)
            inv = obs.inventory.get(c, 0)

            if best_ask > 0 and best_bid > best_ask * 1.05 and obs.cash > best_ask * 3:
                best_buy = (c, best_ask)
            if inv > 3 and best_bid > 0:
                best_sell = (c, best_bid)

        if best_buy and random.random() < 0.6:
            c, price = best_buy
            return MarketAction(
                agent_id=self.agent_id, action_type="buy",
                commodity=c, price=round(price * 1.02, 1),
                quantity=random.randint(2, 5),
            )

        if best_sell and random.random() < 0.5:
            c, price = best_sell
            inv = obs.inventory.get(c, 0)
            return MarketAction(
                agent_id=self.agent_id, action_type="sell",
                commodity=c, price=round(price * 0.98, 1),
                quantity=min(random.randint(1, 3), inv),
            )

        # Join available coalitions
        if random.random() < 0.15 and obs.coalitions:
            return MarketAction(
                agent_id=self.agent_id, action_type="pass",
            )

        # Random trade
        c = random.choice(COMMODITIES)
        if obs.inventory.get(c, 0) > 3:
            price = obs.last_trade_prices.get(c, BASE_PRICES[c])
            if price <= 0:
                price = BASE_PRICES[c]
            return MarketAction(
                agent_id=self.agent_id, action_type="sell",
                commodity=c, price=round(price * random.uniform(0.95, 1.1), 1),
                quantity=random.randint(1, 3),
            )

        return MarketAction(agent_id=self.agent_id, action_type="pass")

    def _speculator_strategy(self, obs: MarketObservation) -> MarketAction:
        """Speculators react to events and try to profit from price movements."""
        event = obs.event.lower()

        # React to events
        for c in COMMODITIES:
            if c in event and ("reduce" in event or "collapse" in event or "embargo" in event):
                # Supply shock - buy before price rises
                if obs.cash > 100:
                    price = obs.last_trade_prices.get(c, BASE_PRICES[c])
                    if price <= 0:
                        price = BASE_PRICES[c]
                    return MarketAction(
                        agent_id=self.agent_id, action_type="buy",
                        commodity=c, price=round(price * 1.15, 1),
                        quantity=random.randint(3, 8),
                    )
            if c in event and ("surplus" in event or "route" in event):
                # Oversupply - sell if holding
                inv = obs.inventory.get(c, 0)
                if inv > 2:
                    price = obs.last_trade_prices.get(c, BASE_PRICES[c])
                    if price <= 0:
                        price = BASE_PRICES[c]
                    return MarketAction(
                        agent_id=self.agent_id, action_type="sell",
                        commodity=c, price=round(price * 0.95, 1),
                        quantity=min(inv, random.randint(2, 5)),
                    )

        # Sell holdings at profit
        for c in COMMODITIES:
            inv = obs.inventory.get(c, 0)
            if inv > 5:
                price = obs.last_trade_prices.get(c, BASE_PRICES[c])
                if price <= 0:
                    price = BASE_PRICES[c]
                return MarketAction(
                    agent_id=self.agent_id, action_type="sell",
                    commodity=c, price=round(price * 1.1, 1),
                    quantity=min(inv - 1, random.randint(2, 4)),
                )

        # Propose coalition for market manipulation
        if random.random() < 0.08:
            return MarketAction(
                agent_id=self.agent_id,
                action_type="propose_coalition",
                message="Speculator alliance - coordinate buying for price push",
            )

        return MarketAction(agent_id=self.agent_id, action_type="pass")


# ===========================================================================
# Simulation Runner
# ===========================================================================
def run_simulation(
    num_rounds: int = 30,
    agent_mode: str = "Heuristic Only",
    llm_agents_str: str = "trader_1, speculator_1",
):
    """Run a complete market simulation and return visualization data.

    Args:
        num_rounds: Number of game rounds.
        agent_mode: "Heuristic Only" or "Trained LLM + Heuristic".
        llm_agents_str: Comma-separated agent IDs to drive with the LLM.
    """
    global sim_history, agent_wealth_history, price_chart_data
    global reward_history, trade_log, coalition_log, negotiation_log
    global _state_last_modified

    sim_history = []
    agent_wealth_history = {aid: [] for aid in AGENT_ROLES}
    price_chart_data = {c: [] for c in COMMODITIES}
    reward_history = []
    trade_log = []
    coalition_log = []
    negotiation_log = []

    # Determine which agents use the trained LLM
    use_llm = agent_mode == "Trained LLM + Heuristic"
    llm_agent_ids = set()
    if use_llm:
        llm_agent_ids = {a.strip() for a in llm_agents_str.split(",") if a.strip()}
        # Validate model is loadable
        _, _, _, err = load_trained_model()
        if err:
            use_llm = False
            llm_agent_ids = set()
            trade_log.append(
                f"**[WARNING]** Trained model unavailable ({err}). "
                f"Falling back to heuristic for all agents."
            )

    obs = env.reset(max_rounds=num_rounds)

    # Create heuristic agents (used for all agents not driven by LLM)
    agents = {}
    for aid, config in AGENT_ROLES.items():
        agents[aid] = HeuristicAgent(
            aid, config["role"], config.get("specialty", "")
        )

    total_rewards = {aid: 0.0 for aid in agents}
    round_data = []

    for round_num in range(num_rounds):
        round_rewards = {}
        for aid, agent in agents.items():
            obs = env._make_observation(aid)

            # Choose decision source: trained LLM or heuristic
            if use_llm and aid in llm_agent_ids:
                action = llm_decide(obs, aid)
            else:
                action = agent.decide(obs)

            result_obs = env.step(action)
            round_rewards[aid] = result_obs.reward
            total_rewards[aid] += result_obs.reward

            # Log interesting actions
            if action.action_type in ("buy", "sell"):
                trade_log.append(
                    f"R{round_num}: {aid} {action.action_type}s "
                    f"{action.quantity}x {action.commodity} @ ${action.price:.1f} "
                    f"(reward: {result_obs.reward:.2f})"
                )
            elif action.action_type == "negotiate":
                negotiation_log.append(
                    f"R{round_num}: {aid} -> {action.target_agent}: "
                    f'"{action.message[:60]}..."'
                )
            elif action.action_type == "propose_coalition":
                coalition_log.append(
                    f"R{round_num}: {aid} proposes coalition: {action.message[:50]}"
                )
            elif action.action_type == "produce":
                trade_log.append(
                    f"R{round_num}: {aid} produces {action.compound_good} (reward: {result_obs.reward:.2f})"
                )

        # Record state after all agents act
        state = env.state
        for aid in agents:
            ag = state.agents.get(aid, {})
            wealth = ag.get("cash", 0) + sum(
                ag.get("inventory", {}).get(c, 0) * BASE_PRICES.get(c, 10)
                for c in COMMODITIES
            )
            agent_wealth_history[aid].append(round(wealth, 2))

        for c in COMMODITIES:
            prices = state.price_history.get(c, [])
            price_chart_data[c].append(prices[-1] if prices else BASE_PRICES[c])

        avg_reward = sum(round_rewards.values()) / max(len(round_rewards), 1)
        reward_history.append(round(avg_reward, 3))

        sim_history.append({
            "round": round_num,
            "event": state.current_event,
            "total_trades": state.market_metrics.get("total_trades", 0),
            "coalitions": len(state.coalitions),
            "compound_produced": state.market_metrics.get("compound_goods_produced", 0),
        })

    _state_last_modified = time.time()
    return _format_results(num_rounds, total_rewards, llm_agent_ids)


def _format_results(num_rounds, total_rewards, llm_agent_ids=None):
    """Format simulation results for Gradio display with Awards and Scoring."""
    llm_agent_ids = llm_agent_ids or set()

    # Compute Awards, Leaderboard, Accuracy
    state = env.state
    awards = env.compute_awards()
    leaderboard = env.compute_leaderboard()
    accuracy = env.compute_accuracy()

    champion = awards["market_champion"]
    strategist = awards["master_strategist"]

    # ---- Summary with Awards ----
    metrics = state.market_metrics
    coal_count = len(state.coalitions)
    summary_lines = [
        "## Simulation Complete",
        f"**Rounds:** {num_rounds}",
        f"**Total Trades:** {metrics.get('total_trades', 0)} "
        f"| **Volume:** ${metrics.get('total_volume', 0):,.1f}",
        f"**Compound Goods Produced:** {metrics.get('compound_goods_produced', 0)}",
        f"**Coalitions Formed:** {metrics.get('coalitions_formed', 0)} "
        f"| **Active:** {coal_count}",
        f"**Deals Negotiated:** {metrics.get('deals_negotiated', 0)}",
        "",
        "---",
        "",
        "## AWARDS",
        "",
        "### Award 1: Market Champion (Wealth)",
        f"**Winner: {champion['agent_id']}**",
        f"Total Wealth: **${champion['total_wealth']:,.2f}**",
        f"Wealth Growth: +${champion.get('wealth_growth', 0):,.2f}",
        "",
        f"> **Why {champion['agent_id']} won:** This agent accumulated the highest "
        f"**total wealth** (cash + inventory market value + compound goods value). "
        f"Wealth = cash on hand + units in inventory valued at current market prices "
        f"+ any compound goods at their production value. Growing wealth by "
        f"**${champion.get('wealth_growth', 0):+,.2f}** means this agent's trading, "
        f"production, and negotiation strategy outperformed all others in "
        f"portfolio accumulation.",
        "",
        "### Award 2: Master Strategist (Skill)",
        f"**Winner: {strategist['agent_id']}**",
        f"Strategic Score: **{strategist['strategic_score']:.4f}**",
    ]
    bd = strategist.get("breakdown", {})
    if bd:
        summary_lines.extend([
            f"  - Trade Efficiency: {bd.get('trade_efficiency', 0):.2%} "
            f"(35% weight — profitable trades / total trades)",
            f"  - Negotiation Mastery: {bd.get('negotiation_mastery', 0):.2%} "
            f"(25% weight — successful deals / total deals)",
            f"  - Cooperation Index: {bd.get('cooperation_index', 0):.2%} "
            f"(20% weight — compound production + coalition activity)",
            f"  - Event Adaptability: {bd.get('event_adaptability', 0):.2%} "
            f"(20% weight — actions responding to market events)",
            "",
            f"> **Why {strategist['agent_id']} won:** This agent scored highest on "
            f"the **strategic skill composite** — a weighted blend of four pillars. "
            f"Unlike the Wealth award (pure portfolio size), the Strategist award "
            f"measures *how skillfully* an agent plays: making profitable trades, "
            f"closing negotiated deals, cooperating in supply chains, and "
            f"adapting behaviour to market shocks.",
        ])

    # ---- Agent Mode Banner ----
    if llm_agent_ids:
        summary_lines.extend([
            "",
            "---",
            "",
            "## AGENT MODE",
            f"**Trained LLM agents:** {', '.join(sorted(llm_agent_ids))}",
            f"**Heuristic agents:** {', '.join(a for a in AGENT_ROLES if a not in llm_agent_ids)}",
            f"**Model:** `{DEFAULT_MODEL_PATH}`",
        ])

    # ---- Leaderboard Table ----
    summary_lines.extend([
        "",
        "---",
        "",
        "## LEADERBOARD & SCORING",
        "",
        "| Rank | Agent | Strategy | Role | Wealth | Strategic | Accuracy | Awards |",
        "|------|-------|----------|------|--------|-----------|----------|--------|",
    ])
    for entry in leaderboard:
        awards_str = ", ".join(entry["awards_won"]) if entry["awards_won"] else "-"
        strategy_tag = "**LLM**" if entry["agent_id"] in llm_agent_ids else "heuristic"
        summary_lines.append(
            f"| {entry['rank']} | {entry['agent_id']} | {strategy_tag} | {entry['role']} "
            f"| ${entry['total_wealth']:,.0f} "
            f"| {entry['strategic_score']:.3f} "
            f"| {entry['accuracy']:.0%} "
            f"| {awards_str} |"
        )

    summary = "\n".join(summary_lines)

    # ---- Charts (6 panels) ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("Multi-Agent MarketForge - Simulation Dashboard",
                 fontsize=14, fontweight='bold')

    # Panel 1: Commodity Prices
    ax1 = axes[0, 0]
    colors_map = {"wheat": "#DAA520", "iron": "#708090",
                  "timber": "#228B22", "oil": "#2F4F4F"}
    for c in COMMODITIES:
        data = price_chart_data.get(c, [])
        if data:
            ax1.plot(data, label=c.capitalize(), color=colors_map[c], linewidth=2)
    ax1.set_title("Commodity Prices Over Time")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Agent Wealth (Award 1 race)
    ax2 = axes[0, 1]
    role_colors = {"producer": "#4CAF50", "consumer": "#2196F3",
                   "trader": "#FF9800", "speculator": "#F44336"}
    for aid in list(agent_wealth_history.keys())[:6]:
        data = agent_wealth_history.get(aid, [])
        if data:
            role = AGENT_ROLES.get(aid, {}).get("role", "trader")
            lw = 3 if aid == champion["agent_id"] else 1.5
            ls = "-" if aid == champion["agent_id"] else "--"
            label = f"{aid[:12]}"
            if aid == champion["agent_id"]:
                label += " [CHAMPION]"
            ax2.plot(data, label=label, color=role_colors.get(role, "#999"),
                     linewidth=lw, linestyle=ls, alpha=0.9)
    ax2.set_title("Award 1: Wealth Race (Market Champion)")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Total Wealth ($)")
    ax2.legend(loc="upper left", fontsize=6)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Average Reward per Round
    ax3 = axes[1, 0]
    if reward_history:
        ax3.plot(reward_history, color="#9C27B0", linewidth=2)
        window = min(5, len(reward_history))
        if window > 1:
            ma = np.convolve(reward_history, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(reward_history)), ma,
                     color="#E91E63", linewidth=2, linestyle="--",
                     label=f"{window}-round MA")
            ax3.legend(fontsize=8)
    ax3.set_title("Average Reward per Round")
    ax3.set_xlabel("Round")
    ax3.set_ylabel("Avg Reward")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Market Activity
    ax4 = axes[1, 1]
    if sim_history:
        rounds_list = [s["round"] for s in sim_history]
        trades_list = [s["total_trades"] for s in sim_history]
        coalitions_list = [s["coalitions"] for s in sim_history]
        compounds_list = [s["compound_produced"] for s in sim_history]
        ax4.bar(rounds_list, trades_list, alpha=0.5, color="#2196F3",
                label="Cumulative Trades")
        ax4_twin = ax4.twinx()
        ax4_twin.plot(rounds_list, coalitions_list, color="#FF5722",
                      linewidth=2, label="Active Coalitions")
        ax4_twin.plot(rounds_list, compounds_list, color="#4CAF50",
                      linewidth=2, linestyle="--", label="Compound Goods")
        ax4.set_title("Market Activity")
        ax4.set_xlabel("Round")
        ax4.set_ylabel("Trades", color="#2196F3")
        ax4_twin.set_ylabel("Count", color="#FF5722")
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper left", fontsize=7)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Award 2 -- Strategic Scores Breakdown
    ax5 = axes[2, 0]
    if leaderboard:
        agent_names = [e["agent_id"][:10] for e in leaderboard[:6]]
        strat_scores = [e["strategic_score"] for e in leaderboard[:6]]
        bar_colors = ["#FFD700" if e["agent_id"] == strategist["agent_id"]
                      else "#90CAF9" for e in leaderboard[:6]]
        bars = ax5.barh(agent_names[::-1], strat_scores[::-1],
                        color=bar_colors[::-1])
        ax5.set_title("Award 2: Strategic Scores (Master Strategist)")
        ax5.set_xlabel("Strategic Score")
        for bar, score in zip(bars, strat_scores[::-1]):
            ax5.text(bar.get_width() + 0.01,
                     bar.get_y() + bar.get_height()/2,
                     f"{score:.3f}", va='center', fontsize=8)
    ax5.grid(True, alpha=0.3, axis='x')

    # Panel 6: Action Accuracy (Model Improvement Tracking)
    ax6 = axes[2, 1]
    if accuracy:
        agent_names_acc = [aid[:10] for aid in accuracy.keys()]
        acc_values = list(accuracy.values())
        bar_colors_acc = ["#4CAF50" if v >= 0.9 else "#FFC107" if v >= 0.7
                          else "#F44336" for v in acc_values]
        bars_acc = ax6.barh(agent_names_acc[::-1], acc_values[::-1],
                            color=bar_colors_acc[::-1])
        ax6.set_title("Action Accuracy (Valid / Total)")
        ax6.set_xlabel("Accuracy")
        ax6.set_xlim(0, 1.15)
        for bar, val in zip(bars_acc, acc_values[::-1]):
            ax6.text(bar.get_width() + 0.02,
                     bar.get_y() + bar.get_height()/2,
                     f"{val:.0%}", va='center', fontsize=8)
    ax6.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # ---- Trade / Negotiation / Coalition Logs ----
    log_text = _format_activity_logs()

    # ---- Event Timeline ----
    event_text = _format_event_timeline()

    return summary, fig, log_text, event_text


def _format_activity_logs() -> str:
    """Format Recent Trades, Negotiations, and Coalitions into structured reports."""
    sections = []

    # --- Recent Trades ---
    sections.append("## Recent Trades")
    if trade_log:
        sections.append("")
        sections.append("| Round | Agent | Action | Details | Reward |")
        sections.append("|:-----:|-------|--------|---------|-------:|")
        for entry in trade_log[-20:]:
            # Parse: "R{round}: {agent} {action}s {qty}x {commodity} @ ${price} (reward: {r})"
            # or:    "R{round}: {agent} produces {compound} (reward: {r})"
            parts = entry.split(": ", 1)
            round_str = parts[0] if parts else ""
            rest = parts[1] if len(parts) > 1 else entry
            # Extract reward
            reward_str = ""
            if "(reward:" in rest:
                reward_part = rest.split("(reward:")[1].rstrip(")")
                reward_str = reward_part.strip()
                rest = rest.split("(reward:")[0].strip()
            # Extract agent (first word after round)
            tokens = rest.split(" ", 1)
            agent = tokens[0] if tokens else ""
            action_detail = tokens[1] if len(tokens) > 1 else ""
            # Split action from details
            if " produces " in action_detail:
                action = "produces"
                details = action_detail.replace("produces ", "")
            elif "buys " in action_detail:
                action = "buy"
                details = action_detail.replace("buys ", "")
            elif "sells " in action_detail:
                action = "sell"
                details = action_detail.replace("sells ", "")
            else:
                action = action_detail.split(" ")[0] if action_detail else ""
                details = " ".join(action_detail.split(" ")[1:])
            sections.append(
                f"| {round_str} | **{agent}** | `{action}` | {details} | {reward_str} |"
            )
    else:
        sections.append("\n*No trades recorded yet.*")

    # --- Negotiations ---
    sections.append("")
    sections.append("---")
    sections.append("")
    sections.append("## Negotiations")
    if negotiation_log:
        sections.append("")
        sections.append("| Round | From | To | Message |")
        sections.append("|:-----:|------|----|---------| ")
        for entry in negotiation_log[-10:]:
            # Parse: "R{round}: {agent} -> {target}: "{message}...""
            parts = entry.split(": ", 1)
            round_str = parts[0] if parts else ""
            rest = parts[1] if len(parts) > 1 else entry
            if " -> " in rest:
                from_agent, remainder = rest.split(" -> ", 1)
                if ": " in remainder:
                    to_agent, message = remainder.split(": ", 1)
                else:
                    to_agent = remainder
                    message = ""
                message = message.strip().strip('"')
            else:
                from_agent = rest.split(" ")[0] if rest else ""
                to_agent = ""
                message = rest
            sections.append(
                f"| {round_str} | **{from_agent}** | **{to_agent}** | *{message}* |"
            )
    else:
        sections.append("\n*No negotiations recorded yet.*")

    # --- Coalitions ---
    sections.append("")
    sections.append("---")
    sections.append("")
    sections.append("## Coalitions")
    if coalition_log:
        sections.append("")
        sections.append("| Round | Agent | Proposal |")
        sections.append("|:-----:|-------|----------|")
        for entry in coalition_log[-10:]:
            # Parse: "R{round}: {agent} proposes coalition: {message}"
            parts = entry.split(": ", 1)
            round_str = parts[0] if parts else ""
            rest = parts[1] if len(parts) > 1 else entry
            if " proposes coalition: " in rest:
                agent, proposal = rest.split(" proposes coalition: ", 1)
            else:
                tokens = rest.split(" ", 1)
                agent = tokens[0] if tokens else ""
                proposal = tokens[1] if len(tokens) > 1 else ""
            sections.append(
                f"| {round_str} | **{agent}** | {proposal} |"
            )
    else:
        sections.append("\n*No coalitions formed yet.*")

    return "\n".join(sections)


def _format_event_timeline() -> str:
    """Format Market Events Timeline into a structured report."""
    lines = ["## Market Events Timeline", ""]

    if not sim_history:
        lines.append("*No events recorded yet.*")
        return "\n".join(lines)

    lines.append("| Round | Event | Trades | Coalitions | Compounds |")
    lines.append("|:-----:|-------|-------:|:----------:|:---------:|")

    for s in sim_history[-15:]:
        event = s["event"]
        # Highlight disruptive events
        if any(kw in event.lower() for kw in ["drought", "collapse", "embargo", "restricted"]):
            event = f"**{event}**"
        lines.append(
            f"| {s['round']} | {event} "
            f"| {s['total_trades']} "
            f"| {s['coalitions']} "
            f"| {s['compound_produced']} |"
        )

    return "\n".join(lines)


def _get_active_coalitions() -> list:
    """Return list of active coalition IDs from the environment."""
    try:
        return list(env.state.coalitions.keys())
    except Exception:
        return []


def _action_effect_explanation(agent_id: str, action_type: str, obs, state) -> str:
    """Generate a human-readable explanation of what a manual action did."""
    reward_quality = "positive" if obs.reward > 0 else "negative" if obs.reward < 0 else "neutral"
    agent_data = state.agents.get(agent_id, {})

    if action_type == "buy":
        return (
            f"**Effect:** {agent_id} placed a **buy order** on the market. "
            f"Cash decreased as inventory grew. Reward was **{reward_quality}** "
            f"({obs.reward:+.3f}) — a buy is profitable when the price is below "
            f"market value, allowing the agent to sell later at a higher price."
        )
    elif action_type == "sell":
        return (
            f"**Effect:** {agent_id} placed a **sell order**, converting inventory "
            f"into cash. Reward was **{reward_quality}** ({obs.reward:+.3f}) — "
            f"selling is most effective when prices are above production cost."
        )
    elif action_type == "produce":
        return (
            f"**Effect:** {agent_id} attempted to **produce a compound good**, "
            f"consuming raw materials from inventory. Reward was **{reward_quality}** "
            f"({obs.reward:+.3f}) — production is the cooperation pillar: it requires "
            f"buying inputs from other agents to craft higher-value goods."
        )
    elif action_type == "negotiate":
        return (
            f"**Effect:** {agent_id} sent a **negotiation proposal** to the target "
            f"agent. Reward was **{reward_quality}** ({obs.reward:+.3f}) — "
            f"negotiation builds reputation and enables bilateral deals outside "
            f"the open auction."
        )
    elif action_type == "propose_coalition":
        coal_count = len(state.coalitions)
        return (
            f"**Effect:** {agent_id} **proposed a new coalition** (alliance). "
            f"There are now **{coal_count} active coalition(s)**. Reward was "
            f"**{reward_quality}** ({obs.reward:+.3f}) — coalitions give members "
            f"synergy bonuses from role diversity and periodic profit sharing."
        )
    elif action_type == "join_coalition":
        return (
            f"**Effect:** {agent_id} **joined an existing coalition**. Reward was "
            f"**{reward_quality}** ({obs.reward:+.3f}) — joining gives a synergy "
            f"bonus that scales with coalition size and member role diversity. "
            f"A negative reward means the coalition ID was not found."
        )
    elif action_type == "leave_coalition":
        return (
            f"**Effect:** {agent_id} **left a coalition**. Reward was "
            f"**{reward_quality}** ({obs.reward:+.3f}) — leaving incurs a small "
            f"reputation penalty but frees the agent from coalition obligations."
        )
    elif action_type == "pass":
        return (
            f"**Effect:** {agent_id} **passed** this turn, taking no action. "
            f"Reward was **{reward_quality}** ({obs.reward:+.3f}) — passing "
            f"incurs a small penalty to discourage inaction."
        )
    else:
        return (
            f"**Effect:** Action `{action_type}` executed. "
            f"Reward: **{obs.reward:+.3f}**."
        )


def step_single_agent(agent_id: str, action_type: str, commodity: str,
                       price: float, quantity: int, target: str,
                       message: str, coalition_id: str):
    """Allow manual agent control and return result + updated leaderboard."""
    global trade_log, negotiation_log, coalition_log, _state_last_modified

    action = MarketAction(
        agent_id=agent_id,
        action_type=action_type,
        commodity=commodity,
        price=price,
        quantity=int(quantity),
        target_agent=target,
        message=message,
        coalition_id=coalition_id.strip() if coalition_id else "",
    )
    obs = env.step(action)
    _state_last_modified = time.time()

    # Update logs so admin dashboard sees manual entries
    state = env.state
    round_num = state.round_number
    if action_type in ("buy", "sell"):
        trade_log.append(
            f"R{round_num}: {agent_id} {action_type}s "
            f"{int(quantity)}x {commodity} @ ${price:.1f} "
            f"(reward: {obs.reward:.2f})"
        )
    elif action_type == "negotiate":
        negotiation_log.append(
            f"R{round_num}: {agent_id} -> {target}: "
            f'"{message[:60]}..."'
        )
    elif action_type == "propose_coalition":
        coalition_log.append(
            f"R{round_num}: {agent_id} proposes coalition: {message[:50]}"
        )
    elif action_type == "join_coalition":
        coalition_log.append(
            f"R{round_num}: {agent_id} joins coalition {coalition_id}"
        )
    elif action_type == "leave_coalition":
        coalition_log.append(
            f"R{round_num}: {agent_id} leaves coalition {coalition_id}"
        )
    elif action_type == "produce":
        trade_log.append(
            f"R{round_num}: {agent_id} produces {commodity} (reward: {obs.reward:.2f})"
        )

    # Build effect explanation
    effect_text = _action_effect_explanation(agent_id, action_type, obs, state)

    result_text = f"""### Action Result
**Agent:** {agent_id} | **Action:** {action_type}
**Reward:** {obs.reward:.3f} | **Round:** {obs.round_number}/{obs.max_rounds}
**Cash:** ${obs.cash:.2f} | **Reputation:** {obs.reputation:.3f}
**Inventory:** {json.dumps({k:v for k,v in obs.inventory.items() if v > 0})}
**Event:** {obs.event}

---
{effect_text}
"""

    # Build updated leaderboard with award explanations
    leaderboard_md = _build_live_leaderboard()

    return result_text, leaderboard_md


def _build_live_leaderboard() -> str:
    """Build a current leaderboard with award explanations from live env state."""
    try:
        leaderboard = env.compute_leaderboard()
        awards = env.compute_awards()
        champion_id = awards["market_champion"]["agent_id"]
        strategist_id = awards["master_strategist"]["agent_id"]
    except Exception:
        return "*No leaderboard data yet — run a simulation or execute more actions.*"

    lines = [
        "### Live Leaderboard",
        "",
        "| Rank | Agent | Role | Wealth | Strategic | Accuracy | Awards |",
        "|:----:|-------|------|-------:|:---------:|:--------:|--------|",
    ]
    for entry in leaderboard:
        awards_str = ", ".join(entry["awards_won"]) if entry["awards_won"] else "-"
        agent_display = entry["agent_id"]
        if entry["agent_id"] == champion_id:
            agent_display += " &#x1F3C6;"
        if entry["agent_id"] == strategist_id:
            agent_display += " &#x1F9E0;"
        lines.append(
            f"| {entry['rank']} | **{agent_display}** | {entry['role']} "
            f"| ${entry['total_wealth']:,.0f} "
            f"| {entry['strategic_score']:.3f} "
            f"| {entry['accuracy']:.0%} "
            f"| {awards_str} |"
        )
    lines.extend([
        "",
        "*&#x1F3C6; = Market Champion &nbsp; &#x1F9E0; = Master Strategist*",
    ])

    # --- Award Explanations ---
    champ = awards["market_champion"]
    strat = awards["master_strategist"]
    strat_breakdown = strat.get("breakdown", {})

    lines.extend([
        "",
        "---",
        "### Why These Winners?",
        "",
        f"**&#x1F3C6; Market Champion: {champion_id}**",
        f"- Highest **total wealth** (cash + inventory market value + compound goods)",
        f"- Total wealth: **${champ['total_wealth']:,.2f}** "
        f"(grew by ${champ.get('wealth_growth', 0):+,.2f} from starting capital)",
        f"- Won by accumulating the most valuable portfolio through smart "
        f"trading and production decisions",
        "",
        f"**&#x1F9E0; Master Strategist: {strategist_id}**",
        f"- Highest **strategic score** (weighted skill composite): "
        f"**{strat['strategic_score']:.4f}**",
        f"- Trade Efficiency: **{strat_breakdown.get('trade_efficiency', 0):.1%}** "
        f"(35% weight) — profitable trades / total trades",
        f"- Negotiation Mastery: **{strat_breakdown.get('negotiation_mastery', 0):.1%}** "
        f"(25% weight) — successful deals / total deals",
        f"- Cooperation Index: **{strat_breakdown.get('cooperation_index', 0):.1%}** "
        f"(20% weight) — compound production + coalition activity",
        f"- Event Adaptability: **{strat_breakdown.get('event_adaptability', 0):.1%}** "
        f"(20% weight) — actions responding to market events",
    ])

    # Market summary
    try:
        metrics = env.state.market_metrics
        coal_count = len(env.state.coalitions)
        lines.extend([
            "",
            "---",
            "### Market Summary",
            f"- **Total Trades:** {metrics.get('total_trades', 0)} "
            f"| **Volume:** ${metrics.get('total_volume', 0):,.1f}",
            f"- **Deals Negotiated:** {metrics.get('deals_negotiated', 0)} "
            f"| **Active Coalitions:** {coal_count}",
            f"- **Compound Goods Produced:** {metrics.get('compound_goods_produced', 0)}",
        ])
    except Exception:
        pass

    return "\n".join(lines)


# ===========================================================================
# Gradio Interface
# ===========================================================================
def create_dashboard():
    with gr.Blocks(
        title="Multi-Agent MarketForge",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"),
        css="""
        .main-header {text-align: center; margin-bottom: 20px;}
        .metric-box {border: 1px solid #ddd; border-radius: 8px; padding: 12px; text-align: center;}
        """
    ) as demo:

        gr.Markdown("""
        # Multi-Agent MarketForge
        ### A Multi-Commodity Market Simulation for LLM Agent Training

        **Four Pillars:** Cooperation (Supply Chains) | Competition (Double Auction) |
        Negotiation (Bilateral Deals) | Coalition Formation (Dynamic Alliances)

        Built with [OpenEnv v0.2.1](https://github.com/meta-pytorch/OpenEnv) |
        Designed for GRPO Training with [TRL](https://github.com/huggingface/trl)
        """)

        with gr.Tabs():
            # ---------------------------------------------------------------
            # Tab 1: Auto Simulation
            # ---------------------------------------------------------------
            with gr.TabItem("Simulation"):
                # ---- Model status banner ----
                model_status = (
                    f"Trained model loaded: **{DEFAULT_MODEL_PATH}** "
                    f"(device: {_model_device})"
                    if _loaded_model is not None
                    else (
                        f"Trained model **not available** "
                        f"({_model_load_error or 'torch not installed'}). "
                        f"Simulation will use heuristic agents only."
                    )
                )
                gr.Markdown(f"> {model_status}")

                with gr.Row():
                    num_rounds = gr.Slider(
                        minimum=10, maximum=100, value=30, step=5,
                        label="Number of Rounds"
                    )
                    agent_mode = gr.Radio(
                        choices=["Heuristic Only", "Trained LLM + Heuristic"],
                        value=(
                            "Trained LLM + Heuristic"
                            if _loaded_model is not None
                            else "Heuristic Only"
                        ),
                        label="Agent Mode",
                        interactive=_loaded_model is not None,
                    )

                with gr.Row():
                    llm_agents_dd = gr.Dropdown(
                        choices=list(AGENT_ROLES.keys()),
                        value=["trader_1", "speculator_1"],
                        multiselect=True,
                        label="LLM-Controlled Agents (select one or more)",
                        interactive=_loaded_model is not None,
                    )
                    run_btn = gr.Button(
                        "Run Simulation", variant="primary", size="lg"
                    )

                summary_output = gr.Markdown(label="Summary")
                chart_output = gr.Plot(label="Dashboard Charts")

                with gr.Row():
                    log_output = gr.Markdown(label="Activity Log")
                    event_output = gr.Markdown(label="Event Timeline")

                def _run_sim_wrapper(rounds, mode, llm_agents_list):
                    """Wrapper to convert multi-select list to comma string."""
                    llm_str = ", ".join(llm_agents_list) if llm_agents_list else ""
                    return run_simulation(int(rounds), mode, llm_str)

                run_btn.click(
                    fn=_run_sim_wrapper,
                    inputs=[num_rounds, agent_mode, llm_agents_dd],
                    outputs=[summary_output, chart_output, log_output, event_output],
                )

            # ---------------------------------------------------------------
            # Tab 2: Manual Agent Control
            # ---------------------------------------------------------------
            with gr.TabItem("Manual Control"):
                gr.Markdown("### Control individual agents manually")
                gr.Markdown(
                    "*Select an agent and action type below. Different actions "
                    "use different fields — see the field labels for guidance.*"
                )

                with gr.Row():
                    agent_dd = gr.Dropdown(
                        choices=list(AGENT_ROLES.keys()),
                        value="trader_1", label="Agent"
                    )
                    action_dd = gr.Dropdown(
                        choices=["buy", "sell", "produce", "negotiate",
                                 "propose_coalition", "join_coalition",
                                 "leave_coalition", "pass"],
                        value="buy", label="Action Type"
                    )

                with gr.Row():
                    commodity_dd = gr.Dropdown(
                        choices=COMMODITIES + list(COMPOUND_GOODS.keys()),
                        value="wheat", label="Commodity (buy/sell/produce)"
                    )
                    price_input = gr.Number(value=10.0, label="Price (buy/sell)")
                    qty_input = gr.Number(value=5, label="Quantity (buy/sell)")

                with gr.Row():
                    target_dd = gr.Dropdown(
                        choices=[""] + list(AGENT_ROLES.keys()),
                        value="", label="Target Agent (for negotiate)"
                    )
                    msg_input = gr.Textbox(
                        value="", label="Message (negotiate/coalition)",
                        placeholder="Enter negotiation message..."
                    )
                    coalition_input = gr.Textbox(
                        value="",
                        label="Coalition ID (join/leave coalition)",
                        placeholder="e.g. abc12345",
                    )

                step_btn = gr.Button("Execute Action", variant="primary")
                result_output = gr.Markdown(label="Result")

                gr.Markdown("---")
                leaderboard_output = gr.Markdown(
                    value="*Execute an action to see the live leaderboard "
                          "with award explanations.*",
                    label="Live Leaderboard"
                )

                step_btn.click(
                    fn=step_single_agent,
                    inputs=[agent_dd, action_dd, commodity_dd,
                            price_input, qty_input, target_dd, msg_input,
                            coalition_input],
                    outputs=[result_output, leaderboard_output],
                )

            # ---------------------------------------------------------------
            # Tab 3: Environment Info
            # ---------------------------------------------------------------
            with gr.TabItem("Environment Info"):
                gr.Markdown("""
## Multi-Commodity MarketForge

### OpenEnv Interface (how to "drive" the agent)

The environment implements the standard OpenEnv contract with **three methods**:

| Method | What it does | Analogy |
|--------|-------------|---------|
| `env.reset()` | Start a new game -- all agents get starting inventory | Starting the car engine |
| `env.step(action)` | Execute one action, get observation + reward | Pressing gas/brake/steering |
| `env.state` | View full game state (debug only) | Looking under the hood |

Think of this like a **driving video game**: the agent picks a control (buy/sell/negotiate/...),
the environment moves forward one tick, and the agent sees the result.

### Agent Controls (your "steering wheel")

| Control | What it does | Pillar |
|---------|-------------|--------|
| `buy` / `sell` | Trade on the open market | Competition |
| `produce` | Craft compound goods from raw materials | Cooperation |
| `negotiate` | Send a deal proposal to another agent | Negotiation |
| `accept_deal` / `reject_deal` | Respond to a deal offer | Negotiation |
| `propose_coalition` / `join_coalition` | Form alliances | Coalition |
| `pass` | Do nothing this turn | -- |

### Awards (how to WIN the game)

**Award 1 -- Market Champion (wealth):**
The agent with the highest `total_wealth` at game end wins.
`total_wealth = cash + market_value(inventory) + compound_value`

**Award 2 -- Master Strategist (skill):**
A composite quality score:
`score = 0.35*trade_efficiency + 0.25*negotiation_mastery + 0.20*cooperation + 0.20*event_adaptability`

### Scoring System

Every agent is scored on:
- **Cumulative Reward** -- sum of all per-step rewards
- **Total Wealth** -- final cash + inventory value
- **Strategic Score** -- quality of decisions (see Award 2)
- **Accuracy** -- valid actions / total actions taken

The **Leaderboard** ranks all agents at game end.

### Agent Roles

| Role | Objective | Key Interactions |
|------|-----------|-----------------|
| **Producer** | Maximize profit from selling raw commodities | Supply chains, Price competition |
| **Consumer** | Minimize cost of compound goods | Negotiation, Coalition buying |
| **Trader** | Profit from arbitrage | Double auction, Negotiation |
| **Speculator** | Profit from price movements | Event reaction, Market manipulation |

### Commodities & Recipes

**Raw:** Wheat ($10), Iron ($15), Timber ($12), Oil ($18)

**Compound Goods (value > sum of ingredients):**
- Bread ($35) = 2x Wheat + 1x Oil
- Tools ($50) = 2x Iron + 1x Timber
- Furniture ($45) = 2x Timber + 1x Oil

### The Four Pillars

1. **Cooperation** - Supply chain formation for compound goods. No agent can produce alone.
2. **Competition** - Continuous Double Auction with scarce resources.
3. **Negotiation** - Natural language bilateral deals with counteroffers.
4. **Coalition Formation** - Dynamic alliances with reputation tracking.

### Theory of Mind Levels

- **Level 0:** React to own state only
- **Level 1:** Model others' likely actions from behavior
- **Level 2:** Model what others believe about you

### Reward Structure

Mixed-motive payoffs: `u_i(a) = alpha * v_self + (1-alpha) * v_collective`

- Immediate: per-trade profit/loss
- Intermediate: contract completion, coalition synergy
- Episode-level: total wealth growth
- Shaped: strategic depth bonuses
                """)

            # ---------------------------------------------------------------
            # Tab 4: Training Pipeline
            # ---------------------------------------------------------------
            with gr.TabItem("Training Pipeline"):
                gr.Markdown("""
## GRPO Training Pipeline

### Architecture

```
[Prompt Dataset] --> [LLM Agent (Qwen 0.5B-3B)]
        |                    |
        v                    v
  [Market Observation]  [JSON Action Output]
        |                    |
        v                    v
  [OpenEnv Server]  <-- [Action Parsing]
        |
        v
  [Multi-Level Rewards]  --> [Awards Check]
        |                        |
        v                        v
  [GRPO Optimizer (TRL)]    [Accuracy Tracking]
        |                        |
        v                        v
  [Updated Policy]          [Iterative Improvement]
```

### How Training Improves the Model

The training loop makes the model **repeatedly more accurate**:

1. **Format Accuracy** -- Does the model output valid JSON? (starts ~40%, reaches 95%+)
2. **Action Quality** -- Does it pick smart actions? (tracked by strategic_score)
3. **Award Winning** -- Can it win Award 1 (wealth) or Award 2 (strategy)?
4. **Consistency** -- Does accuracy stay high across diverse scenarios?

### Reward Functions

| Function | Weight | Signal |
|----------|--------|--------|
| `reward_from_env` | 0.4 | Environment step reward (profit/loss) |
| `reward_valid_json` | 0.2 | Valid JSON action format (accuracy) |
| `reward_strategic_depth` | 0.2 | ToM actions (negotiate, coalition) |
| `reward_event_response` | 0.1 | Adapting to market events |
| `reward_wealth_growth` | 0.1 | Episode wealth change |

### Scoring (what the model optimises for)

| Metric | What it measures |
|--------|-----------------|
| `cumulative_reward` | Sum of all step rewards across the episode |
| `total_wealth` | Cash + inventory value at game end (Award 1) |
| `strategic_score` | Quality composite: efficiency + negotiation + cooperation + events (Award 2) |
| `accuracy` | Valid actions / total actions (model reliability) |

### Training Configuration

- **Model:** Qwen/Qwen2.5-0.5B-Instruct (or 1.7B for better results)
- **Algorithm:** GRPO with vLLM colocate mode
- **Dataset:** 3000 diverse market scenarios
- **Max turns per episode:** 6 (multi-turn interaction)
- **Environment:** OpenEnv v0.2.1 on HF Spaces
- **Win condition:** Agent must maximise Awards (wealth + strategic skill)
                """)

    return demo


# ===========================================================================
# Custom API routes for monitoring by market-forge-admin
# ===========================================================================

from fastapi import FastAPI
from fastapi.responses import JSONResponse

monitoring_app = FastAPI()

@monitoring_app.get("/health")
def health():
    return JSONResponse({"status": "ok", "timestamp": time.time()})

@monitoring_app.get("/state")
def get_state():
    """Expose the current MarketEnvironment state for the admin dashboard."""
    try:
        state = env.state
        return JSONResponse({
            "agents": {
                aid: {
                    "role": data.get("role", "unknown"),
                    "cash": data.get("cash", 0),
                    "reputation": data.get("reputation", 1.0),
                    "inventory": data.get("inventory", {}),
                    "last_action": data.get("last_action", {}),
                    "last_reward": data.get("last_reward", 0),
                    "actions_taken": data.get("actions_taken", 0),
                    "valid_actions": data.get("valid_actions", 0),
                    "cumulative_reward": data.get("cumulative_reward", 0),
                }
                for aid, data in state.agents.items()
            },
            "round": state.round_number,
            "current_event": state.current_event,
            "market_metrics": state.market_metrics,
            "price_history": {
                c: prices[-5:] if prices else []
                for c, prices in state.price_history.items()
            },
            "trade_history": [
                {
                    "buyer": t.get("buyer", ""),
                    "seller": t.get("seller", ""),
                    "commodity": t.get("commodity", ""),
                    "price": t.get("price", 0),
                    "quantity": t.get("quantity", 0),
                }
                for t in (state.trade_history[-20:] if state.trade_history else [])
            ],
            "coalitions": {
                cid: {
                    "proposer": coal.get("proposer", ""),
                    "members": coal.get("members", []),
                    "objective": coal.get("objective", ""),
                    "round_formed": coal.get("round_formed", 0),
                }
                for cid, coal in state.coalitions.items()
            },
            "last_modified": _state_last_modified,
        })
    except Exception as e:
        return JSONResponse({"agents": {}, "round": 0, "current_event": "", "error": str(e)})


# ===========================================================================
# Launch
# ===========================================================================
if __name__ == "__main__":
    demo = create_dashboard()
    # Mount monitoring API so admin dashboard can poll /health and /state
    import gradio as gr
    app = gr.mount_gradio_app(monitoring_app, demo, path="/")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
