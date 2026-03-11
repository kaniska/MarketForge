#!/usr/bin/env python3
"""
run_simulation.py - Run MarketForge with Trained LLM + Baseline Agents
======================================================================
This is the MISSING PIECE that connects the trained model to the live
environment. It runs a full simulation where:

  - Some agents are driven by the trained LLM (reads observation prompt,
    generates a JSON action)
  - Other agents use baseline strategies (random, rule-based) for comparison

Usage:
    # Run with trained model driving trader_1 + speculator_1 vs baselines
    python run_simulation.py

    # Run with a specific model path
    python run_simulation.py --model ./market-forge-agent

    # Run with all agents using the trained model
    python run_simulation.py --model ./market-forge-agent --llm-agents all

    # Run baseline-only (no LLM) for comparison
    python run_simulation.py --baseline-only

    # Custom rounds and seed
    python run_simulation.py --rounds 30 --seed 42
"""
import argparse
import json
import random
import re
import sys
import os
from dataclasses import asdict
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MarketAction, MarketObservation
from server.market_environment import (
    MarketEnvironment, COMMODITIES, COMPOUND_GOODS, BASE_PRICES, AGENT_ROLES,
)
from rewards import extract_action


# ======================================================================
# Agent Strategies
# ======================================================================

class BaseAgent:
    """Base class for all agent strategies."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.strategy_name = "base"

    def decide(self, obs: MarketObservation) -> MarketAction:
        raise NotImplementedError

    def _make_action(self, **kwargs) -> MarketAction:
        kwargs.setdefault("agent_id", self.agent_id)
        kwargs.setdefault("action_type", "pass")
        return MarketAction(**kwargs)


class RandomAgent(BaseAgent):
    """Baseline: picks a random legal action with random parameters."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.strategy_name = "random"

    def decide(self, obs: MarketObservation) -> MarketAction:
        legal = obs.legal_actions or ["buy", "sell", "pass"]
        action_type = random.choice(legal)

        if action_type in ("buy", "sell"):
            commodity = random.choice(COMMODITIES)
            price = round(random.uniform(3, 30), 1)
            quantity = random.randint(1, 10)
            return self._make_action(
                action_type=action_type,
                commodity=commodity,
                price=price,
                quantity=quantity,
            )
        elif action_type == "produce":
            good = random.choice(list(COMPOUND_GOODS.keys()))
            return self._make_action(action_type="produce", compound_good=good)
        elif action_type == "negotiate":
            others = [a for a in AGENT_ROLES if a != self.agent_id]
            target = random.choice(others)
            commodity = random.choice(COMMODITIES)
            return self._make_action(
                action_type="negotiate",
                target_agent=target,
                commodity=commodity,
                price=round(random.uniform(5, 25), 1),
                quantity=random.randint(1, 8),
                message=f"Want to trade {commodity}?",
            )
        elif action_type == "propose_coalition":
            return self._make_action(
                action_type="propose_coalition",
                message="Let's form a group",
            )
        elif action_type in ("accept_deal", "reject_deal"):
            if obs.pending_deals:
                deal = obs.pending_deals[0]
                return self._make_action(
                    action_type=action_type,
                    coalition_id=deal.get("deal_id", ""),
                )
            return self._make_action(action_type="pass")
        elif action_type in ("join_coalition", "leave_coalition"):
            if obs.coalitions:
                return self._make_action(
                    action_type=action_type,
                    coalition_id=obs.coalitions[0],
                )
            return self._make_action(action_type="pass")
        else:
            return self._make_action(action_type="pass")


class RuleBasedAgent(BaseAgent):
    """Baseline: simple heuristic strategy.

    - Producers sell their specialty when price > base price
    - Consumers buy ingredients they need
    - Traders buy low, sell high based on spread
    - Speculators react to events
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.strategy_name = "rule-based"

    def decide(self, obs: MarketObservation) -> MarketAction:
        role = obs.role

        if role == "producer":
            return self._producer_strategy(obs)
        elif role == "consumer":
            return self._consumer_strategy(obs)
        elif role == "trader":
            return self._trader_strategy(obs)
        elif role == "speculator":
            return self._speculator_strategy(obs)
        return self._make_action(action_type="pass")

    def _producer_strategy(self, obs: MarketObservation) -> MarketAction:
        # Sell specialty commodity if we have stock
        for commodity, qty in obs.inventory.items():
            if qty >= 5 and commodity in COMMODITIES:
                base = BASE_PRICES.get(commodity, 10)
                best_bid = obs.top_of_book.get(commodity, {}).get("best_bid", 0)
                sell_price = max(best_bid * 1.05, base * 1.1)
                sell_qty = min(qty // 2, 10)
                if sell_qty > 0:
                    return self._make_action(
                        action_type="sell",
                        commodity=commodity,
                        price=round(sell_price, 1),
                        quantity=sell_qty,
                    )
        return self._make_action(action_type="pass")

    def _consumer_strategy(self, obs: MarketObservation) -> MarketAction:
        # Try to produce compound goods if we have ingredients
        for good, recipe in COMPOUND_GOODS.items():
            has_all = all(
                obs.inventory.get(ing, 0) >= qty
                for ing, qty in recipe.items()
            )
            if has_all:
                return self._make_action(
                    action_type="produce", compound_good=good
                )

        # Buy cheapest ingredient we're missing
        for good, recipe in COMPOUND_GOODS.items():
            for ingredient, qty_needed in recipe.items():
                if obs.inventory.get(ingredient, 0) < qty_needed:
                    best_ask = obs.top_of_book.get(ingredient, {}).get("best_ask", 0)
                    buy_price = best_ask * 1.05 if best_ask > 0 else BASE_PRICES[ingredient] * 1.1
                    return self._make_action(
                        action_type="buy",
                        commodity=ingredient,
                        price=round(buy_price, 1),
                        quantity=qty_needed - obs.inventory.get(ingredient, 0),
                    )
        return self._make_action(action_type="pass")

    def _trader_strategy(self, obs: MarketObservation) -> MarketAction:
        # Look for the best spread to exploit
        best_spread = 0
        best_commodity = None

        for commodity in COMMODITIES:
            tob = obs.top_of_book.get(commodity, {})
            bid = tob.get("best_bid", 0)
            ask = tob.get("best_ask", 0)
            if bid > 0 and ask > 0 and bid > ask:
                spread = bid - ask
                if spread > best_spread:
                    best_spread = spread
                    best_commodity = commodity

        if best_commodity and best_spread > 1.0:
            tob = obs.top_of_book[best_commodity]
            if obs.inventory.get(best_commodity, 0) > 2:
                return self._make_action(
                    action_type="sell",
                    commodity=best_commodity,
                    price=round(tob["best_bid"] * 0.98, 1),
                    quantity=min(3, obs.inventory.get(best_commodity, 0)),
                )
            elif obs.cash > 50:
                return self._make_action(
                    action_type="buy",
                    commodity=best_commodity,
                    price=round(tob["best_ask"] * 1.02, 1),
                    quantity=3,
                )

        # Fallback: buy cheapest commodity
        if obs.cash > 100:
            cheapest = min(COMMODITIES, key=lambda c: obs.top_of_book.get(c, {}).get("best_ask", 999))
            ask = obs.top_of_book.get(cheapest, {}).get("best_ask", 0)
            if ask > 0:
                return self._make_action(
                    action_type="buy",
                    commodity=cheapest,
                    price=round(ask * 1.05, 1),
                    quantity=3,
                )
        return self._make_action(action_type="pass")

    def _speculator_strategy(self, obs: MarketObservation) -> MarketAction:
        event = obs.event.lower() if obs.event else ""

        # React to events
        if "drought" in event or "wheat" in event:
            if obs.cash > 100:
                return self._make_action(
                    action_type="buy", commodity="wheat",
                    price=round(BASE_PRICES["wheat"] * 1.3, 1), quantity=5,
                )
        elif "embargo" in event or "oil" in event:
            if obs.cash > 100:
                return self._make_action(
                    action_type="buy", commodity="oil",
                    price=round(BASE_PRICES["oil"] * 1.3, 1), quantity=5,
                )
        elif "surplus" in event or "timber" in event:
            if obs.inventory.get("timber", 0) > 3:
                return self._make_action(
                    action_type="sell", commodity="timber",
                    price=round(BASE_PRICES["timber"] * 0.9, 1),
                    quantity=obs.inventory.get("timber", 0),
                )

        # Default: buy whatever is cheapest, sell whatever is most expensive
        if obs.cash > 200:
            cheapest = min(
                COMMODITIES,
                key=lambda c: obs.last_trade_prices.get(c, BASE_PRICES[c])
            )
            price = obs.last_trade_prices.get(cheapest, BASE_PRICES[cheapest])
            return self._make_action(
                action_type="buy", commodity=cheapest,
                price=round(price * 1.05, 1), quantity=5,
            )

        # Sell most expensive holding
        holdings = {c: q for c, q in obs.inventory.items() if q > 0 and c in COMMODITIES}
        if holdings:
            most_expensive = max(
                holdings,
                key=lambda c: obs.last_trade_prices.get(c, BASE_PRICES[c])
            )
            price = obs.last_trade_prices.get(most_expensive, BASE_PRICES[most_expensive])
            return self._make_action(
                action_type="sell", commodity=most_expensive,
                price=round(price * 1.1, 1),
                quantity=min(5, holdings[most_expensive]),
            )
        return self._make_action(action_type="pass")


class TrainedLLMAgent(BaseAgent):
    """Agent driven by a trained HuggingFace model.

    Loads the GRPO-trained model (e.g. Qwen/Qwen2.5-0.5B-Instruct fine-tuned
    on MarketForge), feeds it the observation prompt, and parses the JSON
    action from the model's output.
    """

    def __init__(self, agent_id: str, model_path: str,
                 device: str = "auto", system_prompt: str = None):
        super().__init__(agent_id)
        self.strategy_name = f"trained-llm ({os.path.basename(model_path)})"
        self.model_path = model_path
        self.system_prompt = system_prompt or self._default_system_prompt()

        # Load model and tokenizer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print(f"  Loading trained model from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )
        self.model.eval()
        self.device = self.model.device
        print(f"  Model loaded on {self.device}")

    @staticmethod
    def _default_system_prompt() -> str:
        return (
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

    def decide(self, obs: MarketObservation) -> MarketAction:
        import torch

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": obs.prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse JSON action from model output
        parsed = extract_action(raw_output)
        if parsed and "action_type" in parsed:
            parsed["agent_id"] = self.agent_id
            # Filter to valid MarketAction fields
            valid_fields = set(MarketAction.__dataclass_fields__.keys())
            filtered = {k: v for k, v in parsed.items() if k in valid_fields}
            return MarketAction(**filtered)

        # Fallback if model output is unparseable
        return self._make_action(action_type="pass")


# ======================================================================
# Simulation Runner
# ======================================================================

def run_episode(
    env: MarketEnvironment,
    agents: Dict[str, BaseAgent],
    max_rounds: int = 30,
    seed: int = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one full episode of the market simulation.

    Each round, every agent gets an observation and decides an action.
    The environment processes actions one at a time. After all agents
    act, the round advances.

    Returns a results dict with awards, leaderboard, and per-agent stats.
    """
    obs = env.reset(max_rounds=max_rounds, seed=seed)
    agent_ids = list(env.state.agents.keys())

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Episode Start | {len(agent_ids)} agents | {max_rounds} rounds")
        print(f"{'='*60}")

    last_obs = {aid: None for aid in agent_ids}
    round_num = 0

    while True:
        for aid in agent_ids:
            # Get fresh observation for this agent
            agent_obs = env._make_observation(aid)

            if agent_obs.done:
                # Run one final step to trigger awards computation
                final_action = MarketAction(agent_id=aid, action_type="pass")
                final_obs = env.step(final_action)
                if final_obs.market_summary.get("game_over"):
                    return _extract_results(final_obs, agents, env)
                continue

            # Agent decides
            agent_strategy = agents.get(aid)
            if agent_strategy is None:
                # Default to rule-based for agents without an assigned strategy
                agent_strategy = RuleBasedAgent(aid)
                agents[aid] = agent_strategy

            action = agent_strategy.decide(agent_obs)
            step_obs = env.step(action)
            last_obs[aid] = step_obs

            if step_obs.done and step_obs.market_summary.get("game_over"):
                return _extract_results(step_obs, agents, env)

        new_round = env.state.round_number
        if new_round != round_num:
            round_num = new_round
            if verbose and round_num % 5 == 0:
                _print_round_summary(env, round_num, max_rounds)

    # Should not reach here, but safety fallback
    return _extract_results(last_obs[agent_ids[0]], agents, env)


def _extract_results(
    final_obs: MarketObservation,
    agents: Dict[str, BaseAgent],
    env: MarketEnvironment,
) -> Dict[str, Any]:
    """Extract structured results from the final observation."""
    awards = final_obs.market_summary.get("awards", {})
    leaderboard = final_obs.market_summary.get("leaderboard", [])

    # Annotate leaderboard with strategy names
    for entry in leaderboard:
        aid = entry["agent_id"]
        agent_strategy = agents.get(aid)
        entry["strategy"] = agent_strategy.strategy_name if agent_strategy else "unknown"

    return {
        "awards": awards,
        "leaderboard": leaderboard,
        "market_metrics": dict(env.state.market_metrics),
        "total_rounds": env.state.round_number,
    }


def _print_round_summary(env: MarketEnvironment, round_num: int, max_rounds: int):
    """Print a brief round summary."""
    metrics = env.state.market_metrics
    print(f"  Round {round_num:3d}/{max_rounds} | "
          f"trades: {metrics.get('total_trades', 0):4d} | "
          f"coalitions: {metrics.get('coalitions_formed', 0):2d} | "
          f"compounds: {metrics.get('compound_goods_produced', 0):2d}")


# ======================================================================
# Main
# ======================================================================

def build_agents(
    model_path: Optional[str],
    llm_agent_ids: List[str],
    baseline_type: str = "rule-based",
) -> Dict[str, BaseAgent]:
    """Create agent strategy instances.

    Args:
        model_path: Path to the trained model (None = all baselines)
        llm_agent_ids: Which agents the trained model should control
        baseline_type: "random" or "rule-based" for non-LLM agents
    """
    all_agent_ids = list(AGENT_ROLES.keys())
    agents = {}

    # Create LLM-driven agents
    llm_agent = None
    if model_path and llm_agent_ids:
        llm_agent = TrainedLLMAgent(
            agent_id=llm_agent_ids[0],  # shares model across agents
            model_path=model_path,
        )

    for aid in all_agent_ids:
        if model_path and aid in llm_agent_ids:
            if aid == llm_agent_ids[0]:
                agents[aid] = llm_agent
            else:
                # Share model, just change agent_id for actions
                shared = TrainedLLMAgent.__new__(TrainedLLMAgent)
                shared.__dict__.update(llm_agent.__dict__)
                shared.agent_id = aid
                agents[aid] = shared
        else:
            if baseline_type == "random":
                agents[aid] = RandomAgent(aid)
            else:
                agents[aid] = RuleBasedAgent(aid)

    return agents


def main():
    parser = argparse.ArgumentParser(description="Run MarketForge simulation")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (default: use base model ID)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model ID to use if --model not provided")
    parser.add_argument("--llm-agents", type=str, default="trader_1,speculator_1",
                        help="Comma-separated agent IDs for the LLM to control, or 'all'")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run with only baseline agents (no LLM)")
    parser.add_argument("--baseline-type", choices=["random", "rule-based"],
                        default="rule-based", help="Baseline strategy type")
    parser.add_argument("--rounds", type=int, default=30,
                        help="Number of rounds per episode")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-round output")
    args = parser.parse_args()

    # Determine model path
    model_path = None
    if not args.baseline_only:
        model_path = args.model or args.base_model

    # Determine which agents the LLM controls
    if args.llm_agents == "all":
        llm_agent_ids = list(AGENT_ROLES.keys())
    else:
        llm_agent_ids = [a.strip() for a in args.llm_agents.split(",")]

    print("MarketForge Simulation")
    print("-" * 40)
    if model_path:
        print(f"  LLM Model:    {model_path}")
        print(f"  LLM Controls: {', '.join(llm_agent_ids)}")
    else:
        print(f"  Mode: baseline-only ({args.baseline_type})")
    print(f"  Rounds: {args.rounds}")
    if args.seed is not None:
        print(f"  Seed: {args.seed}")

    # Build agents
    agents = build_agents(
        model_path=model_path,
        llm_agent_ids=llm_agent_ids if model_path else [],
        baseline_type=args.baseline_type,
    )

    # Print agent roster
    print(f"\n  Agent Roster:")
    for aid in AGENT_ROLES:
        strategy = agents[aid].strategy_name
        role = AGENT_ROLES[aid]["role"]
        print(f"    {aid:20s} | {role:12s} | {strategy}")

    # Run simulation
    env = MarketEnvironment()
    results = run_episode(
        env=env,
        agents=agents,
        max_rounds=args.rounds,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Print results
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")

    awards = results["awards"]
    if "market_champion" in awards:
        champ = awards["market_champion"]
        print(f"\n  Award 1 - Market Champion: {champ['agent_id']}")
        print(f"    Total Wealth: ${champ['total_wealth']:.2f}")
        print(f"    Wealth Growth: ${champ.get('wealth_growth', 0):.2f}")
        champ_strategy = agents.get(champ["agent_id"])
        if champ_strategy:
            print(f"    Strategy: {champ_strategy.strategy_name}")

    if "master_strategist" in awards:
        strat = awards["master_strategist"]
        print(f"\n  Award 2 - Master Strategist: {strat['agent_id']}")
        print(f"    Strategic Score: {strat['strategic_score']:.4f}")
        if "breakdown" in strat:
            bd = strat["breakdown"]
            print(f"    Trade Efficiency:    {bd.get('trade_efficiency', 0):.4f}")
            print(f"    Negotiation Mastery: {bd.get('negotiation_mastery', 0):.4f}")
            print(f"    Cooperation Index:   {bd.get('cooperation_index', 0):.4f}")
            print(f"    Event Adaptability:  {bd.get('event_adaptability', 0):.4f}")
        strat_strategy = agents.get(strat["agent_id"])
        if strat_strategy:
            print(f"    Strategy: {strat_strategy.strategy_name}")

    print(f"\n  Leaderboard:")
    print(f"  {'Rank':<5} {'Agent':<20} {'Strategy':<18} {'Wealth':>10} {'Growth':>10} {'Strategic':>10} {'Awards'}")
    print(f"  {'-'*95}")
    for entry in results.get("leaderboard", []):
        awards_str = ", ".join(entry.get("awards_won", [])) or "-"
        print(f"  {entry['rank']:<5} {entry['agent_id']:<20} {entry.get('strategy', '?'):<18} "
              f"${entry['total_wealth']:>9.2f} ${entry['wealth_growth']:>9.2f} "
              f"{entry['strategic_score']:>10.4f} {awards_str}")

    metrics = results.get("market_metrics", {})
    print(f"\n  Market Activity:")
    print(f"    Total Trades:     {metrics.get('total_trades', 0)}")
    print(f"    Total Volume:     ${metrics.get('total_volume', 0):.2f}")
    print(f"    Coalitions Formed:{metrics.get('coalitions_formed', 0)}")
    print(f"    Compounds Made:   {metrics.get('compound_goods_produced', 0)}")
    print(f"    Deals Negotiated: {metrics.get('deals_negotiated', 0)}")

    return results


if __name__ == "__main__":
    main()
