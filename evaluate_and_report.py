#!/usr/bin/env python3
"""
evaluate_and_report.py - Compare Trained LLM vs Baselines & Generate Report
============================================================================
Runs multiple episodes with different agent configurations and produces
a Markdown report + JSON data showing how the trained model improves
over random and rule-based baselines.

Usage:
    # Full evaluation with trained model
    python evaluate_and_report.py --model ./market-forge-agent

    # Use base (untrained) model as the LLM agent for comparison
    python evaluate_and_report.py --model Qwen/Qwen2.5-0.5B-Instruct

    # Quick evaluation (fewer episodes)
    python evaluate_and_report.py --model ./market-forge-agent --episodes 3

    # Custom output path
    python evaluate_and_report.py --model ./market-forge-agent --output my_report.md
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.market_environment import MarketEnvironment, AGENT_ROLES
from run_simulation import (
    BaseAgent, RandomAgent, RuleBasedAgent, TrainedLLMAgent,
    run_episode, build_agents,
)


# ======================================================================
# Evaluation Runner
# ======================================================================

def run_evaluation(
    model_path: str,
    num_episodes: int = 5,
    max_rounds: int = 30,
    llm_agent_ids: List[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a full evaluation comparing trained LLM vs baselines.

    Runs three configurations:
      1. random-only    — all agents use random strategy
      2. rule-based-only — all agents use rule-based heuristics
      3. trained-llm    — specified agents use the trained model,
                          remaining agents use rule-based

    Returns aggregated results for all configurations.
    """
    if llm_agent_ids is None:
        llm_agent_ids = ["trader_1", "speculator_1"]

    configs = {
        "random-baseline": {
            "model_path": None,
            "llm_agent_ids": [],
            "baseline_type": "random",
        },
        "rule-based-baseline": {
            "model_path": None,
            "llm_agent_ids": [],
            "baseline_type": "rule-based",
        },
        "trained-llm": {
            "model_path": model_path,
            "llm_agent_ids": llm_agent_ids,
            "baseline_type": "rule-based",
        },
    }

    all_results = {}

    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {config_name}")
        print(f"  Episodes: {num_episodes} x {max_rounds} rounds")
        print(f"{'='*60}")

        try:
            agents = build_agents(
                model_path=config["model_path"],
                llm_agent_ids=config["llm_agent_ids"],
                baseline_type=config["baseline_type"],
            )
        except (ImportError, ModuleNotFoundError, OSError) as e:
            print(f"\n  SKIPPED: {config_name} — {e}")
            print(f"  (Install torch + transformers to run LLM evaluation)")
            continue

        episode_results = []
        for ep in range(num_episodes):
            seed = 1000 + ep  # deterministic seeds for fair comparison
            print(f"\n  Episode {ep + 1}/{num_episodes} (seed={seed})")

            env = MarketEnvironment()
            # Rebuild per-episode to reset any shared state in agents
            if config_name != "trained-llm":
                agents = build_agents(
                    model_path=config["model_path"],
                    llm_agent_ids=config["llm_agent_ids"],
                    baseline_type=config["baseline_type"],
                )

            result = run_episode(
                env=env,
                agents=agents,
                max_rounds=max_rounds,
                seed=seed,
                verbose=verbose,
            )
            episode_results.append(result)

            # Print episode winner
            champ = result.get("awards", {}).get("market_champion", {})
            if champ:
                champ_agent = agents.get(champ["agent_id"])
                strat_name = champ_agent.strategy_name if champ_agent else "?"
                print(f"    Champion: {champ['agent_id']} ({strat_name}) "
                      f"wealth=${champ['total_wealth']:.2f}")

        all_results[config_name] = {
            "config": config,
            "episodes": episode_results,
            "summary": _aggregate_episodes(episode_results, llm_agent_ids),
        }

    return all_results


def _aggregate_episodes(
    episodes: List[Dict[str, Any]],
    llm_agent_ids: List[str],
) -> Dict[str, Any]:
    """Compute aggregate statistics across episodes."""
    n = len(episodes)
    if n == 0:
        return {}

    # Track per-agent metrics across episodes
    agent_wealth = defaultdict(list)
    agent_growth = defaultdict(list)
    agent_strategic = defaultdict(list)
    agent_ranks = defaultdict(list)
    champion_counts = defaultdict(int)
    strategist_counts = defaultdict(int)
    total_trades = []
    total_volume = []

    for ep in episodes:
        for entry in ep.get("leaderboard", []):
            aid = entry["agent_id"]
            agent_wealth[aid].append(entry["total_wealth"])
            agent_growth[aid].append(entry["wealth_growth"])
            agent_strategic[aid].append(entry["strategic_score"])
            agent_ranks[aid].append(entry["rank"])

        champ_id = ep.get("awards", {}).get("market_champion", {}).get("agent_id", "")
        strat_id = ep.get("awards", {}).get("master_strategist", {}).get("agent_id", "")
        if champ_id:
            champion_counts[champ_id] += 1
        if strat_id:
            strategist_counts[strat_id] += 1

        metrics = ep.get("market_metrics", {})
        total_trades.append(metrics.get("total_trades", 0))
        total_volume.append(metrics.get("total_volume", 0))

    # Compute averages
    avg = lambda lst: sum(lst) / len(lst) if lst else 0

    agent_summaries = {}
    for aid in agent_wealth:
        agent_summaries[aid] = {
            "avg_wealth": round(avg(agent_wealth[aid]), 2),
            "avg_growth": round(avg(agent_growth[aid]), 2),
            "avg_strategic": round(avg(agent_strategic[aid]), 4),
            "avg_rank": round(avg(agent_ranks[aid]), 2),
            "champion_wins": champion_counts.get(aid, 0),
            "strategist_wins": strategist_counts.get(aid, 0),
            "is_llm_agent": aid in llm_agent_ids,
        }

    # Compute LLM vs non-LLM aggregate
    llm_wealth = []
    baseline_wealth = []
    llm_growth = []
    baseline_growth = []
    llm_strategic = []
    baseline_strategic = []
    llm_ranks = []
    baseline_ranks = []

    for aid, summary in agent_summaries.items():
        if summary["is_llm_agent"]:
            llm_wealth.append(summary["avg_wealth"])
            llm_growth.append(summary["avg_growth"])
            llm_strategic.append(summary["avg_strategic"])
            llm_ranks.append(summary["avg_rank"])
        else:
            baseline_wealth.append(summary["avg_wealth"])
            baseline_growth.append(summary["avg_growth"])
            baseline_strategic.append(summary["avg_strategic"])
            baseline_ranks.append(summary["avg_rank"])

    llm_champion_wins = sum(
        champion_counts.get(aid, 0) for aid in llm_agent_ids
    )
    llm_strategist_wins = sum(
        strategist_counts.get(aid, 0) for aid in llm_agent_ids
    )

    return {
        "num_episodes": n,
        "agent_summaries": agent_summaries,
        "llm_agents": {
            "ids": llm_agent_ids,
            "avg_wealth": round(avg(llm_wealth), 2) if llm_wealth else None,
            "avg_growth": round(avg(llm_growth), 2) if llm_growth else None,
            "avg_strategic": round(avg(llm_strategic), 4) if llm_strategic else None,
            "avg_rank": round(avg(llm_ranks), 2) if llm_ranks else None,
            "champion_wins": llm_champion_wins,
            "strategist_wins": llm_strategist_wins,
        },
        "baseline_agents": {
            "avg_wealth": round(avg(baseline_wealth), 2) if baseline_wealth else None,
            "avg_growth": round(avg(baseline_growth), 2) if baseline_growth else None,
            "avg_strategic": round(avg(baseline_strategic), 4) if baseline_strategic else None,
            "avg_rank": round(avg(baseline_ranks), 2) if baseline_ranks else None,
        },
        "market_activity": {
            "avg_trades": round(avg(total_trades), 1),
            "avg_volume": round(avg(total_volume), 2),
        },
    }


# ======================================================================
# Report Generator
# ======================================================================

def generate_report(
    all_results: Dict[str, Any],
    model_path: str,
    llm_agent_ids: List[str],
    output_path: str = "evaluation_report.md",
    json_path: str = "evaluation_data.json",
) -> str:
    """Generate a Markdown report comparing all configurations."""

    random_summary = all_results.get("random-baseline", {}).get("summary", {})
    rule_summary = all_results.get("rule-based-baseline", {}).get("summary", {})
    llm_summary = all_results.get("trained-llm", {}).get("summary", {})

    n_episodes = llm_summary.get("num_episodes", 0)

    # --- Compute deltas ---
    def delta(llm_val, baseline_val):
        if llm_val is None or baseline_val is None:
            return "N/A"
        d = llm_val - baseline_val
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.2f}"

    def pct_delta(llm_val, baseline_val):
        if llm_val is None or baseline_val is None or baseline_val == 0:
            return "N/A"
        d = ((llm_val - baseline_val) / abs(baseline_val)) * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}%"

    llm_agg = llm_summary.get("llm_agents", {})
    rule_bl_agg = rule_summary.get("baseline_agents", {}) if rule_summary else {}
    rand_bl_agg = random_summary.get("baseline_agents", {}) if random_summary else {}

    # For the trained-llm config, get the LLM agents' stats vs the baselines
    # that played in the SAME episodes alongside the LLM
    llm_in_mixed = llm_summary.get("llm_agents", {})
    baseline_in_mixed = llm_summary.get("baseline_agents", {})

    lines = []
    lines.append("# MarketForge Evaluation Report")
    lines.append(f"## Trained LLM vs Baseline Agents")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Model:** `{model_path}`")
    lines.append(f"**LLM-controlled agents:** {', '.join(llm_agent_ids)}")
    lines.append(f"**Episodes per configuration:** {n_episodes}")
    lines.append("")

    # --- Executive Summary ---
    lines.append("## Executive Summary")
    lines.append("")

    wealth_vs_rule = delta(
        llm_in_mixed.get("avg_wealth"),
        baseline_in_mixed.get("avg_wealth"),
    )
    wealth_pct = pct_delta(
        llm_in_mixed.get("avg_wealth"),
        baseline_in_mixed.get("avg_wealth"),
    )
    strategic_vs_rule = delta(
        llm_in_mixed.get("avg_strategic"),
        baseline_in_mixed.get("avg_strategic"),
    )
    rank_vs_rule = delta(
        llm_in_mixed.get("avg_rank"),
        baseline_in_mixed.get("avg_rank"),
    )

    lines.append(f"In head-to-head episodes (LLM agents vs rule-based agents "
                  f"in the same simulation):")
    lines.append("")
    lines.append(f"| Metric | LLM Agents | Rule-Based Agents | Delta |")
    lines.append(f"|--------|-----------|-------------------|-------|")
    lines.append(f"| Avg Wealth | "
                  f"${llm_in_mixed.get('avg_wealth', 'N/A')} | "
                  f"${baseline_in_mixed.get('avg_wealth', 'N/A')} | "
                  f"{wealth_vs_rule} ({wealth_pct}) |")
    lines.append(f"| Avg Growth | "
                  f"${llm_in_mixed.get('avg_growth', 'N/A')} | "
                  f"${baseline_in_mixed.get('avg_growth', 'N/A')} | "
                  f"{delta(llm_in_mixed.get('avg_growth'), baseline_in_mixed.get('avg_growth'))} |")
    lines.append(f"| Avg Strategic Score | "
                  f"{llm_in_mixed.get('avg_strategic', 'N/A')} | "
                  f"{baseline_in_mixed.get('avg_strategic', 'N/A')} | "
                  f"{strategic_vs_rule} |")
    lines.append(f"| Avg Rank (1=best) | "
                  f"{llm_in_mixed.get('avg_rank', 'N/A')} | "
                  f"{baseline_in_mixed.get('avg_rank', 'N/A')} | "
                  f"{rank_vs_rule} |")
    lines.append(f"| Champion Wins | "
                  f"{llm_in_mixed.get('champion_wins', 0)}/{n_episodes} | "
                  f"- | - |")
    lines.append(f"| Strategist Wins | "
                  f"{llm_in_mixed.get('strategist_wins', 0)}/{n_episodes} | "
                  f"- | - |")
    lines.append("")

    # --- Configuration comparison ---
    lines.append("## Configuration Comparison")
    lines.append("")
    lines.append("Average metrics across all agents in each configuration:")
    lines.append("")

    def config_avg_wealth(summary):
        agents = summary.get("agent_summaries", {})
        if not agents:
            return "N/A"
        return round(sum(a["avg_wealth"] for a in agents.values()) / len(agents), 2)

    def config_avg_strategic(summary):
        agents = summary.get("agent_summaries", {})
        if not agents:
            return "N/A"
        return round(sum(a["avg_strategic"] for a in agents.values()) / len(agents), 4)

    lines.append(f"| Configuration | Avg Wealth | Avg Strategic | Avg Trades | Avg Volume |")
    lines.append(f"|---------------|-----------|--------------|-----------|-----------|")

    for name, summary in [("Random Baseline", random_summary),
                           ("Rule-Based Baseline", rule_summary),
                           ("Trained LLM + Rule-Based", llm_summary)]:
        market = summary.get("market_activity", {})
        lines.append(f"| {name} | "
                      f"${config_avg_wealth(summary)} | "
                      f"{config_avg_strategic(summary)} | "
                      f"{market.get('avg_trades', 'N/A')} | "
                      f"${market.get('avg_volume', 'N/A')} |")
    lines.append("")

    # --- Per-Agent Breakdown (Trained LLM config) ---
    lines.append("## Per-Agent Breakdown (Trained LLM Configuration)")
    lines.append("")
    lines.append(f"| Agent | Role | Strategy | Avg Wealth | Avg Growth | "
                  f"Avg Strategic | Avg Rank | Champion Wins | Strategist Wins |")
    lines.append(f"|-------|------|----------|-----------|-----------|"
                  f"-------------|---------|--------------|----------------|")

    agent_summaries = llm_summary.get("agent_summaries", {})
    sorted_agents = sorted(agent_summaries.items(),
                           key=lambda x: x[1]["avg_rank"])
    for aid, s in sorted_agents:
        role = AGENT_ROLES.get(aid, {}).get("role", "?")
        strat = "TRAINED LLM" if s["is_llm_agent"] else "rule-based"
        lines.append(f"| {aid} | {role} | {strat} | "
                      f"${s['avg_wealth']} | ${s['avg_growth']} | "
                      f"{s['avg_strategic']} | {s['avg_rank']} | "
                      f"{s['champion_wins']} | {s['strategist_wins']} |")
    lines.append("")

    # --- Episode-by-Episode Results ---
    lines.append("## Episode-by-Episode Results (Trained LLM Configuration)")
    lines.append("")

    llm_episodes = all_results.get("trained-llm", {}).get("episodes", [])
    for i, ep in enumerate(llm_episodes):
        champ = ep.get("awards", {}).get("market_champion", {})
        strat = ep.get("awards", {}).get("master_strategist", {})
        champ_is_llm = champ.get("agent_id", "") in llm_agent_ids
        strat_is_llm = strat.get("agent_id", "") in llm_agent_ids

        lines.append(f"### Episode {i + 1}")
        lines.append(f"- **Champion:** {champ.get('agent_id', '?')} "
                      f"(wealth=${champ.get('total_wealth', 0):.2f}) "
                      f"{'**[LLM]**' if champ_is_llm else '[baseline]'}")
        lines.append(f"- **Strategist:** {strat.get('agent_id', '?')} "
                      f"(score={strat.get('strategic_score', 0):.4f}) "
                      f"{'**[LLM]**' if strat_is_llm else '[baseline]'}")

        lines.append(f"- Leaderboard:")
        lines.append(f"  | Rank | Agent | Wealth | Strategic | Awards |")
        lines.append(f"  |------|-------|--------|-----------|--------|")
        for entry in ep.get("leaderboard", []):
            is_llm = entry["agent_id"] in llm_agent_ids
            marker = " **[LLM]**" if is_llm else ""
            awards_str = ", ".join(entry.get("awards_won", [])) or "-"
            lines.append(f"  | {entry['rank']} | {entry['agent_id']}{marker} | "
                          f"${entry['total_wealth']:.2f} | "
                          f"{entry['strategic_score']:.4f} | {awards_str} |")
        lines.append("")

    # --- Methodology ---
    lines.append("## Methodology")
    lines.append("")
    lines.append("Three configurations were tested with identical random seeds "
                  "for fair comparison:")
    lines.append("")
    lines.append("1. **Random Baseline** - All 8 agents use random action selection")
    lines.append("2. **Rule-Based Baseline** - All 8 agents use hand-crafted "
                  "heuristic strategies (producers sell specialty, consumers buy "
                  "ingredients, traders exploit spreads, speculators react to events)")
    lines.append(f"3. **Trained LLM** - {', '.join(llm_agent_ids)} are controlled "
                  f"by the trained model (`{model_path}`); remaining agents use "
                  f"rule-based strategy")
    lines.append("")
    lines.append("Each configuration ran for the same number of episodes with "
                  "deterministic seeds (1000, 1001, ...) so market events and "
                  "initial conditions are identical across configurations.")
    lines.append("")
    lines.append("**Winner determination:**")
    lines.append("- Award 1 (Market Champion): Agent with highest "
                  "total_wealth = cash + inventory_value + compound_value")
    lines.append("- Award 2 (Master Strategist): Agent with highest "
                  "strategic_score = 0.35*trade_efficiency + "
                  "0.25*negotiation_mastery + 0.20*cooperation_index + "
                  "0.20*event_adaptability")
    lines.append("")

    report_text = "\n".join(lines)

    # Write Markdown report
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {output_path}")

    # Write raw JSON data
    # Make JSON serializable
    json_data = _make_serializable(all_results)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"Raw data saved to: {json_path}")

    return report_text


def _make_serializable(obj):
    """Recursively convert non-serializable types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model vs baselines and generate report"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Path to trained model or HF model ID")
    parser.add_argument("--llm-agents", type=str, default="trader_1,speculator_1",
                        help="Comma-separated agent IDs for the LLM to control")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes per configuration")
    parser.add_argument("--rounds", type=int, default=30,
                        help="Rounds per episode")
    parser.add_argument("--output", type=str, default="evaluation_report.md",
                        help="Output Markdown report path")
    parser.add_argument("--json-output", type=str, default="evaluation_data.json",
                        help="Output JSON data path")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-round output")
    args = parser.parse_args()

    llm_agent_ids = [a.strip() for a in args.llm_agents.split(",")]

    print("MarketForge Evaluation")
    print("=" * 60)
    print(f"  Model:     {args.model}")
    print(f"  LLM Agents:{', '.join(llm_agent_ids)}")
    print(f"  Episodes:  {args.episodes} per configuration")
    print(f"  Rounds:    {args.rounds} per episode")
    print(f"  Output:    {args.output}")

    start_time = time.time()

    all_results = run_evaluation(
        model_path=args.model,
        num_episodes=args.episodes,
        max_rounds=args.rounds,
        llm_agent_ids=llm_agent_ids,
        verbose=args.verbose,
    )

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.1f}s")

    report = generate_report(
        all_results=all_results,
        model_path=args.model,
        llm_agent_ids=llm_agent_ids,
        output_path=args.output,
        json_path=args.json_output,
    )

    # Print summary to console
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)

    llm_config = all_results.get("trained-llm", {}).get("summary", {})
    llm_agg = llm_config.get("llm_agents", {})
    bl_agg = llm_config.get("baseline_agents", {})
    n = llm_config.get("num_episodes", 0)

    if llm_agg.get("avg_wealth") is not None and bl_agg.get("avg_wealth") is not None:
        wealth_diff = llm_agg["avg_wealth"] - bl_agg["avg_wealth"]
        wealth_pct = (wealth_diff / abs(bl_agg["avg_wealth"])) * 100 if bl_agg["avg_wealth"] != 0 else 0
        print(f"\n  LLM vs Baseline (head-to-head in same episodes):")
        print(f"    Wealth:    LLM ${llm_agg['avg_wealth']:.2f} vs Baseline ${bl_agg['avg_wealth']:.2f}"
              f"  ({'+' if wealth_diff >= 0 else ''}{wealth_pct:.1f}%)")
        print(f"    Strategic: LLM {llm_agg['avg_strategic']:.4f} vs Baseline {bl_agg['avg_strategic']:.4f}")
        print(f"    Rank:      LLM {llm_agg['avg_rank']:.1f} vs Baseline {bl_agg['avg_rank']:.1f}")
        print(f"    Champion wins:   {llm_agg['champion_wins']}/{n}")
        print(f"    Strategist wins: {llm_agg['strategist_wins']}/{n}")

    print(f"\n  Report: {args.output}")
    print(f"  Data:   {args.json_output}")


if __name__ == "__main__":
    main()
