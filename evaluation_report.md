# MarketForge Evaluation Report
## Trained LLM vs Baseline Agents

**Date:** 2026-03-10 12:21
**Model:** `./market-forge-agent`
**LLM-controlled agents:** trader_1, speculator_1
**Episodes per configuration:** 0

## Executive Summary

In head-to-head episodes (LLM agents vs rule-based agents in the same simulation):

| Metric | LLM Agents | Rule-Based Agents | Delta |
|--------|-----------|-------------------|-------|
| Avg Wealth | $N/A | $N/A | N/A (N/A) |
| Avg Growth | $N/A | $N/A | N/A |
| Avg Strategic Score | N/A | N/A | N/A |
| Avg Rank (1=best) | N/A | N/A | N/A |
| Champion Wins | 0/0 | - | - |
| Strategist Wins | 0/0 | - | - |

## Configuration Comparison

Average metrics across all agents in each configuration:

| Configuration | Avg Wealth | Avg Strategic | Avg Trades | Avg Volume |
|---------------|-----------|--------------|-----------|-----------|
| Random Baseline | $1948.1 | 0.4208 | 4.6 | $294.28 |
| Rule-Based Baseline | $1665.1 | 0.615 | 61.0 | $1854.6 |
| Trained LLM + Rule-Based | $N/A | N/A | N/A | $N/A |

## Per-Agent Breakdown (Trained LLM Configuration)

| Agent | Role | Strategy | Avg Wealth | Avg Growth | Avg Strategic | Avg Rank | Champion Wins | Strategist Wins |
|-------|------|----------|-----------|-----------|-------------|---------|--------------|----------------|

## Episode-by-Episode Results (Trained LLM Configuration)

## Methodology

Three configurations were tested with identical random seeds for fair comparison:

1. **Random Baseline** - All 8 agents use random action selection
2. **Rule-Based Baseline** - All 8 agents use hand-crafted heuristic strategies (producers sell specialty, consumers buy ingredients, traders exploit spreads, speculators react to events)
3. **Trained LLM** - trader_1, speculator_1 are controlled by the trained model (`./market-forge-agent`); remaining agents use rule-based strategy

Each configuration ran for the same number of episodes with deterministic seeds (1000, 1001, ...) so market events and initial conditions are identical across configurations.

**Winner determination:**
- Award 1 (Market Champion): Agent with highest total_wealth = cash + inventory_value + compound_value
- Award 2 (Master Strategist): Agent with highest strategic_score = 0.35*trade_efficiency + 0.25*negotiation_mastery + 0.20*cooperation_index + 0.20*event_adaptability
