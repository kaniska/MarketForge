"""
MarketForge - Reward Functions for GRPO Training
==============================================================
Multi-level reward signals following the use case design:
  - Immediate: per-action trade profit/loss
  - Intermediate: contract completion, coalition formation, event response
  - Episode-level: total wealth, market share, reputation
  - Shaped: bonuses for strategic depth, theory-of-mind behavior
"""
import json
import re
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Primary reward: environment feedback from step()
# ---------------------------------------------------------------------------
def reward_from_env(completions, **kwargs) -> List[float]:
    """Primary reward from the environment step() call."""
    rewards = kwargs.get("env_reward", [])
    if rewards:
        return [float(r) for r in rewards]
    return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# Format reward: valid JSON action
# ---------------------------------------------------------------------------
def reward_valid_json(completions, **kwargs) -> List[float]:
    """Reward for producing a valid JSON action with required fields."""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else str(c)
        try:
            # Extract JSON from text (may have surrounding text)
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                d = json.loads(json_match.group())
                if "action_type" in d:
                    valid_types = {
                        "buy", "sell", "produce", "negotiate",
                        "accept_deal", "reject_deal",
                        "propose_coalition", "join_coalition",
                        "leave_coalition", "pass",
                    }
                    if d["action_type"] in valid_types:
                        rewards.append(0.3)
                    else:
                        rewards.append(0.05)  # valid JSON but wrong type
                else:
                    rewards.append(0.0)
            else:
                rewards.append(-0.2)
        except Exception:
            rewards.append(-0.2)
    return rewards


# ---------------------------------------------------------------------------
# Strategic depth reward
# ---------------------------------------------------------------------------
def reward_strategic_depth(completions, **kwargs) -> List[float]:
    """Reward for using advanced actions that require theory-of-mind."""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else str(c)
        try:
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                d = json.loads(json_match.group())
                atype = d.get("action_type", "")

                # Highest reward: actions requiring ToM reasoning
                if atype in ("negotiate", "propose_coalition", "accept_deal"):
                    score = 0.3
                    # Extra bonus for including a message (shows ToM)
                    if d.get("message") and len(d.get("message", "")) > 10:
                        score += 0.1
                    rewards.append(score)
                elif atype == "produce":
                    rewards.append(0.25)  # cooperation pillar
                elif atype in ("buy", "sell"):
                    # Reward reasonable pricing
                    price = float(d.get("price", 0))
                    qty = int(d.get("quantity", 0))
                    if 1 <= price <= 100 and 1 <= qty <= 50:
                        rewards.append(0.15)
                    else:
                        rewards.append(0.05)
                elif atype == "join_coalition":
                    rewards.append(0.2)
                elif atype == "regulate":
                    rewards.append(0.2)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Wealth growth reward (episode-level)
# ---------------------------------------------------------------------------
def reward_wealth_growth(completions, **kwargs) -> List[float]:
    """Reward based on wealth growth over the episode."""
    wealth_changes = kwargs.get("wealth_change", [])
    if wealth_changes:
        return [
            min(1.0, max(-0.5, float(w) / 500.0))
            for w in wealth_changes
        ]
    return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# Cooperation reward
# ---------------------------------------------------------------------------
def reward_cooperation(completions, **kwargs) -> List[float]:
    """Reward for cooperative behaviors (deals, coalitions, production)."""
    coop_scores = kwargs.get("cooperation_score", [])
    if coop_scores:
        return [float(s) for s in coop_scores]
    return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# Event response reward
# ---------------------------------------------------------------------------
def reward_event_response(completions, **kwargs) -> List[float]:
    """Reward for adapting strategy to market events."""
    rewards = []
    events = kwargs.get("current_events", [])
    for i, c in enumerate(completions):
        text = c[0]["content"] if isinstance(c, list) else str(c)
        try:
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                d = json.loads(json_match.group())
                event = events[i] if i < len(events) else ""
                commodity = d.get("commodity", "")

                # Reward strategic response to events
                if "drought" in event.lower() and "wheat" in event.lower():
                    if d.get("action_type") == "buy" and commodity == "wheat":
                        rewards.append(0.3)  # buying scarce resource
                    elif d.get("action_type") == "sell" and commodity == "wheat":
                        rewards.append(0.2)  # selling at high price
                    else:
                        rewards.append(0.05)
                elif "embargo" in event.lower() and "oil" in event.lower():
                    if commodity == "oil":
                        rewards.append(0.25)
                    else:
                        rewards.append(0.05)
                else:
                    rewards.append(0.05)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Extract action from LLM output
# ---------------------------------------------------------------------------
def extract_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON action from LLM output text."""
    try:
        # Try to find JSON object in text
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        # Try the whole text
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None
