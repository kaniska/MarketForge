"""
Multi-Agent MarketForge - Environment Server Implementation
=============================================================
A multi-commodity market simulation with cooperation, competition,
negotiation, and coalition formation. Designed for OpenEnv v0.2.1.

Four Pillars:
  1. Cooperation  - Supply chain formation for compound goods
  2. Competition  - Continuous Double Auction with resource scarcity
  3. Negotiation  - Natural language bilateral deal-making
  4. Coalition    - Dynamic alliance formation with reputation

OpenEnv Contract
-----------------
This Environment implements the standard OpenEnv interface:

    reset(**kwargs)  -> MarketObservation
        Resets all state and returns the initial observation.
        Think of it like starting a new game -- every agent goes
        back to their starting inventory, cash, and reputation.

    step(action: MarketAction) -> MarketObservation
        Accepts ONE agent action, executes it inside the market,
        and returns the resulting observation (including reward,
        done flag, and a rich text prompt for the next decision).
        Think of it like pressing a button in a driving game:
        the agent picks an action (buy/sell/negotiate/etc.)
        and the environment moves forward one tick.

    state -> MarketState  (property)
        Returns the full server-side game state -- used for
        debugging, visualization, and evaluation but *never*
        shown to agents (they only see MarketObservation).

Scoring & Awards
-----------------
The environment tracks two distinct Awards that determine who
"wins the game":

  AWARD 1 -- Market Champion (wealth-based):
      The agent with the highest total_wealth at game end wins.
      total_wealth = cash + market_value(inventory) + compound_value

  AWARD 2 -- Master Strategist (skill-based):
      A composite score measuring *how well* the agent played:
        strategic_score = 0.35 * trade_efficiency
                        + 0.25 * negotiation_mastery
                        + 0.20 * cooperation_index
                        + 0.20 * event_adaptability

      This second award ensures that an agent who makes
      *intelligent* moves is recognized, even if another agent
      accumulated more wealth through lucky starting conditions.

Both awards are computed at game end and are returned in the
final observation so LLM agents can learn to optimize for them.
"""
import uuid
import copy
import random
import math
import json
from typing import Dict, List, Any, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import MarketAction, MarketObservation, MarketState

# Try to import OpenEnv base class
try:
    from openenv_core.env_server import Environment
except ImportError:
    try:
        from openenv.core.env_server import Environment
    except ImportError:
        class Environment:
            """Fallback base class matching the OpenEnv v0.2.1 contract.

            Every OpenEnv environment must implement three methods:
              reset()  -- reset the game and return the first observation
              step()   -- accept an action, execute it, return observation
              state    -- property that exposes the full game state
            """
            SUPPORTS_CONCURRENT_SESSIONS = True
            def reset(self, **kwargs): ...
            def step(self, action, **kwargs): ...
            @property
            def state(self): ...


# ==========================================================================
# Constants
# ==========================================================================
COMMODITIES = ["wheat", "iron", "timber", "oil"]

COMPOUND_GOODS = {
    "bread":     {"wheat": 2, "oil": 1},
    "tools":     {"iron": 2, "timber": 1},
    "furniture": {"timber": 2, "oil": 1},
}

# Base prices for valuation
BASE_PRICES = {"wheat": 10, "iron": 15, "timber": 12, "oil": 18}
COMPOUND_PRICES = {"bread": 35, "tools": 50, "furniture": 45}

EVENTS = [
    {"text": "Drought reduces wheat supply by 30%.",
     "effects": {"wheat": 0.7}, "severity": "high"},
    {"text": "Iron mine collapse -- iron production halved this round.",
     "effects": {"iron": 0.5}, "severity": "high"},
    {"text": "Timber surplus -- prices expected to drop.",
     "effects": {"timber": 1.4}, "severity": "medium"},
    {"text": "Oil embargo -- oil supply restricted.",
     "effects": {"oil": 0.4}, "severity": "high"},
    {"text": "Trade festival -- all transaction costs waived this round.",
     "effects": {}, "severity": "low"},
    {"text": "Consumer demand for furniture surges 50%.",
     "effects": {"timber": 0.9, "oil": 0.9}, "severity": "medium"},
    {"text": "New trade route opens -- all commodities +10% supply.",
     "effects": {"wheat": 1.1, "iron": 1.1, "timber": 1.1, "oil": 1.1}, "severity": "low"},
    {"text": "Calm markets -- no special event.",
     "effects": {}, "severity": "low"},
    {"text": "Speculator rumor: Oil prices expected to skyrocket next round.",
     "effects": {}, "severity": "medium"},
    {"text": "Government subsidy announced for tool production.",
     "effects": {"iron": 1.1, "timber": 1.1}, "severity": "medium"},
    {"text": "Storm damages wheat and timber warehouses.",
     "effects": {"wheat": 0.8, "timber": 0.8}, "severity": "medium"},
    {"text": "International trade deal signed -- demand up across all goods.",
     "effects": {"wheat": 0.95, "iron": 0.95, "timber": 0.95, "oil": 0.95}, "severity": "medium"},
]

AGENT_ROLES = {
    "producer_wheat":  {"role": "producer",   "specialty": "wheat",  "cash": 500,
                        "inventory": {"wheat": 40, "iron": 0, "timber": 0, "oil": 0},
                        "alpha": 0.7},
    "producer_iron":   {"role": "producer",   "specialty": "iron",   "cash": 500,
                        "inventory": {"wheat": 0, "iron": 40, "timber": 0, "oil": 0},
                        "alpha": 0.7},
    "producer_timber": {"role": "producer",   "specialty": "timber", "cash": 500,
                        "inventory": {"wheat": 0, "iron": 0, "timber": 40, "oil": 0},
                        "alpha": 0.7},
    "producer_oil":    {"role": "producer",   "specialty": "oil",    "cash": 500,
                        "inventory": {"wheat": 0, "iron": 0, "timber": 0, "oil": 40},
                        "alpha": 0.7},
    "consumer_1":      {"role": "consumer",   "specialty": "bread",  "cash": 1000,
                        "inventory": {"wheat": 0, "iron": 0, "timber": 0, "oil": 0},
                        "alpha": 0.5},
    "consumer_2":      {"role": "consumer",   "specialty": "tools",  "cash": 1000,
                        "inventory": {"wheat": 0, "iron": 0, "timber": 0, "oil": 0},
                        "alpha": 0.5},
    "trader_1":        {"role": "trader",     "specialty": "",       "cash": 800,
                        "inventory": {"wheat": 5, "iron": 5, "timber": 5, "oil": 5},
                        "alpha": 0.8},
    "speculator_1":    {"role": "speculator", "specialty": "",       "cash": 1500,
                        "inventory": {"wheat": 0, "iron": 0, "timber": 0, "oil": 0},
                        "alpha": 0.9},
}


class MarketEnvironment(Environment):
    """Multi-agent commodity market with cooperation, competition,
    negotiation, and coalition formation.

    Think of this like a video game where each agent "drives" a trading
    company.  Just as a driving game gives you a steering wheel, gas, and
    brake, this environment gives each agent a clear set of *controls*:

        buy / sell      -- the gas & brake of trading
        negotiate       -- talk to other drivers to form deals
        propose/join/leave coalition -- team up with others
        produce         -- craft compound goods from raw materials
        pass            -- do nothing this turn

    At every step the agent receives a MarketObservation that tells it
    exactly what it can see (prices, inventory, messages, events) and
    what actions are legal.  The agent picks ONE action, the environment
    executes it, and returns a new observation + reward.

    AWARDS (two ways to win):
        Award 1 -- Market Champion: highest total wealth at game end
        Award 2 -- Master Strategist: best strategic_score composite

    SCORING:
        Each agent accumulates a score that combines:
          - cumulative_reward  (sum of per-step rewards)
          - total_wealth       (cash + inventory value)
          - strategic_score    (composite quality metric)
        The leaderboard ranks agents by total_wealth first, then by
        strategic_score as tie-breaker.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state = MarketState()
        self._message_queues: Dict[str, list] = {}
        self._pending_deals: Dict[str, Dict] = {}
        self._production_costs: Dict[str, float] = {}
        # Tracking for accuracy / iterative improvement
        self._accuracy_history: List[Dict[str, float]] = []

    # ==================================================================
    # reset()  -- Start a brand-new game
    # ==================================================================
    def reset(self, seed=None, episode_id=None, **kwargs) -> MarketObservation:
        """Reset the environment to a fresh starting state.

        This is called at the beginning of every episode.  It:
          1. Creates all agents with their starting cash & inventory
          2. Clears the order book, trade history, and coalitions
          3. Initialises the scoring and award tracking systems
          4. Returns the first observation for the default agent

        Keyword args:
            seed       -- random seed for reproducibility
            max_rounds -- number of rounds in this episode (default 50)

        Returns:
            MarketObservation for the first agent
        """
        if seed is not None:
            random.seed(seed)

        self._state = MarketState(
            episode_id=episode_id or str(uuid.uuid4()),
            max_rounds=kwargs.get("max_rounds", 50),
        )
        self._state.agents = copy.deepcopy(AGENT_ROLES)

        # Initialise per-agent tracking for scoring & awards
        for aid in self._state.agents:
            agent = self._state.agents[aid]
            agent["reputation"] = 1.0
            agent["total_trades"] = 0
            agent["successful_deals"] = 0
            agent["contracts_broken"] = 0
            agent["compound_produced"] = 0
            # --- Scoring fields ---
            agent["cumulative_reward"] = 0.0
            agent["starting_wealth"] = (
                agent["cash"]
                + sum(agent["inventory"].get(c, 0) * BASE_PRICES[c]
                      for c in COMMODITIES)
            )
            agent["actions_taken"] = 0
            agent["valid_actions"] = 0
            agent["profitable_trades"] = 0
            agent["event_responsive_actions"] = 0
            agent["rounds_active"] = 0

        self._state.order_books = {c: {"bids": [], "asks": []} for c in COMMODITIES}
        self._state.trade_history = []
        self._state.price_history = {c: [BASE_PRICES[c]] for c in COMMODITIES}
        self._state.coalitions = {}
        self._state.pending_deals = {}
        self._state.current_event = "Market opens. Welcome traders!"
        self._state.event_effects = {}
        self._state.round_number = 0
        self._state.step_count = 0
        self._state.round_rewards = {aid: [] for aid in self._state.agents}
        self._state.market_metrics = {
            "total_trades": 0, "total_volume": 0,
            "coalitions_formed": 0, "deals_negotiated": 0,
            "compound_goods_produced": 0,
        }

        self._message_queues = {aid: [] for aid in self._state.agents}
        self._pending_deals = {}
        self._production_costs = {c: BASE_PRICES[c] * 0.6 for c in COMMODITIES}
        self._accuracy_history = []

        return self._make_observation("consumer_1")

    # ==================================================================
    # step()  -- Execute one agent action
    # ==================================================================
    def step(self, action: MarketAction, **kwargs) -> MarketObservation:
        """Execute one action and return the resulting observation.

        This is the core game loop -- equivalent to pressing a button
        in a driving game:
          1. Validate the action and the acting agent
          2. Dispatch to the correct handler (buy/sell/negotiate/...)
          3. Compute the step reward
          4. Advance the round clock if all agents have acted
          5. Build and return a fresh MarketObservation

        The returned observation contains:
          - reward     : immediate reward for this action
          - done       : True if the game is over
          - score      : running cumulative score
          - prompt     : text description for LLM agents
          - legal_actions : list of valid action_types

        If done=True, the observation also contains:
          - awards     : dict with Award 1 (champion) and Award 2 (strategist)
          - leaderboard: ranked list of all agents with final scores
        """
        aid = action.agent_id or "trader_1"

        # Auto-register unknown agents (lets external LLMs join)
        if aid not in self._state.agents:
            self._state.agents[aid] = {
                "role": "trader", "specialty": "", "cash": 800,
                "inventory": {c: 5 for c in COMMODITIES},
                "reputation": 1.0, "alpha": 0.8,
                "total_trades": 0, "successful_deals": 0,
                "contracts_broken": 0, "compound_produced": 0,
                "cumulative_reward": 0.0,
                "starting_wealth": 800 + 5 * sum(BASE_PRICES[c] for c in COMMODITIES),
                "actions_taken": 0, "valid_actions": 0,
                "profitable_trades": 0, "event_responsive_actions": 0,
                "rounds_active": 0,
            }
            self._message_queues[aid] = []
            self._state.round_rewards[aid] = []

        agent = self._state.agents[aid]
        reward = 0.0

        # --- Dispatch to action handler ---
        action_handlers = {
            "buy":               self._handle_buy,
            "sell":              self._handle_sell,
            "produce":           self._handle_produce,
            "negotiate":         self._handle_negotiate,
            "accept_deal":       self._handle_accept_deal,
            "reject_deal":       self._handle_reject_deal,
            "propose_coalition": self._handle_propose_coalition,
            "join_coalition":    self._handle_join_coalition,
            "leave_coalition":   self._handle_leave_coalition,
            "regulate":          self._handle_regulate,
        }

        handler = action_handlers.get(action.action_type)
        is_valid = handler is not None or action.action_type == "pass"

        if handler:
            reward = handler(action)
        elif action.action_type == "pass":
            reward = -0.01  # small penalty for inaction
        else:
            reward = -0.15  # invalid action type

        # --- Update scoring counters ---
        agent["actions_taken"] = agent.get("actions_taken", 0) + 1
        if is_valid:
            agent["valid_actions"] = agent.get("valid_actions", 0) + 1
        agent["cumulative_reward"] = agent.get("cumulative_reward", 0.0) + reward

        # Track event-responsive behaviour
        if self._state.event_effects and action.action_type not in ("pass", ""):
            affected_commodities = set(self._state.event_effects.keys())
            if action.commodity in affected_commodities:
                agent["event_responsive_actions"] = (
                    agent.get("event_responsive_actions", 0) + 1
                )

        self._state.step_count += 1
        num_agents = max(len(self._state.agents), 1)
        if self._state.step_count % num_agents == 0:
            self._advance_round()

        # Track reward history
        self._state.round_rewards.setdefault(aid, []).append(reward)

        # Build observation
        obs = self._make_observation(aid)
        obs.reward = reward
        obs.done = self._state.round_number >= self._state.max_rounds

        # --- If game is over, compute Awards and Leaderboard ---
        if obs.done:
            awards = self.compute_awards()
            leaderboard = self.compute_leaderboard()
            obs.market_summary["awards"] = awards
            obs.market_summary["leaderboard"] = leaderboard
            obs.market_summary["game_over"] = True

        return obs

    # ==================================================================
    # state  -- Full server-side game state (property)
    # ==================================================================
    @property
    def state(self) -> MarketState:
        """Return the full game state.

        This is for debugging, visualization, and evaluation only.
        Agents should NEVER see this -- they only see the
        MarketObservation returned by step().

        The state includes:
          - All agent data (cash, inventory, reputation, scores)
          - Full order books (not just top-of-book)
          - Complete trade history
          - All coalitions and pending deals
          - Market metrics and price history
        """
        return self._state

    # ==================================================================
    # AWARDS -- Determining who wins the game
    # ==================================================================
    def compute_awards(self) -> Dict[str, Any]:
        """Compute the two game Awards.

        Award 1 -- Market Champion:
            The agent with the highest total_wealth.
            total_wealth = cash + market_value(inventory) + compound_value

        Award 2 -- Master Strategist:
            The agent with the highest strategic_score.
            strategic_score = weighted combination of:
              - trade_efficiency    (profitable trades / total trades)
              - negotiation_mastery (successful deals / total deals)
              - cooperation_index   (compound goods + coalition activity)
              - event_adaptability  (event-responsive actions / events)

        Returns:
            dict with "market_champion" and "master_strategist" entries,
            each containing agent_id, score, and breakdown.
        """
        # --- Award 1: Market Champion (wealth-based) ---
        wealth_scores = {}
        for aid, agent in self._state.agents.items():
            wealth_scores[aid] = self._calculate_wealth(agent)

        champion_id = max(wealth_scores, key=wealth_scores.get)

        # --- Award 2: Master Strategist (skill-based) ---
        strategic_scores = {}
        for aid, agent in self._state.agents.items():
            strategic_scores[aid] = self._compute_strategic_score(aid, agent)

        strategist_id = max(strategic_scores, key=strategic_scores.get)

        return {
            "market_champion": {
                "agent_id": champion_id,
                "total_wealth": round(wealth_scores[champion_id], 2),
                "wealth_growth": round(
                    wealth_scores[champion_id]
                    - agent.get("starting_wealth", 0), 2
                ),
                "all_scores": {
                    aid: round(w, 2) for aid, w in
                    sorted(wealth_scores.items(), key=lambda x: x[1], reverse=True)
                },
            },
            "master_strategist": {
                "agent_id": strategist_id,
                "strategic_score": round(strategic_scores[strategist_id], 4),
                "breakdown": self._strategic_score_breakdown(
                    strategist_id,
                    self._state.agents[strategist_id]
                ),
                "all_scores": {
                    aid: round(s, 4) for aid, s in
                    sorted(strategic_scores.items(), key=lambda x: x[1], reverse=True)
                },
            },
        }

    def _compute_strategic_score(self, aid: str, agent: Dict) -> float:
        """Compute the composite strategic score for one agent.

        Formula:
            score = 0.35 * trade_efficiency
                  + 0.25 * negotiation_mastery
                  + 0.20 * cooperation_index
                  + 0.20 * event_adaptability

        Each component is normalised to [0, 1].
        """
        total_trades = max(agent.get("total_trades", 0), 1)
        profitable = agent.get("profitable_trades", 0)
        trade_efficiency = min(profitable / total_trades, 1.0)

        successful_deals = agent.get("successful_deals", 0)
        total_deals = successful_deals + agent.get("contracts_broken", 0)
        negotiation_mastery = (
            successful_deals / max(total_deals, 1)
            if total_deals > 0 else 0.5  # neutral if never negotiated
        )

        compound_produced = agent.get("compound_produced", 0)
        coalition_count = sum(
            1 for coal in self._state.coalitions.values()
            if aid in coal.get("members", [])
        )
        cooperation_index = min(
            (compound_produced * 0.3 + coalition_count * 0.2
             + successful_deals * 0.1),
            1.0,
        )

        event_actions = agent.get("event_responsive_actions", 0)
        max_rounds = max(self._state.round_number, 1)
        event_adaptability = min(event_actions / (max_rounds * 0.3), 1.0)

        return (
            0.35 * trade_efficiency
            + 0.25 * negotiation_mastery
            + 0.20 * cooperation_index
            + 0.20 * event_adaptability
        )

    def _strategic_score_breakdown(self, aid: str, agent: Dict) -> Dict[str, float]:
        """Return the four sub-components of the strategic score."""
        total_trades = max(agent.get("total_trades", 0), 1)
        profitable = agent.get("profitable_trades", 0)
        trade_efficiency = min(profitable / total_trades, 1.0)

        successful_deals = agent.get("successful_deals", 0)
        total_deals = successful_deals + agent.get("contracts_broken", 0)
        negotiation_mastery = (
            successful_deals / max(total_deals, 1)
            if total_deals > 0 else 0.5
        )

        compound_produced = agent.get("compound_produced", 0)
        coalition_count = sum(
            1 for coal in self._state.coalitions.values()
            if aid in coal.get("members", [])
        )
        cooperation_index = min(
            (compound_produced * 0.3 + coalition_count * 0.2
             + successful_deals * 0.1),
            1.0,
        )

        event_actions = agent.get("event_responsive_actions", 0)
        max_rounds = max(self._state.round_number, 1)
        event_adaptability = min(event_actions / (max_rounds * 0.3), 1.0)

        return {
            "trade_efficiency": round(trade_efficiency, 4),
            "negotiation_mastery": round(negotiation_mastery, 4),
            "cooperation_index": round(cooperation_index, 4),
            "event_adaptability": round(event_adaptability, 4),
        }

    # ==================================================================
    # LEADERBOARD / SCORING
    # ==================================================================
    def compute_leaderboard(self) -> List[Dict[str, Any]]:
        """Compute the final leaderboard ranking all agents.

        Each entry contains:
          - rank               (1 = best)
          - agent_id
          - role
          - total_wealth       (Award 1 metric)
          - strategic_score    (Award 2 metric)
          - cumulative_reward  (sum of all step rewards)
          - accuracy           (valid_actions / actions_taken)
          - trades, deals, compounds produced
          - awards_won         (list of award names won)

        Agents are sorted by total_wealth (primary) then
        strategic_score (tie-breaker).
        """
        entries = []
        awards = self.compute_awards()
        champion = awards["market_champion"]["agent_id"]
        strategist = awards["master_strategist"]["agent_id"]

        for aid, agent in self._state.agents.items():
            wealth = self._calculate_wealth(agent)
            strat = self._compute_strategic_score(aid, agent)
            actions_taken = max(agent.get("actions_taken", 0), 1)
            valid = agent.get("valid_actions", 0)
            accuracy = valid / actions_taken

            won = []
            if aid == champion:
                won.append("Market Champion")
            if aid == strategist:
                won.append("Master Strategist")

            entries.append({
                "agent_id": aid,
                "role": agent.get("role", "?"),
                "total_wealth": round(wealth, 2),
                "wealth_growth": round(
                    wealth - agent.get("starting_wealth", wealth), 2
                ),
                "strategic_score": round(strat, 4),
                "cumulative_reward": round(
                    agent.get("cumulative_reward", 0.0), 3
                ),
                "accuracy": round(accuracy, 3),
                "total_trades": agent.get("total_trades", 0),
                "successful_deals": agent.get("successful_deals", 0),
                "compound_produced": agent.get("compound_produced", 0),
                "reputation": round(agent.get("reputation", 1.0), 3),
                "awards_won": won,
            })

        # Sort by wealth (desc), then strategic_score (desc)
        entries.sort(key=lambda e: (e["total_wealth"], e["strategic_score"]),
                     reverse=True)
        for i, e in enumerate(entries):
            e["rank"] = i + 1

        return entries

    def compute_accuracy(self) -> Dict[str, float]:
        """Compute per-agent action accuracy (valid / total).

        This metric tracks whether the model is *repeatedly making
        things accurate* -- i.e. consistently producing valid,
        well-formed actions. Used for iterative improvement tracking.
        """
        result = {}
        for aid, agent in self._state.agents.items():
            actions_taken = max(agent.get("actions_taken", 0), 1)
            valid = agent.get("valid_actions", 0)
            result[aid] = round(valid / actions_taken, 4)
        return result

    # ------------------------------------------------------------------
    # BUY / SELL - Continuous Double Auction
    # ------------------------------------------------------------------
    def _handle_buy(self, action: MarketAction) -> float:
        aid, commodity, price, qty = (
            action.agent_id, action.commodity, action.price, action.quantity
        )
        if commodity not in COMMODITIES or qty <= 0 or price <= 0:
            return -0.1
        agent = self._state.agents[aid]
        if agent["cash"] < price * qty:
            return -0.2  # insufficient funds

        book = self._state.order_books[commodity]
        book["bids"].append({"agent": aid, "price": price, "qty": qty})
        match_reward = self._match_orders(commodity)
        return match_reward if match_reward > 0 else 0.02  # small reward for placing order

    def _handle_sell(self, action: MarketAction) -> float:
        aid, commodity, price, qty = (
            action.agent_id, action.commodity, action.price, action.quantity
        )
        if commodity not in COMMODITIES or qty <= 0 or price <= 0:
            return -0.1
        agent = self._state.agents[aid]
        if agent["inventory"].get(commodity, 0) < qty:
            return -0.2  # insufficient inventory

        book = self._state.order_books[commodity]
        book["asks"].append({"agent": aid, "price": price, "qty": qty})
        match_reward = self._match_orders(commodity)
        return match_reward if match_reward > 0 else 0.02

    def _match_orders(self, commodity: str) -> float:
        book = self._state.order_books[commodity]
        book["bids"].sort(key=lambda x: x["price"], reverse=True)
        book["asks"].sort(key=lambda x: x["price"])
        total_reward = 0.0

        while book["bids"] and book["asks"]:
            best_bid = book["bids"][0]
            best_ask = book["asks"][0]
            if best_bid["price"] < best_ask["price"]:
                break

            trade_price = (best_bid["price"] + best_ask["price"]) / 2.0
            trade_qty = min(best_bid["qty"], best_ask["qty"])
            buyer = self._state.agents[best_bid["agent"]]
            seller = self._state.agents[best_ask["agent"]]
            cost = trade_price * trade_qty

            if buyer["cash"] >= cost and seller["inventory"].get(commodity, 0) >= trade_qty:
                buyer["cash"] -= cost
                buyer["inventory"][commodity] = buyer["inventory"].get(commodity, 0) + trade_qty
                seller["cash"] += cost
                seller["inventory"][commodity] -= trade_qty

                # Track trade
                trade_record = {
                    "round": self._state.round_number,
                    "commodity": commodity,
                    "price": round(trade_price, 2),
                    "qty": trade_qty,
                    "buyer": best_bid["agent"],
                    "seller": best_ask["agent"],
                }
                self._state.trade_history.append(trade_record)
                self._state.market_metrics["total_trades"] += 1
                self._state.market_metrics["total_volume"] += cost

                buyer["total_trades"] = buyer.get("total_trades", 0) + 1
                seller["total_trades"] = seller.get("total_trades", 0) + 1

                # Update price history
                self._state.price_history.setdefault(commodity, []).append(trade_price)

                # Reward based on trade surplus
                buyer_surplus = (best_bid["price"] - trade_price) * trade_qty
                seller_surplus = (trade_price - best_ask["price"]) * trade_qty
                total_reward += 0.3 + 0.1 * min(buyer_surplus + seller_surplus, 5.0) / 5.0

                # Track profitable trades for Award 2 scoring
                if buyer_surplus > 0:
                    buyer["profitable_trades"] = buyer.get("profitable_trades", 0) + 1
                if seller_surplus > 0:
                    seller["profitable_trades"] = seller.get("profitable_trades", 0) + 1

            best_bid["qty"] -= trade_qty
            best_ask["qty"] -= trade_qty
            if best_bid["qty"] <= 0:
                book["bids"].pop(0)
            if best_ask["qty"] <= 0:
                book["asks"].pop(0)

        return total_reward

    # ------------------------------------------------------------------
    # PRODUCE - Compound goods (cooperation pillar)
    # ------------------------------------------------------------------
    def _handle_produce(self, action: MarketAction) -> float:
        aid = action.agent_id
        good = action.compound_good or action.commodity
        if good not in COMPOUND_GOODS:
            return -0.1

        agent = self._state.agents[aid]
        recipe = COMPOUND_GOODS[good]

        # Check if agent has all ingredients
        for ingredient, qty_needed in recipe.items():
            if agent["inventory"].get(ingredient, 0) < qty_needed:
                return -0.15  # missing ingredients

        # Consume ingredients and produce compound good
        for ingredient, qty_needed in recipe.items():
            agent["inventory"][ingredient] -= qty_needed

        # Add compound good to inventory
        agent["inventory"][good] = agent["inventory"].get(good, 0) + 1
        agent["compound_produced"] = agent.get("compound_produced", 0) + 1
        self._state.market_metrics["compound_goods_produced"] += 1

        # Reward: value of compound good minus cost of ingredients
        ingredient_cost = sum(
            self._get_market_price(ing) * qty
            for ing, qty in recipe.items()
        )
        compound_value = COMPOUND_PRICES.get(good, 40)
        profit_ratio = max(0, compound_value - ingredient_cost) / max(compound_value, 1)
        return 0.5 + 0.3 * profit_ratio

    # ------------------------------------------------------------------
    # NEGOTIATE - Natural language bilateral deals
    # ------------------------------------------------------------------
    def _handle_negotiate(self, action: MarketAction) -> float:
        target = action.target_agent
        aid = action.agent_id
        if target not in self._state.agents:
            return -0.1
        if target == aid:
            return -0.05

        # Create a pending deal
        deal_id = str(uuid.uuid4())[:8]
        deal = {
            "id": deal_id,
            "proposer": aid,
            "target": target,
            "commodity": action.commodity,
            "price": action.price,
            "quantity": action.quantity,
            "message": action.message,
            "round": self._state.round_number,
        }
        self._pending_deals[deal_id] = deal
        self._state.pending_deals[deal_id] = deal

        # Deliver message
        self._message_queues.setdefault(target, []).append({
            "from": aid,
            "type": "deal_proposal",
            "deal_id": deal_id,
            "message": action.message,
            "commodity": action.commodity,
            "price": action.price,
            "quantity": action.quantity,
        })

        self._state.market_metrics["deals_negotiated"] += 1
        return 0.05  # reward for engaging in negotiation

    def _handle_accept_deal(self, action: MarketAction) -> float:
        deal_id = action.coalition_id  # reuse field for deal_id
        if deal_id not in self._pending_deals:
            return -0.1

        deal = self._pending_deals[deal_id]
        aid = action.agent_id

        if deal["target"] != aid:
            return -0.1

        commodity = deal["commodity"]
        price = deal["price"]
        qty = deal["quantity"]
        proposer_id = deal["proposer"]

        if commodity not in COMMODITIES or qty <= 0 or price <= 0:
            return -0.1

        proposer = self._state.agents[proposer_id]
        acceptor = self._state.agents[aid]

        # Determine who is buying/selling based on proposer role
        # For simplicity: proposer wants to buy
        cost = price * qty
        if proposer["cash"] >= cost and acceptor["inventory"].get(commodity, 0) >= qty:
            proposer["cash"] -= cost
            proposer["inventory"][commodity] = proposer["inventory"].get(commodity, 0) + qty
            acceptor["cash"] += cost
            acceptor["inventory"][commodity] -= qty

            # Record trade
            self._state.trade_history.append({
                "round": self._state.round_number,
                "commodity": commodity, "price": price, "qty": qty,
                "buyer": proposer_id, "seller": aid,
                "type": "negotiated",
            })

            # Reputation boost for both
            proposer["reputation"] = min(2.0, proposer.get("reputation", 1.0) + 0.02)
            acceptor["reputation"] = min(2.0, acceptor.get("reputation", 1.0) + 0.02)
            proposer["successful_deals"] = proposer.get("successful_deals", 0) + 1
            acceptor["successful_deals"] = acceptor.get("successful_deals", 0) + 1

            del self._pending_deals[deal_id]
            if deal_id in self._state.pending_deals:
                del self._state.pending_deals[deal_id]

            return 0.4 + 0.1 * min(acceptor.get("reputation", 1.0), 2.0)
        return -0.2

    def _handle_reject_deal(self, action: MarketAction) -> float:
        deal_id = action.coalition_id
        if deal_id in self._pending_deals:
            del self._pending_deals[deal_id]
            if deal_id in self._state.pending_deals:
                del self._state.pending_deals[deal_id]
        return 0.0  # neutral - rejecting is fine

    # ------------------------------------------------------------------
    # COALITION - Dynamic alliance formation
    # ------------------------------------------------------------------
    def _handle_propose_coalition(self, action: MarketAction) -> float:
        cid = action.coalition_id or str(uuid.uuid4())[:8]
        aid = action.agent_id
        self._state.coalitions[cid] = {
            "id": cid,
            "proposer": aid,
            "members": [aid],
            "objective": action.message or "market alliance",
            "round_formed": self._state.round_number,
            "total_value": 0.0,
        }
        self._state.market_metrics["coalitions_formed"] += 1
        return 0.05

    def _handle_join_coalition(self, action: MarketAction) -> float:
        cid = action.coalition_id
        if cid not in self._state.coalitions:
            return -0.1
        coalition = self._state.coalitions[cid]
        aid = action.agent_id
        if aid not in coalition["members"]:
            coalition["members"].append(aid)

        # Coalition synergy reward - scales with size and diversity
        member_roles = set(
            self._state.agents[m].get("role", "")
            for m in coalition["members"]
            if m in self._state.agents
        )
        diversity_bonus = len(member_roles) * 0.05
        size_bonus = min(len(coalition["members"]) * 0.03, 0.15)
        return 0.1 + diversity_bonus + size_bonus

    def _handle_leave_coalition(self, action: MarketAction) -> float:
        cid = action.coalition_id
        if cid in self._state.coalitions:
            members = self._state.coalitions[cid]["members"]
            aid = action.agent_id
            if aid in members:
                members.remove(aid)
                # Reputation penalty for leaving
                self._state.agents[aid]["reputation"] = max(
                    0.0, self._state.agents[aid].get("reputation", 1.0) - 0.08
                )
                self._state.agents[aid]["contracts_broken"] = (
                    self._state.agents[aid].get("contracts_broken", 0) + 1
                )
            if not members:
                del self._state.coalitions[cid]
        return -0.08

    # ------------------------------------------------------------------
    # REGULATE - Regulatory actions
    # ------------------------------------------------------------------
    def _handle_regulate(self, action: MarketAction) -> float:
        aid = action.agent_id
        agent = self._state.agents.get(aid, {})
        if agent.get("role") != "regulator":
            # Non-regulators can still try but get penalized
            return -0.15

        target = action.target_agent
        if target not in self._state.agents:
            return -0.05

        # Check for price manipulation (large single-agent volume)
        target_agent = self._state.agents[target]
        recent_trades = [
            t for t in self._state.trade_history[-20:]
            if t.get("buyer") == target or t.get("seller") == target
        ]

        if len(recent_trades) > 5:  # suspicious high volume
            # Apply penalty to the target
            penalty = min(0.1 * len(recent_trades), 50)
            target_agent["cash"] -= penalty
            target_agent["reputation"] = max(
                0.0, target_agent.get("reputation", 1.0) - 0.1
            )

            # Reward regulator for catching manipulation
            return 0.3
        return 0.0

    # ------------------------------------------------------------------
    # Round management
    # ------------------------------------------------------------------
    def _advance_round(self):
        self._state.round_number += 1

        # Generate new event
        event_data = random.choice(EVENTS)
        self._state.current_event = event_data["text"]
        self._state.event_effects = event_data["effects"]

        # Producers regenerate stock (affected by events)
        for aid, agent in self._state.agents.items():
            if agent.get("role") == "producer":
                specialty = agent.get("specialty", "")
                if specialty in COMMODITIES:
                    base_regen = random.randint(3, 8)
                    event_mult = self._state.event_effects.get(specialty, 1.0)
                    regen = max(1, int(base_regen * event_mult))
                    agent["inventory"][specialty] = (
                        agent["inventory"].get(specialty, 0) + regen
                    )

        # Coalition value distribution
        for cid, coalition in list(self._state.coalitions.items()):
            members = coalition["members"]
            if len(members) >= 2:
                # Small bonus for being in a coalition
                per_member = 0.5 * len(members)
                for mid in members:
                    if mid in self._state.agents:
                        self._state.agents[mid]["cash"] += per_member

        # Update price history
        for commodity in COMMODITIES:
            prices = self._state.price_history.get(commodity, [])
            if prices:
                last = prices[-1]
            else:
                last = BASE_PRICES[commodity]
            # Price drift based on supply/demand
            total_supply = sum(
                a["inventory"].get(commodity, 0)
                for a in self._state.agents.values()
            )
            event_effect = self._state.event_effects.get(commodity, 1.0)
            noise = random.gauss(0, 0.5)
            supply_pressure = -0.1 * (total_supply - 40) / 40  # mean-reverting
            new_price = max(1.0, last * event_effect + supply_pressure + noise)
            self._state.price_history.setdefault(commodity, []).append(
                round(new_price, 2)
            )

        # Clean up expired deals (older than 3 rounds)
        expired = [
            did for did, deal in self._pending_deals.items()
            if self._state.round_number - deal.get("round", 0) > 3
        ]
        for did in expired:
            del self._pending_deals[did]
            if did in self._state.pending_deals:
                del self._state.pending_deals[did]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_market_price(self, commodity: str) -> float:
        prices = self._state.price_history.get(commodity, [])
        if prices:
            return prices[-1]
        return BASE_PRICES.get(commodity, 10)

    def _calculate_wealth(self, agent: Dict) -> float:
        cash = agent.get("cash", 0)
        inv_value = sum(
            agent["inventory"].get(c, 0) * self._get_market_price(c)
            for c in COMMODITIES
        )
        compound_value = sum(
            agent["inventory"].get(g, 0) * COMPOUND_PRICES.get(g, 0)
            for g in COMPOUND_GOODS
        )
        return cash + inv_value + compound_value

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _make_observation(self, agent_id: str) -> MarketObservation:
        agent = self._state.agents.get(agent_id, {})

        # Top-of-book (partial observability)
        top_of_book = {}
        for c in COMMODITIES:
            bids = self._state.order_books[c]["bids"]
            asks = self._state.order_books[c]["asks"]
            top_of_book[c] = {
                "best_bid": round(bids[0]["price"], 2) if bids else 0,
                "best_ask": round(asks[0]["price"], 2) if asks else 0,
                "bid_volume": sum(b["qty"] for b in bids[:3]),
                "ask_volume": sum(a["qty"] for a in asks[:3]),
            }

        # Recent trade prices
        recent_trades = {c: 0.0 for c in COMMODITIES}
        for t in reversed(self._state.trade_history[-20:]):
            if recent_trades.get(t["commodity"], -1) == 0.0:
                recent_trades[t["commodity"]] = t["price"]

        # Price history (last 10 rounds)
        price_hist = []
        for i in range(-min(10, self._state.round_number + 1), 0):
            round_prices = {}
            for c in COMMODITIES:
                hist = self._state.price_history.get(c, [])
                idx = len(hist) + i
                if 0 <= idx < len(hist):
                    round_prices[c] = hist[idx]
            if round_prices:
                price_hist.append(round_prices)

        # Agent's coalitions
        my_coalitions = [
            cid for cid, coal in self._state.coalitions.items()
            if agent_id in coal.get("members", [])
        ]
        coalition_details = {
            cid: {
                "members": self._state.coalitions[cid]["members"],
                "objective": self._state.coalitions[cid].get("objective", ""),
                "size": len(self._state.coalitions[cid]["members"]),
            }
            for cid in my_coalitions
            if cid in self._state.coalitions
        }

        # Pending deals for this agent
        my_deals = [
            {
                "deal_id": did,
                "from": deal["proposer"],
                "commodity": deal.get("commodity", ""),
                "price": deal.get("price", 0),
                "quantity": deal.get("quantity", 0),
                "message": deal.get("message", ""),
            }
            for did, deal in self._pending_deals.items()
            if deal.get("target") == agent_id
        ]

        # Messages
        msgs = self._message_queues.get(agent_id, [])
        self._message_queues[agent_id] = []

        # Market summary
        market_summary = {
            "total_trades_this_episode": self._state.market_metrics.get("total_trades", 0),
            "total_volume": round(self._state.market_metrics.get("total_volume", 0), 2),
            "active_coalitions": len(self._state.coalitions),
            "compound_goods_produced": self._state.market_metrics.get("compound_goods_produced", 0),
            "active_agents": len(self._state.agents),
        }

        # Legal actions
        role = agent.get("role", "trader")
        legal = ["buy", "sell", "negotiate", "propose_coalition", "pass"]
        if role == "consumer" or any(
            agent["inventory"].get(ing, 0) >= qty
            for g, recipe in COMPOUND_GOODS.items()
            for ing, qty in recipe.items()
        ):
            legal.append("produce")
        if my_deals:
            legal.extend(["accept_deal", "reject_deal"])
        if my_coalitions:
            legal.append("leave_coalition")
        if any(cid for cid in self._state.coalitions if agent_id not in self._state.coalitions[cid]["members"]):
            legal.append("join_coalition")
        if role == "regulator":
            legal.append("regulate")

        # Build text prompt for LLM agents
        prompt = self._build_prompt(agent_id, agent, top_of_book, recent_trades,
                                     msgs, my_deals, my_coalitions, market_summary)

        total_wealth = self._calculate_wealth(agent)

        return MarketObservation(
            agent_id=agent_id,
            role=agent.get("role", "trader"),
            cash=round(agent.get("cash", 0), 2),
            inventory=dict(agent.get("inventory", {})),
            reputation=round(agent.get("reputation", 1.0), 3),
            top_of_book=top_of_book,
            last_trade_prices=recent_trades,
            price_history=price_hist,
            messages=msgs,
            pending_deals=my_deals,
            coalitions=my_coalitions,
            coalition_details=coalition_details,
            event=self._state.current_event,
            event_effects=dict(self._state.event_effects),
            round_number=self._state.round_number,
            max_rounds=self._state.max_rounds,
            done=self._state.round_number >= self._state.max_rounds,
            reward=0.0,
            total_wealth=round(total_wealth, 2),
            market_summary=market_summary,
            prompt=prompt,
            legal_actions=list(set(legal)),
        )

    def _build_prompt(self, agent_id, agent, top_of_book, recent_trades,
                       messages, pending_deals, coalitions, market_summary) -> str:
        role = agent.get("role", "trader")
        inv = agent.get("inventory", {})
        cash = agent.get("cash", 0)

        lines = [
            f"You are agent '{agent_id}', a {role} in a multi-commodity MarketForge.",
            f"Round {self._state.round_number}/{self._state.max_rounds}.",
            f"Event: {self._state.current_event}",
            f"",
            f"Your status: Cash=${cash:.0f}, Reputation={agent.get('reputation', 1.0):.2f}",
            f"Inventory: {json.dumps({k: v for k, v in inv.items() if v > 0})}",
            f"",
            f"Market Prices (best bid / best ask):",
        ]
        for c in COMMODITIES:
            tb = top_of_book.get(c, {})
            last = recent_trades.get(c, 0)
            lines.append(
                f"  {c}: bid=${tb.get('best_bid', 0):.1f} / ask=${tb.get('best_ask', 0):.1f}"
                f"  (last trade: ${last:.1f})"
            )

        if messages:
            lines.append(f"\nIncoming messages ({len(messages)}):")
            for m in messages[:5]:
                lines.append(f"  From {m.get('from', '?')}: {m.get('message', '')}")

        if pending_deals:
            lines.append(f"\nPending deal offers ({len(pending_deals)}):")
            for d in pending_deals[:3]:
                lines.append(
                    f"  Deal {d['deal_id']}: {d['from']} offers "
                    f"{d['quantity']}x {d['commodity']} @ ${d['price']:.1f}"
                )

        if coalitions:
            lines.append(f"\nYour coalitions: {', '.join(coalitions)}")

        lines.append(f"\nMarket: {market_summary.get('total_trades_this_episode', 0)} trades, "
                      f"{market_summary.get('active_coalitions', 0)} coalitions active")

        # Scoring summary
        cum_reward = agent.get("cumulative_reward", 0.0)
        actions = max(agent.get("actions_taken", 0), 1)
        accuracy = agent.get("valid_actions", 0) / actions
        lines.append(f"\nYour Score: cumulative_reward={cum_reward:.2f}, "
                      f"accuracy={accuracy:.0%}, "
                      f"trades={agent.get('total_trades', 0)}, "
                      f"deals={agent.get('successful_deals', 0)}, "
                      f"compounds={agent.get('compound_produced', 0)}")

        # Awards race
        rounds_left = self._state.max_rounds - self._state.round_number
        if rounds_left <= 5:
            lines.append(f"\n*** {rounds_left} rounds left! ***")
            lines.append("Awards at stake:")
            lines.append("  Award 1 (Market Champion): highest total wealth wins")
            lines.append("  Award 2 (Master Strategist): best trade efficiency + negotiation + cooperation + event response")

        lines.append(f"\nRespond with a JSON action. Valid action_types: "
                      f"buy, sell, produce, negotiate, accept_deal, reject_deal, "
                      f"propose_coalition, join_coalition, leave_coalition, pass")

        return "\n".join(lines)
