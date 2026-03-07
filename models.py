"""
MarketForge - OpenEnv Data Models
==================================
Type-safe contracts for the MarketForge multi-commodity market environment.
Compatible with OpenEnv v0.2.1 stable release.

These models form the "API contract" between the Environment (server)
and the EnvClient (agent).  Three dataclasses define everything:

    MarketAction      -- what the agent sends TO the environment
    MarketObservation -- what the agent receives FROM the environment
    MarketState       -- the full server-side game state (debug only)

The relationship is simple:

    Agent  --[MarketAction]-->  Environment
    Agent  <--[MarketObservation]--  Environment
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# Base classes - compatible with both openenv_core and openenv.core
# We define lightweight bases so the package works standalone too.
# ---------------------------------------------------------------------------
try:
    from openenv_core.env_server import Action, Observation, State
except ImportError:
    try:
        from openenv.core.env_server import Action, Observation, State
    except ImportError:
        # Fallback: define minimal base classes for standalone usage
        @dataclass
        class Action:
            """Base class for all actions.  Every action can carry metadata."""
            metadata: Dict[str, Any] = field(default_factory=dict)

        @dataclass
        class Observation:
            """Base class for all observations.

            Every observation must include:
              done   -- is the episode (game) finished?
              reward -- how good was the last action?
            """
            done: bool = False
            reward: float = 0.0
            metadata: Dict[str, Any] = field(default_factory=dict)

        @dataclass
        class State:
            """Base class for the full environment state."""
            episode_id: str = ""
            step_count: int = 0


# ---------------------------------------------------------------------------
# Market Action  (Agent -> Environment)
# ---------------------------------------------------------------------------
@dataclass
class MarketAction(Action):
    """An agent's action in the market environment.

    This is the "controller" the agent uses to interact with the market.
    Think of it like a game controller with different buttons:

    Trading Controls (Competition):
      - "buy"   : place a buy order   (needs: commodity, price, quantity)
      - "sell"  : place a sell order  (needs: commodity, price, quantity)

    Production Controls (Cooperation):
      - "produce": craft a compound good (needs: compound_good name)

    Negotiation Controls:
      - "negotiate"    : send a deal proposal  (needs: target_agent, commodity, price, quantity, message)
      - "accept_deal"  : accept an offer       (needs: coalition_id as deal_id)
      - "reject_deal"  : reject an offer       (needs: coalition_id as deal_id)

    Coalition Controls:
      - "propose_coalition" : create an alliance (needs: message)
      - "join_coalition"    : join an alliance   (needs: coalition_id)
      - "leave_coalition"   : leave an alliance  (needs: coalition_id)

    Other:
      - "regulate" : regulatory action (regulator role only)
      - "pass"     : do nothing this turn

    Example:
        action = MarketAction(
            agent_id="trader_1",
            action_type="buy",
            commodity="wheat",
            price=12.0,
            quantity=5,
        )
    """
    agent_id: str = ""
    action_type: str = "pass"
    commodity: str = ""               # e.g. "wheat", "iron", "timber", "oil"
    price: float = 0.0
    quantity: int = 0
    target_agent: str = ""            # for negotiate / coalition actions
    message: str = ""                 # free-text negotiation utterance
    coalition_id: str = ""            # for coalition actions / deal_id
    compound_good: str = ""           # for produce action (e.g. "bread")


# ---------------------------------------------------------------------------
# Market Observation  (Environment -> Agent)
# ---------------------------------------------------------------------------
@dataclass
class MarketObservation(Observation):
    """What the agent sees after taking an action.

    This is the agent's "dashboard" -- it shows everything the agent
    is allowed to know (partial observability). Key fields:

    Identity:
        agent_id       -- who am I?
        role           -- producer / consumer / trader / speculator

    Resources:
        cash           -- how much money do I have?
        inventory      -- what commodities do I hold? {name: qty}
        reputation     -- how trustworthy am I? (0.0 to 2.0)

    Market Data (what I can see):
        top_of_book        -- best bid/ask for each commodity
        last_trade_prices  -- most recent trade price per commodity
        price_history      -- last N rounds of prices

    Social:
        messages       -- incoming messages from other agents
        pending_deals  -- deal offers waiting for my response
        coalitions     -- IDs of coalitions I belong to

    Game State:
        event          -- current market event (e.g. "Drought...")
        round_number   -- current round (0 to max_rounds)
        done           -- is the game over?
        reward         -- reward from my last action

    Scoring:
        total_wealth   -- my total wealth (cash + inventory value)
        market_summary -- overall market statistics + awards at game end

    Agent Interface:
        prompt         -- text description for LLM agents
        legal_actions  -- list of valid action_types right now

    When done=True, market_summary contains:
        "awards"      -- who won Award 1 (Champion) and Award 2 (Strategist)
        "leaderboard" -- ranked list of all agents with scores
        "game_over"   -- True
    """
    agent_id: str = ""
    role: str = ""
    cash: float = 0.0
    inventory: Dict[str, int] = field(default_factory=dict)
    reputation: float = 1.0
    top_of_book: Dict[str, Dict[str, float]] = field(default_factory=dict)
    last_trade_prices: Dict[str, float] = field(default_factory=dict)
    price_history: List[Dict[str, float]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    pending_deals: List[Dict[str, Any]] = field(default_factory=list)
    coalitions: List[str] = field(default_factory=list)
    coalition_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    event: str = ""
    event_effects: Dict[str, float] = field(default_factory=dict)
    round_number: int = 0
    max_rounds: int = 50
    done: bool = False
    reward: float = 0.0
    total_wealth: float = 0.0
    market_summary: Dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    legal_actions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Market State  (full server-side state -- NOT shown to agents)
# ---------------------------------------------------------------------------
@dataclass
class MarketState(State):
    """Full environment state (server-side only).

    This is the "god view" of the game -- it sees everything.
    Used for:
      - Visualisation (the Gradio dashboard reads this)
      - Evaluation (compute awards, leaderboard, accuracy)
      - Debugging (check that the engine is correct)

    Agents should NEVER see this directly -- that would be like
    seeing all the cards in poker.
    """
    round_number: int = 0
    max_rounds: int = 50
    agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    order_books: Dict[str, Dict[str, list]] = field(default_factory=dict)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    price_history: Dict[str, List[float]] = field(default_factory=dict)
    coalitions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pending_deals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    current_event: str = ""
    event_effects: Dict[str, float] = field(default_factory=dict)
    episode_id: str = ""
    step_count: int = 0
    round_rewards: Dict[str, List[float]] = field(default_factory=dict)
    market_metrics: Dict[str, Any] = field(default_factory=dict)
