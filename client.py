"""
Multi-Agent MarketForge - OpenEnv Client (EnvClient)
======================================================
HTTP client for the MarketForge environment.
Compatible with OpenEnv v0.2.1.

This client is the "steering wheel" for your LLM agent.
Just like a driving game gives you clear controls (gas, brake,
steering wheel), this client gives you three clear methods:

    env.reset()            -- Start a new game
    env.step(action)       -- Take one action and see what happens
    env.get_state()        -- Peek at the full game state (debug only)

Quick Start
-----------
    from client import MarketForgeEnv
    from models import MarketAction

    # Connect to the environment (locally or on HuggingFace Spaces)
    env = MarketForgeEnv(base_url="http://localhost:8000")

    # Start a new game
    result = env.reset(max_rounds=30)
    print(result.observation.prompt)   # See what the agent sees
    print(result.observation.legal_actions)  # See what the agent can do

    # Take an action (like pressing a button in a game)
    result = env.step(MarketAction(
        agent_id="trader_1",
        action_type="buy",
        commodity="wheat",
        price=12.0,
        quantity=5,
    ))
    print(f"Reward: {result.reward}")  # Did it go well?
    print(f"Done: {result.done}")      # Is the game over?

    # When done=True, check who won
    if result.done:
        awards = result.observation.market_summary.get("awards", {})
        print(f"Market Champion: {awards['market_champion']['agent_id']}")
        print(f"Master Strategist: {awards['master_strategist']['agent_id']}")

Available Actions (your "controls")
------------------------------------
    buy            -- Buy a commodity on the open market
    sell           -- Sell a commodity on the open market
    produce        -- Craft a compound good (bread/tools/furniture)
    negotiate      -- Send a deal proposal to another agent
    accept_deal    -- Accept a pending deal offer
    reject_deal    -- Reject a pending deal offer
    propose_coalition -- Create a new alliance
    join_coalition -- Join an existing alliance
    leave_coalition -- Leave an alliance (reputation penalty)
    pass           -- Do nothing this turn
"""
from dataclasses import asdict
from typing import Any

from models import MarketAction, MarketObservation

# Try importing OpenEnv client base
try:
    from openenv_core.env_client import EnvClient, StepResult
except ImportError:
    try:
        from openenv.core.env_client import EnvClient, StepResult
    except ImportError:
        # Fallback: standalone HTTP client
        # This lets the code run without installing openenv_core
        import requests
        from dataclasses import dataclass

        @dataclass
        class StepResult:
            """Result from a single step in the environment.

            Attributes:
                observation: What the agent sees after acting
                reward:      How good was this action? (float)
                done:        Is the game over? (bool)
            """
            observation: Any = None
            reward: float = 0.0
            done: bool = False

        class EnvClient:
            """Base HTTP client for OpenEnv environments.

            Implements the three core methods that every EnvClient must have:
                reset()      -- start a new game
                step(action) -- take an action, get observation back
                get_state()  -- retrieve full server state (debug only)

            All communication happens over HTTP (JSON payloads).
            """

            def __init__(self, base_url: str = "http://localhost:8000", **kwargs):
                self.base_url = base_url.rstrip("/")

            def sync(self):
                return self

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

            def close(self):
                pass

            # ---- The Three Core Methods ----

            def reset(self, **kwargs) -> StepResult:
                """Reset the environment and start a new game.

                Keyword args are forwarded to the server (e.g. max_rounds=30).
                Returns a StepResult with the initial observation.
                """
                resp = requests.post(
                    f"{self.base_url}/reset", json=kwargs, timeout=30
                )
                resp.raise_for_status()
                return self._payload_to_step_result(resp.json())

            def step(self, action) -> StepResult:
                """Execute one action and get the result.

                Args:
                    action: A MarketAction specifying what the agent does.

                Returns:
                    StepResult with observation, reward, and done flag.
                    When done=True, the game is over and awards are computed.
                """
                payload = self._step_to_payload(action)
                resp = requests.post(
                    f"{self.base_url}/step", json=payload, timeout=30
                )
                resp.raise_for_status()
                return self._payload_to_step_result(resp.json())

            def get_state(self) -> Any:
                """Get the full server-side game state.

                This is for debugging and visualization ONLY.
                Agents should never use this to make decisions --
                that would be cheating (like seeing all cards in poker).
                """
                resp = requests.get(f"{self.base_url}/state", timeout=30)
                resp.raise_for_status()
                return self._payload_to_state(resp.json())

            # ---- Serialisation hooks (override in subclass) ----

            def _step_to_payload(self, action) -> dict:
                """Convert an action object to a JSON-serialisable dict."""
                raise NotImplementedError

            def _payload_to_step_result(self, payload: dict) -> StepResult:
                """Convert a JSON response back into a StepResult."""
                raise NotImplementedError

            def _payload_to_state(self, payload: dict) -> Any:
                """Convert a JSON state response into a state object."""
                raise NotImplementedError


class MarketForgeEnv(EnvClient):
    """Client for the Multi-Agent MarketForge environment.

    This is the concrete EnvClient that knows how to serialise
    MarketAction objects and deserialise MarketObservation responses.

    Usage:
        # Connect to a deployed HuggingFace Space
        env = MarketForgeEnv(base_url="https://YOUR-SPACE.hf.space")

        # Or connect locally
        env = MarketForgeEnv(base_url="http://localhost:8000")

        # Start a new game (30 rounds)
        result = env.reset(max_rounds=30)
        print(result.observation.prompt)

        # Game loop -- keep taking actions until done
        while not result.done:
            action = MarketAction(
                agent_id="trader_1",
                action_type="buy",
                commodity="wheat",
                price=12.0,
                quantity=5,
            )
            result = env.step(action)
            print(f"Reward: {result.reward}, Cash: {result.observation.cash}")

        # Game over -- check awards
        awards = result.observation.market_summary.get("awards", {})
        print(f"Champion: {awards['market_champion']['agent_id']}")
    """

    def _step_to_payload(self, action: MarketAction) -> dict:
        """Convert MarketAction -> JSON dict for the server."""
        return asdict(action)

    def _payload_to_step_result(self, payload: dict) -> StepResult:
        """Convert server JSON response -> StepResult with MarketObservation."""
        obs_data = payload.get("observation", payload)
        # Filter to only valid fields (ignore any extra server fields)
        valid_fields = set(MarketObservation.__dataclass_fields__.keys())
        filtered = {k: v for k, v in obs_data.items() if k in valid_fields}
        obs = MarketObservation(**filtered)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _payload_to_state(self, payload: dict) -> Any:
        """Convert server JSON state -> dict."""
        return payload
