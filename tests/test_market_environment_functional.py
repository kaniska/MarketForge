import math

from server.market_environment import (
    MarketEnvironment,
    BASE_PRICES,
    COMMODITIES,
)
from models import MarketAction, MarketObservation


def make_env(max_rounds: int = 5, seed: int = 123) -> tuple[MarketEnvironment, MarketObservation]:
    """Create a fresh environment and return (env, initial_observation)."""
    env = MarketEnvironment()
    obs = env.reset(seed=seed, max_rounds=max_rounds)
    assert isinstance(obs, MarketObservation)
    return env, obs


def test_reset_returns_initial_observation():
    env = MarketEnvironment()
    obs = env.reset(seed=123, max_rounds=3)

    assert isinstance(obs, MarketObservation)
    assert obs.round_number == 0
    assert obs.max_rounds == 3
    assert not obs.done
    assert obs.event  # non-empty event string
    assert obs.legal_actions, "Expected at least one legal action after reset"


def test_pass_actions_eventually_finish_episode():
    env, obs = make_env(max_rounds=2, seed=42)

    last_obs = obs
    max_steps = 100
    steps = 0

    # Repeatedly take "pass" actions until the episode ends.
    while not last_obs.done and steps < max_steps:
        action = MarketAction(agent_id="trader_1", action_type="pass")
        last_obs = env.step(action)
        steps += 1

    assert last_obs.done, "Episode did not finish within expected number of steps"
    assert env.state.round_number == env.state.max_rounds == 2


def test_buy_and_sell_create_trade_and_update_state():
    env, _ = make_env(max_rounds=1, seed=0)
    state = env.state

    assert "consumer_1" in state.agents
    assert "producer_wheat" in state.agents

    buyer = state.agents["consumer_1"]
    seller = state.agents["producer_wheat"]

    # Sanity checks on initial resources
    assert buyer["cash"] > 0
    assert seller["inventory"].get("wheat", 0) > 0

    buyer_cash_before = buyer["cash"]
    seller_cash_before = seller["cash"]
    seller_wheat_before = seller["inventory"]["wheat"]

    # Place a buy order and then a sell order that should cross the book.
    buy_action = MarketAction(
        agent_id="consumer_1",
        action_type="buy",
        commodity="wheat",
        price=BASE_PRICES["wheat"] + 2.0,
        quantity=1,
    )
    sell_action = MarketAction(
        agent_id="producer_wheat",
        action_type="sell",
        commodity="wheat",
        price=BASE_PRICES["wheat"] - 2.0,
        quantity=1,
    )

    env.step(buy_action)
    env.step(sell_action)

    # At least one trade should have been recorded.
    assert env.state.trade_history, "Expected at least one trade in trade_history"
    trade = env.state.trade_history[-1]

    assert trade["commodity"] == "wheat"
    assert {trade["buyer"], trade["seller"]} == {"consumer_1", "producer_wheat"}

    buyer_after = env.state.agents["consumer_1"]
    seller_after = env.state.agents["producer_wheat"]

    assert buyer_after["cash"] < buyer_cash_before
    assert seller_after["cash"] > seller_cash_before
    assert seller_after["inventory"]["wheat"] == seller_wheat_before - 1


def test_produce_compound_good_updates_inventory_and_reward():
    env, _ = make_env(max_rounds=1, seed=0)
    state = env.state

    agent_id = "consumer_1"
    agent = state.agents[agent_id]

    # Give the agent exactly enough ingredients to produce one bread.
    agent["inventory"]["wheat"] = 2
    agent["inventory"]["oil"] = 1
    starting_bread = agent["inventory"].get("bread", 0)

    action = MarketAction(
        agent_id=agent_id,
        action_type="produce",
        compound_good="bread",
    )
    obs = env.step(action)

    agent_after = env.state.agents[agent_id]

    assert agent_after["inventory"].get("bread", 0) == starting_bread + 1
    assert agent_after["inventory"].get("wheat", 0) == 0
    assert agent_after["inventory"].get("oil", 0) == 0
    assert obs.reward > 0.0


def test_negotiate_and_accept_deal_flow():
    env, _ = make_env(max_rounds=3, seed=1)
    state = env.state

    proposer_id = "trader_1"
    target_id = "producer_wheat"

    # Ensure target has inventory so a negotiated trade can succeed.
    state.agents[target_id]["inventory"]["wheat"] = max(
        1, state.agents[target_id]["inventory"].get("wheat", 0)
    )

    # Proposer sends a deal proposal.
    negotiate_action = MarketAction(
        agent_id=proposer_id,
        action_type="negotiate",
        target_agent=target_id,
        commodity="wheat",
        price=BASE_PRICES["wheat"],
        quantity=1,
        message="I would like to buy 1 wheat at a fair price.",
    )
    env.step(negotiate_action)

    assert env.state.pending_deals, "Expected at least one pending deal after negotiation"
    deal_id = next(iter(env.state.pending_deals.keys()))

    proposer_rep_before = state.agents[proposer_id].get("reputation", 1.0)
    target_rep_before = state.agents[target_id].get("reputation", 1.0)

    # Target accepts the deal.
    accept_action = MarketAction(
        agent_id=target_id,
        action_type="accept_deal",
        coalition_id=deal_id,
    )
    env.step(accept_action)

    # Deal should be removed from pending_deals.
    assert deal_id not in env.state.pending_deals

    # A negotiated trade should have been added.
    negotiated_trades = [
        t for t in env.state.trade_history if t.get("type") == "negotiated"
    ]
    assert negotiated_trades, "Expected at least one negotiated trade"

    proposer_after = env.state.agents[proposer_id]
    target_after = env.state.agents[target_id]

    # Both parties should get a small reputation boost.
    assert proposer_after["reputation"] >= proposer_rep_before
    assert target_after["reputation"] >= target_rep_before

