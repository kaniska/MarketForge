#!/usr/bin/env python3
"""
train_market_forge.py - GRPO Training Script for MarketForge
================================================================
Trains an LLM agent to act as a market trader in the Multi-Agent
MarketForge using HuggingFace TRL (GRPOTrainer) + OpenEnv v0.2.1.

Follows the same pattern as the official OpenEnv Wordle GRPO notebook:
  https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb

Run in Google Colab (T4/A100 GPU):
    !pip install "openenv-core[core]>=0.2.1" trl transformers datasets accelerate
    !pip install git+https://huggingface.co/spaces/YOUR_USER/market_forge_env

Training Flow:
    1. Build prompt dataset from market scenarios
    2. For each prompt: generate action -> step environment -> collect reward
    3. Multi-level rewards guide the agent to learn:
       - Valid JSON formatting
       - Strategic trading (buy low / sell high)
       - Negotiation and coalition building (Theory of Mind)
       - Event-responsive adaptation
    4. GRPO optimizes the policy using group relative advantage
"""
import json
import random
import re
from dataclasses import asdict
from typing import Dict, List, Any, Optional

from datasets import Dataset
from transformers import AutoTokenizer


# ==========================================================================
# 1. Environment Connection
# ==========================================================================
# Option A: Connect to deployed HF Space (production)
# from client import MarketForgeEnv
# env = MarketForgeEnv(base_url="https://YOUR-USER-market-forge.hf.space")

# Option B: Local mock for demo/testing (works without Docker)
class MockMarketEnv:
    """Lightweight mock that mimics the OpenEnv MarketForge interface.

    This enables the training script to run standalone in Colab
    without needing the full server deployment. Replace with Option A
    for real training against the deployed environment.
    """

    COMMODITIES = ["wheat", "iron", "timber", "oil"]
    COMPOUND_GOODS = {"bread": {"wheat": 2, "oil": 1},
                      "tools": {"iron": 2, "timber": 1},
                      "furniture": {"timber": 2, "oil": 1}}
    EVENTS = [
        "Drought reduces wheat supply by 30%.",
        "Iron mine collapse -- iron production halved.",
        "Timber surplus -- prices expected to drop.",
        "Oil embargo -- oil supply restricted.",
        "Trade festival -- transaction costs waived.",
        "Consumer demand for furniture surges 50%.",
        "New trade route opens -- all commodities +10%.",
        "Calm markets -- no special event.",
        "Speculator rumor: oil prices to skyrocket.",
        "Government subsidy for tool production.",
    ]

    class Obs:
        def __init__(self):
            self.agent_id = "trader_1"
            self.role = random.choice(["producer", "consumer", "trader", "speculator"])
            self.cash = round(random.uniform(500, 1500), 2)
            self.inventory = {c: random.randint(0, 20)
                            for c in MockMarketEnv.COMMODITIES}
            self.reputation = round(random.uniform(0.7, 1.3), 2)
            self.top_of_book = {
                c: {"best_bid": round(random.uniform(5, 20), 1),
                     "best_ask": round(random.uniform(8, 25), 1)}
                for c in MockMarketEnv.COMMODITIES
            }
            self.last_trade_prices = {
                c: round(random.uniform(6, 22), 1)
                for c in MockMarketEnv.COMMODITIES
            }
            self.event = random.choice(MockMarketEnv.EVENTS)
            self.round_number = 0
            self.max_rounds = 50
            self.reward = 0.0
            self.done = False
            self.messages = []
            self.coalitions = []
            self.pending_deals = []
            self.prompt = ""
            self.legal_actions = ["buy", "sell", "negotiate", "propose_coalition", "pass"]

        def build_prompt(self):
            inv_str = json.dumps({k: v for k, v in self.inventory.items() if v > 0})
            prices_str = ", ".join(
                f"{c}: bid=${tb['best_bid']}/ask=${tb['best_ask']}"
                for c, tb in self.top_of_book.items()
            )
            self.prompt = (
                f"You are agent '{self.agent_id}', a {self.role} in a MarketForge.\n"
                f"Round {self.round_number}/{self.max_rounds}. Event: {self.event}\n"
                f"Cash: ${self.cash:.0f}, Reputation: {self.reputation:.2f}\n"
                f"Inventory: {inv_str}\n"
                f"Market: {prices_str}\n"
                f"Decide your action as JSON."
            )
            return self.prompt

    class Result:
        def __init__(self, obs, reward):
            self.observation = obs
            self.reward = reward
            self.done = obs.done

    def reset(self, **kwargs):
        self._round = 0
        obs = self.Obs()
        obs.build_prompt()
        return self.Result(obs, 0.0)

    def step(self, action_input):
        self._round += 1
        obs = self.Obs()
        obs.round_number = self._round
        reward = 0.0

        try:
            if isinstance(action_input, str):
                # Try extracting JSON from text
                match = re.search(r'\{[^}]+\}', action_input, re.DOTALL)
                parsed = json.loads(match.group()) if match else json.loads(action_input)
            elif isinstance(action_input, dict):
                parsed = action_input
            else:
                parsed = asdict(action_input) if hasattr(action_input, '__dataclass_fields__') else {}

            atype = parsed.get("action_type", "pass")
            commodity = parsed.get("commodity", "")
            price = float(parsed.get("price", 0))
            qty = int(parsed.get("quantity", 0))
            message = parsed.get("message", "")

            if atype in ("buy", "sell") and commodity in self.COMMODITIES:
                if 1 <= price <= 50 and 1 <= qty <= 20:
                    # Good trade parameters
                    market_price = obs.last_trade_prices.get(commodity, 10)
                    if atype == "buy" and price <= market_price * 1.1:
                        reward = 0.4 + random.uniform(0, 0.2)  # good buy price
                    elif atype == "sell" and price >= market_price * 0.9:
                        reward = 0.4 + random.uniform(0, 0.2)  # good sell price
                    else:
                        reward = 0.2 + random.uniform(-0.1, 0.1)
                else:
                    reward = -0.15  # bad parameters

            elif atype == "produce" and parsed.get("compound_good", "") in self.COMPOUND_GOODS:
                reward = 0.5 + random.uniform(0, 0.2)

            elif atype == "negotiate":
                reward = 0.1
                if message and len(message) > 15:
                    reward += 0.15  # reward for substantive messages
                if parsed.get("target_agent", ""):
                    reward += 0.05

            elif atype == "propose_coalition":
                reward = 0.15
                if message and len(message) > 10:
                    reward += 0.1

            elif atype == "join_coalition":
                reward = 0.2

            elif atype == "accept_deal":
                reward = 0.3

            elif atype == "pass":
                reward = -0.05

            else:
                reward = -0.1

        except Exception:
            reward = -0.3  # malformed action

        obs.reward = reward
        obs.done = self._round >= 6  # 6 turns per episode for training
        obs.build_prompt()
        return self.Result(obs, reward)


env = MockMarketEnv()


# ==========================================================================
# 2. System Prompt & Dataset
# ==========================================================================
SYSTEM_PROMPT = """You are an autonomous trading agent in a multi-commodity MarketForge.
You trade wheat, iron, timber, and oil. You can also produce compound goods (bread, tools, furniture).

RESPOND WITH EXACTLY ONE JSON OBJECT choosing your action. Valid action_types:
  buy, sell, produce, negotiate, propose_coalition, join_coalition, accept_deal, pass

EXAMPLES:
  {"action_type":"buy","commodity":"wheat","price":10,"quantity":5}
  {"action_type":"sell","commodity":"iron","price":18,"quantity":3}
  {"action_type":"produce","compound_good":"bread"}
  {"action_type":"negotiate","target_agent":"producer_wheat","commodity":"wheat","price":9,"quantity":10,"message":"I need wheat for bread. Can we do $9/unit for 10?"}
  {"action_type":"propose_coalition","message":"Form buying group to get bulk discount on iron"}
  {"action_type":"pass"}

STRATEGY GUIDELINES:
- Buy commodities when prices are low, sell when high
- React to market events (droughts, embargoes raise prices)
- Negotiate deals for better prices than the open market
- Form coalitions for collective bargaining power
- Produce compound goods when you have ingredients (bread=2wheat+1oil)
- Maintain good reputation by honoring deals

Your goal: maximize profit through smart trading, negotiation, and coalition-building."""


def make_prompts(n: int = 64) -> List[Dict[str, str]]:
    """Generate diverse market scenarios as training prompts."""
    rows = []
    for _ in range(n):
        result = env.reset()
        obs = result.observation
        prompt_text = obs.prompt
        rows.append({"prompt": prompt_text})
    return rows


dataset = Dataset.from_list(make_prompts(256))


# ==========================================================================
# 3. Rollout Function (following Wordle notebook pattern)
# ==========================================================================
def rollout_func(prompts: list, trainer) -> dict:
    """Rollout function for GRPO training with environment interaction.

    This follows the exact pattern from the OpenEnv Wordle GRPO notebook:
    - Generate completions using trainer
    - Step the environment with each completion
    - Collect multi-level rewards
    - Return structured rollout data with env_mask

    Args:
        prompts: List of prompt texts
        trainer: GRPOTrainer instance

    Returns:
        Dictionary with prompt_ids, completion_ids, logprobs, env_mask, and rewards
    """
    from trl.experimental.openenv import generate_rollout_completions

    episode_prompt_ids = []
    episode_completion_ids = []
    episode_logprobs = []
    episode_env_masks = []
    env_rewards = []
    format_rewards = []
    strategic_rewards = []

    tokenizer = trainer.processing_class

    for prompt_text in prompts:
        episode = rollout_once(
            trainer=trainer,
            env=env,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            system_prompt=SYSTEM_PROMPT,
            max_turns=6,
            max_new_tokens=128,
        )
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        episode_env_masks.append(episode["env_mask"])
        env_rewards.append(episode["env_reward"])
        format_rewards.append(episode["format_reward"])
        strategic_rewards.append(episode["strategic_reward"])

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "env_mask": episode_env_masks,
        "env_reward": env_rewards,
        "format_reward": format_rewards,
        "strategic_reward": strategic_rewards,
    }


def rollout_once(trainer, env, tokenizer, prompt_text, system_prompt,
                  max_turns, max_new_tokens) -> dict:
    """Run one full episode of market interaction.

    Following the Wordle notebook pattern:
    1. Reset environment
    2. Build prompt from system + observation
    3. Generate completion (agent's action)
    4. Parse action, step environment
    5. Collect rewards and env_mask
    6. Repeat for max_turns
    """
    from trl.experimental.openenv import generate_rollout_completions

    result = env.reset()
    observation = result.observation

    prompt_ids = []
    completion_ids = []
    logprobs = []
    env_mask = []
    model_outputs = []
    raw_rewards = []

    accumulated_messages = [{"role": "system", "content": system_prompt}]

    # Build initial prompt
    initial_user_prompt = observation.prompt or prompt_text
    initial_messages = accumulated_messages + [
        {"role": "user", "content": initial_user_prompt}
    ]
    initial_prompt_text = tokenizer.apply_chat_template(
        initial_messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    initial_prompt_ids = tokenizer.encode(initial_prompt_text, add_special_tokens=False)
    prompt_ids.extend(initial_prompt_ids)

    for _turn in range(max_turns):
        if result.done:
            break

        user_prompt = observation.prompt or prompt_text
        messages = accumulated_messages + [
            {"role": "user", "content": user_prompt}
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        # Generate completion
        rollout_outputs = generate_rollout_completions(
            trainer, [full_prompt],
            generation_overrides={"max_tokens": max_new_tokens}
        )[0]

        # Add newline separators
        newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
        completion_ids.extend(newline_tokens)
        logprobs.extend([0.0] * len(newline_tokens))
        env_mask.extend([1] * len(newline_tokens))

        # Add model completion
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])
        env_mask.extend([1] * len(rollout_outputs["completion_ids"]))

        completion_ids.extend(newline_tokens)
        logprobs.extend([0.0] * len(newline_tokens))
        env_mask.extend([1] * len(newline_tokens))

        # Decode completion
        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )
        model_outputs.append(completion_text.strip())

        # Step environment
        result = env.step(completion_text)
        raw_rewards.append(float(result.reward or 0.0))
        observation = result.observation

        # Add environment feedback as context (masked out from loss)
        env_feedback = f"\nReward: {result.reward:.2f}. {observation.prompt}"
        env_tokens = tokenizer.encode(env_feedback, add_special_tokens=False)
        completion_ids.extend(env_tokens)
        logprobs.extend([0.0] * len(env_tokens))
        env_mask.extend([0] * len(env_tokens))  # environment tokens masked

        # Update conversation history
        accumulated_messages.append({"role": "user", "content": user_prompt})
        accumulated_messages.append({
            "role": "assistant",
            "content": completion_text + "\n" + env_feedback
        })

    # Compute episode-level rewards
    env_reward = sum(raw_rewards) / max(len(raw_rewards), 1)
    format_reward = compute_format_reward(model_outputs)
    strategic_reward = compute_strategic_reward(model_outputs)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask,
        "env_reward": env_reward,
        "format_reward": format_reward,
        "strategic_reward": strategic_reward,
        "model_outputs": model_outputs,
    }


# ==========================================================================
# 4. Reward Functions
# ==========================================================================
def compute_format_reward(model_outputs: List[str]) -> float:
    """Reward for valid JSON action format across all turns."""
    if not model_outputs:
        return 0.0
    correct = 0
    for text in model_outputs:
        try:
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                d = json.loads(match.group())
                valid_types = {"buy", "sell", "produce", "negotiate",
                             "propose_coalition", "join_coalition",
                             "accept_deal", "reject_deal", "pass"}
                if d.get("action_type") in valid_types:
                    correct += 1
        except Exception:
            pass
    return correct / len(model_outputs)


def compute_strategic_reward(model_outputs: List[str]) -> float:
    """Reward for strategic diversity and depth."""
    if not model_outputs:
        return 0.0
    action_types = set()
    tom_score = 0.0
    for text in model_outputs:
        try:
            match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if match:
                d = json.loads(match.group())
                atype = d.get("action_type", "")
                action_types.add(atype)
                # Theory of Mind bonus
                if atype in ("negotiate", "propose_coalition") and d.get("message", ""):
                    msg = d["message"]
                    if len(msg) > 20:
                        tom_score += 0.15
        except Exception:
            pass

    diversity = min(len(action_types) / 4.0, 1.0) * 0.3
    return diversity + tom_score / max(len(model_outputs), 1)


def reward_env(completions, **kwargs):
    """Primary environment reward."""
    rewards = kwargs.get("env_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_format(completions, **kwargs):
    """Format reward (pre-computed)."""
    rewards = kwargs.get("format_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_strategic(completions, **kwargs):
    """Strategic depth reward (pre-computed)."""
    rewards = kwargs.get("strategic_reward", [])
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


# ==========================================================================
# 5. Configure and Launch GRPO Training
# ==========================================================================
if __name__ == "__main__":
    from trl import GRPOConfig, GRPOTrainer

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for demo; use 1.7B+ for better results
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    output_dir = "market-forge-agent"

    grpo_config = GRPOConfig(
        # Training schedule
        num_train_epochs=1,
        learning_rate=1e-6,
        gradient_accumulation_steps=16,
        per_device_train_batch_size=1,
        warmup_steps=10,
        max_grad_norm=1.0,

        # GRPO configuration
        num_generations=2,
        max_completion_length=1024,
        log_completions=False,

        # vLLM (for GPU environments)
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.15,
        vllm_max_model_length=3072,

        # Output
        output_dir=output_dir,
        report_to="none",
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,

        # Memory optimization
        gradient_checkpointing=True,

        # Hub
        push_to_hub=False,
    )

    trainer = GRPOTrainer(
        model=model_id,
        processing_class=tokenizer,
        reward_funcs=[reward_env, reward_format, reward_strategic],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    print("Starting MarketForge agent training with GRPO...")
    print(f"Model: {model_id}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Output: {output_dir}")
    trainer.train()
    print(f"Training complete! Model saved to ./{output_dir}")
