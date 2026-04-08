import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from warehouse_env import WarehouseOrderExceptionEnv
from models import WarehouseAction

SYSTEM_PROMPT = """
You are a deterministic warehouse order exception resolution agent.

Your goal is to maximize task reward and final grader score by solving warehouse or e-commerce order exception cases correctly and efficiently.

Follow this policy strictly:
1. First identify the issue type.
2. Then assign the correct priority.
3. Then choose the best final resolution using stock availability, refund policy, shipment state, and customer tier.
4. Close the case only after the correct resolution has been selected.

Allowed action types only:
- identify_issue
- assign_priority
- resolve_case
- close_case

Allowed issue types:
- damaged_item
- wrong_item
- missing_item
- delivery_delay

Allowed priorities:
- low
- medium
- high

Allowed resolutions:
- approve_refund
- approve_replacement
- expedite_shipment
- escalate_to_human

Rules:
- Return exactly one valid JSON object.
- Do not include markdown.
- Do not explain reasoning.
- Do not invent unavailable actions or labels.
- Avoid repeated useless actions.
- Do not close the case before resolution is selected.
- If stock is unavailable, avoid approve_replacement.
- If refund is not allowed, avoid approve_refund.
- Premium or VIP customers with serious issues usually require higher priority.
- Delivery delays usually favor expedite_shipment before refund unless policy strongly suggests otherwise.
- Wrong item or damaged item should usually lead to replacement or refund depending on constraints.

Output schema:
{
  "action_type": "identify_issue|assign_priority|resolve_case|close_case",
  "payload": {}
}

Payload formats:
- identify_issue -> {"issue_type": "damaged_item|wrong_item|missing_item|delivery_delay"}
- assign_priority -> {"priority": "low|medium|high"}
- resolve_case -> {"resolution": "approve_refund|approve_replacement|expedite_shipment|escalate_to_human"}
- close_case -> {}

Return JSON only.
""".strip()

ALLOWED_ACTIONS = {
    "identify_issue": {"issue_type": {"damaged_item", "wrong_item", "missing_item", "delivery_delay"}},
    "assign_priority": {"priority": {"low", "medium", "high"}},
    "resolve_case": {"resolution": {"approve_refund", "approve_replacement", "expedite_shipment", "escalate_to_human"}},
    "close_case": {}
}


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done} error={error}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)


def build_user_prompt(observation, last_reward: float, history: List[str]) -> str:
    return f"""
Current warehouse exception case:

task_id: {observation.task_id}
difficulty: {observation.difficulty}
order_id: {observation.order_id}
shipment_status: {observation.shipment_status}
customer_tier: {observation.customer_tier}
known_issue_flags: {observation.known_issue_flags}
delay_days: {observation.delay_days}
stock_available: {observation.stock_available}
refund_allowed: {observation.refund_allowed}
last_action: {observation.last_action}
last_reward: {last_reward}

Previous action history:
{history if history else 'No previous actions.'}

Choose the single best next action to maximize task completion score and reward.
Return JSON only.
""".strip()


def fallback_action(observation, history: List[str]) -> Dict[str, Any]:
    if observation.last_action is None:
        flags = observation.known_issue_flags
        if flags.get("damage_flag"):
            return {"action_type": "identify_issue", "payload": {"issue_type": "damaged_item"}}
        if flags.get("wrong_item_flag"):
            return {"action_type": "identify_issue", "payload": {"issue_type": "wrong_item"}}
        if flags.get("missing_item_flag"):
            return {"action_type": "identify_issue", "payload": {"issue_type": "missing_item"}}
        return {"action_type": "identify_issue", "payload": {"issue_type": "delivery_delay"}}

    if observation.last_action == "identify_issue":
        priority = "high" if observation.customer_tier in {"premium", "vip"} or observation.delay_days >= 3 else "medium"
        return {"action_type": "assign_priority", "payload": {"priority": priority}}

    if observation.last_action == "assign_priority":
        flags = observation.known_issue_flags

        if flags.get("damage_flag"):
            resolution = "approve_replacement" if observation.stock_available else ("approve_refund" if observation.refund_allowed else "escalate_to_human")
        elif flags.get("wrong_item_flag") or flags.get("missing_item_flag"):
            resolution = "approve_replacement" if observation.stock_available else ("approve_refund" if observation.refund_allowed else "escalate_to_human")
        else:
            resolution = "expedite_shipment" if observation.shipment_status == "in_transit" else ("approve_refund" if observation.refund_allowed else "escalate_to_human")

        return {"action_type": "resolve_case", "payload": {"resolution": resolution}}

    return {"action_type": "close_case", "payload": {}}


def sanitize_action(raw: Dict[str, Any], observation, history: List[str]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return fallback_action(observation, history)

    action_type = raw.get("action_type")
    payload = raw.get("payload", {})

    if action_type not in ALLOWED_ACTIONS:
        return fallback_action(observation, history)

    if not isinstance(payload, dict):
        payload = {}

    rules = ALLOWED_ACTIONS[action_type]
    cleaned = {}

    for key, valid_values in rules.items():
        value = payload.get(key)
        if value not in valid_values:
            return fallback_action(observation, history)
        cleaned[key] = value

    if action_type == "close_case":
        cleaned = {}

    return {"action_type": action_type, "payload": cleaned}


def get_model_action(client: OpenAI, observation, last_reward: float, history: List[str]) -> Dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(observation, last_reward, history)}
            ]
        )
        text = response.choices[0].message.content.strip()
        raw = json.loads(text)
        return sanitize_action(raw, observation, history)
    except Exception:
        return fallback_action(observation, history)


def run_task(difficulty: str, client: OpenAI):
    env = WarehouseOrderExceptionEnv(difficulty=difficulty)
    result = env.reset()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    max_steps = 6
    last_reward = 0.0

    log_start(task=difficulty, env="warehouse-order-exception-openenv", model=os.getenv("MODEL_NAME", "gpt-4o-mini"))

    for step in range(1, max_steps + 1):
        if result.done:
            break

        action_dict = get_model_action(client, result.observation, last_reward, history)
        action = WarehouseAction(**action_dict)
        result = env.step(action)

        reward = result.reward or 0.0
        rewards.append(reward)
        steps_taken = step
        last_reward = reward

        compact_action = json.dumps(action_dict, separators=(",", ":"))
        log_step(step=step, action=compact_action, reward=reward, done=result.done, error=None)

        history.append(f"step={step} action={compact_action} reward={reward:.2f}")

        if result.done:
            break

    score = float((result.info or {}).get("grader_score", 0.0) or 0.0)
    success = score >= 0.7
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    client = OpenAI(
        api_key=os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("API_BASE_URL")
    )

    for difficulty in ["easy", "medium", "hard"]:
        run_task(difficulty, client)


if __name__ == "__main__":
    main()

import gradio as gr

def run_demo():
    return "✅ OpenEnv is running successfully!"

iface = gr.Interface(
    fn=run_demo,
    inputs=[],
    outputs="text",
    title="Warehouse OpenEnv",
    description="AI Agent for resolving warehouse order exceptions"
)

iface.launch(server_name="0.0.0.0", server_port=7860)