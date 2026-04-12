import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from openenv.server import OpenEnvServer  # Or from openenv_core.env_server import Environment if different
from warehouse_env import WarehouseOrderExceptionEnv
from models import WarehouseAction

# ===== YOUR EXISTING CODE - PASTE ALL THESE EXACTLY =====
SYSTEM_PROMPT = """
You are a deterministic warehouse order exception resolution agent.
... (your full SYSTEM_PROMPT exactly as you have it)
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
... (your full function body exactly)
""".strip()

def fallback_action(observation, history: List[str]) -> Dict[str, Any]:
    # Your full fallback_action exactly
    if observation.last_action is None:
        # ... your exact logic
    # ... rest exactly

def sanitize_action(raw: Dict[str, Any], observation, history: List[str]) -> Dict[str, Any]:
    # Your full sanitize_action exactly

def get_model_action(client: OpenAI, observation, last_reward: float, history: List[str]) -> Dict[str, Any]:
    # Your full get_model_action exactly
    # ... 

# ===== END OF YOUR CODE =====

class WarehouseOpenEnv(OpenEnvServer):
    def __init__(self):
        super().__init__()
        self.env = None
        self.history: List[str] = []
        self.rewards: List[float] = []
        self.last_reward = 0.0
        self.client = OpenAI(
            api_key=os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("API_BASE_URL")
        )

    def reset(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            difficulty = task_config.get("difficulty", "medium")
            self.env = WarehouseOrderExceptionEnv(difficulty=difficulty)
            result = self.env.reset()
            
            self.history = []
            self.rewards = []
            self.last_reward = 0.0
            log_start(task=difficulty, env="warehouse-order-exception-openenv", 
                     model=os.getenv("MODEL_NAME", "gpt-4o-mini"))
            
            obs_dict = result.observation.__dict__ if hasattr(result.observation, '__dict__') else {}
            return {
                "observation": obs_dict,
                "reward": float(result.reward or 0.0),
                "done": result.done,
                "info": result.info or {}
            }
        except Exception as e:
            return {"observation": {"error": str(e)}, "reward": 0.0, "done": True, "info": {"error": str(e)}}

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if self.env is None:
                return {"observation": {"error": "Call reset first"}, "reward": 0.0, "done": True, "info": {}}

            # Get model decision
            action_dict = get_model_action(self.client, self.env.observation, self.last_reward, self.history)
            warehouse_action = WarehouseAction(**action_dict)
            
            result = self.env.step(warehouse_action)
            
            reward = float(result.reward or 0.0)
            self.rewards.append(reward)
            self.last_reward = reward
            
            step_num = len(self.rewards)
            compact_action = json.dumps(action_dict, separators=(",", ":"))
            self.history.append(f"step={step_num} {compact_action}")
            
            log_step(step=step_num, action=compact_action, reward=reward, done=result.done)
            
            obs_dict = result.observation.__dict__ if hasattr(result.observation, '__dict__') else {}
            info = result.info or {}
            info["total_reward"] = sum(self.rewards)
            
            return {
                "observation": obs_dict,
                "reward": reward,
                "done": result.done,
                "info": info
            }
        except Exception as e:
            log_step(len(self.rewards), json.dumps(action), 0.0, True, str(e))
            return {"observation": {"error": str(e)}, "reward": 0.0, "done": True, "info": {"error": str(e)}}

if __name__ == "__main__":
    app = WarehouseOpenEnv()
    app.run(host="0.0.0.0", port=7860)
