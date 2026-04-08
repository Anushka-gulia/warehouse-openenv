from copy import deepcopy
from copy import deepcopy
from models import WarehouseAction, WarehouseObservation, WarehouseState, StepResult
from tasks import get_task_by_difficulty
from reward import compute_reward
from graders import grade_task

class WarehouseOrderExceptionEnv:
    def __init__(self, difficulty: str = "easy", max_steps: int = 6):
        self.difficulty = difficulty
        self.max_steps = max_steps
        self._state = None

    def reset(self):
        task = deepcopy(get_task_by_difficulty(self.difficulty))
        self._state = WarehouseState(
            step_count=0,
            max_steps=self.max_steps,
            task=task,
            resolved=False,
            history=[]
        )
        return StepResult(
            observation=self._build_observation("New order exception case loaded."),
            reward=0.0,
            done=False,
            info={"task_id": task["task_id"]}
        )

    def state(self):
        return self._state

    def step(self, action: WarehouseAction):
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._state.step_count += 1
        self._state.history.append({
            "action_type": action.action_type,
            "payload": action.payload
        })

        if action.action_type == "identify_issue":
            self._state.identified_issue = action.payload.get("issue_type")

        elif action.action_type == "assign_priority":
            self._state.assigned_priority = action.payload.get("priority")

        elif action.action_type == "resolve_case":
            self._state.final_resolution = action.payload.get("resolution")

        elif action.action_type == "close_case":
            self._state.resolved = True

        reward = compute_reward(self._state, action.action_type)
        done = self._state.resolved or self._state.step_count >= self._state.max_steps
        score = grade_task(self._state) if done else None

        return StepResult(
            observation=self._build_observation(f"Action processed: {action.action_type}"),
            reward=reward,
            done=done,
            info={"grader_score": score}
        )

    def _build_observation(self, message: str):
        task = self._state.task
        return WarehouseObservation(
            task_id=task["task_id"],
            difficulty=task["difficulty"],
            order_id=task["order_id"],
            shipment_status=task["shipment_status"],
            customer_tier=task["customer_tier"],
            known_issue_flags={
                "damage_flag": task["damage_flag"],
                "wrong_item_flag": task["wrong_item_flag"],
                "missing_item_flag": task["missing_item_flag"]
            },
            delay_days=task["delay_days"],
            stock_available=task["stock_available"],
            refund_allowed=task["refund_allowed"],
            last_action=self._state.history[-1]["action_type"] if self._state.history else None,
            message=message
        )