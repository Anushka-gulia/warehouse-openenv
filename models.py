from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class WarehouseAction(BaseModel):
    action_type: str = Field(..., description="Type of action taken by the agent")
    payload: Optional[Dict[str, Any]] = Field(default_factory=dict)

class WarehouseObservation(BaseModel):
    task_id: str
    difficulty: str
    order_id: str
    shipment_status: str
    customer_tier: str
    known_issue_flags: Dict[str, bool]
    delay_days: int
    stock_available: bool
    refund_allowed: bool
    last_action: Optional[str] = None
    message: str

class WarehouseState(BaseModel):
    step_count: int
    max_steps: int
    task: Dict[str, Any]
    resolved: bool = False
    history: List[Dict[str, Any]] = Field(default_factory=list)
    identified_issue: Optional[str] = None
    assigned_priority: Optional[str] = None
    final_resolution: Optional[str] = None

class StepResult(BaseModel):
    observation: WarehouseObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)