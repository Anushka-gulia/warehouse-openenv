from fastapi import FastAPI
from warehouse_env import WarehouseOrderExceptionEnv
from models import WarehouseAction
from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from warehouse_env import WarehouseOrderExceptionEnv
from models import WarehouseAction

app = FastAPI(title="warehouse-order-exception-openenv", version="0.1.0")

_sessions: Dict[str, WarehouseOrderExceptionEnv] = {}


class ResetRequest(BaseModel):
    difficulty: str = Field(default="easy")


class StepRequest(BaseModel):
    session_id: str
    action_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "warehouse-order-exception-openenv",
        "status": "ok",
        "routes": ["GET /health", "POST /reset", "POST /step", "GET /state/{session_id}"],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None) -> Dict[str, Any]:
    difficulty = (req.difficulty if req else "easy")
    env = WarehouseOrderExceptionEnv(difficulty=difficulty)
    session_id = str(uuid4())
    _sessions[session_id] = env

    result = env.reset()
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    data["session_id"] = session_id
    data["difficulty"] = difficulty
    return data


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    action = WarehouseAction(action_type=req.action_type, payload=req.payload)
    result = env.step(action)
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    data["session_id"] = req.session_id
    return data


@app.get("/state/{session_id}")
def state(session_id: str) -> Dict[str, Any]:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Invalid session_id")

    if hasattr(env, "state"):
        current = env.state()
        return current.model_dump() if hasattr(current, "model_dump") else dict(current)

    if hasattr(env, "get_state"):
        current = env.get_state()
        return current.model_dump() if hasattr(current, "model_dump") else dict(current)

    return {
        "session_id": session_id,
        "detail": "State endpoint available, but WarehouseOrderExceptionEnv does not expose state()/get_state().",
    }
app = FastAPI()

# create a global environment
env = WarehouseOrderExceptionEnv(difficulty="easy")

@app.get("/")
def home():
    return {"message": "OpenEnv API is running"}

# ✅ REQUIRED ENDPOINT
@app.post("/reset")
def reset():
    result = env.reset()
    return result.model_dump()

# ✅ REQUIRED ENDPOINT
@app.post("/step")
def step(action: dict):
    action_obj = WarehouseAction(**action)
    result = env.step(action_obj)
    return result.model_dump()
