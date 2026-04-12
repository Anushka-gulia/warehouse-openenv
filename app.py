from fastapi import FastAPI
from warehouse_env import WarehouseOrderExceptionEnv
from models import WarehouseAction

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
