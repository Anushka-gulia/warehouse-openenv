from warehouse_env import WarehouseOrderExceptionEnv
from models import WarehouseAction
from fastapi import FastAPI

# ✅ CREATE APP AT TOP LEVEL
app = FastAPI()

@app.get("/")
def home():
    return {"message": "OpenEnv is running successfully!"}

# ✅ REQUIRED FOR SUBMISSION
@app.post("/reset")
def reset():
    env = WarehouseOrderExceptionEnv(difficulty="easy")
    result = env.reset()
    return result.model_dump()

# (Optional but good)
@app.post("/step")
def step(action: dict):
    env = WarehouseOrderExceptionEnv(difficulty="easy")
    result = env.step(WarehouseAction(**action))
    return result.model_dump()


# 👇 OPTIONAL DEMO (NOT USED BY SERVER)
def demo():
    env = WarehouseOrderExceptionEnv(difficulty="easy")
    result = env.reset()
    print(result.model_dump())


if __name__ == "__main__":
    demo()