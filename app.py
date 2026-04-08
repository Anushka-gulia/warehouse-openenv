from warehouse_env import WarehouseOrderExceptionEnv
from models import WarehouseAction

def demo():
    env = WarehouseOrderExceptionEnv(difficulty="easy")
    result = env.reset()
    print(result.model_dump())

    result = env.step(WarehouseAction(action_type="identify_issue", payload={"issue_type": "damaged_item"}))
    print(result.model_dump())

    result = env.step(WarehouseAction(action_type="assign_priority", payload={"priority": "medium"}))
    print(result.model_dump())

    result = env.step(WarehouseAction(action_type="resolve_case", payload={"resolution": "approve_replacement"}))
    print(result.model_dump())

    result = env.step(WarehouseAction(action_type="close_case", payload={}))
    print(result.model_dump())

if __name__ == "__main__":
    demo()