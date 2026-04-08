import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "task_bank.json"

def load_tasks():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_task_by_difficulty(difficulty: str):
    tasks = load_tasks()
    for task in tasks:
        if task["difficulty"] == difficulty:
            return task
    raise ValueError(f"No task found for difficulty={difficulty}")