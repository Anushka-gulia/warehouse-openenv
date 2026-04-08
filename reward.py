def compute_reward(state, action_type: str):
    task = state.task
    reward = -0.02

    if action_type == "identify_issue":
        proposed = state.history[-1]["payload"].get("issue_type")
        if proposed == task["issue_type"]:
            reward += 0.25
        else:
            reward -= 0.05

    elif action_type == "assign_priority":
        proposed = state.history[-1]["payload"].get("priority")
        if proposed == task["correct_priority"]:
            reward += 0.20
        else:
            reward -= 0.05

    elif action_type == "resolve_case":
        proposed = state.history[-1]["payload"].get("resolution")
        if proposed == task["correct_resolution"]:
            reward += 0.35
        else:
            reward -= 0.10

    elif action_type == "close_case":
        if state.final_resolution == task["correct_resolution"]:
            reward += 0.20
        else:
            reward -= 0.10

    return max(-1.0, min(reward, 1.0))