def grade_task(state):
    task = state.task
    score = 0.0

    if state.identified_issue == task["issue_type"]:
        score += 0.3

    if state.assigned_priority == task["correct_priority"]:
        score += 0.2

    if state.final_resolution == task["correct_resolution"]:
        score += 0.4

    if state.resolved:
        score += 0.1

    return max(0.0, min(score, 1.0))