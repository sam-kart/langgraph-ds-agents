from typing import TypedDict


class CodeResult(TypedDict):
    agent: str    # "engineer" | "feature_engineer" | "analyst" | "scientist"
    task: str
    code: str
    output: str
    error: str
    success: bool
    backend: str
    duration_seconds: float


class AgentState(TypedDict):
    problem: str          # Original user DS problem
    plan: str             # Manager's initial plan
    current_task: str     # Current instruction for the assigned agent
    code_history: list    # List of CodeResult dicts
    reviews: list         # List of review strings from reviewer
    reviewer_corrections: int
    iteration: int
    next: str             # "engineer"|"feature_engineer"|"analyst"|"scientist"|"reviewer"|"reporter"|"dashboarder"|"done"
    final_summary: str
    report_path: str      # Path to the generated markdown report
    dashboard_path: str   # Path to the generated HTML dashboard
