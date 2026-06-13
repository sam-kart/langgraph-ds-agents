from pydantic import ValidationError

from graph.state import AgentState
from tools.contracts import ManagerDecision
from tools.llm import complete
from tools.structured import parse_model

MAX_ITERATIONS = 20

SYSTEM_PROMPT = """You are a Data Science team manager. You coordinate a team of seven specialized agents to solve data science problems end-to-end. Your role is to plan the work, review results after each step, and route the next task to the right agent.

## Your Team

**engineer** — Data Engineer
Responsibilities: fetching data from URLs, APIs, or web pages; loading CSV/JSON/Excel files; cleaning raw data (nulls, duplicates, bad types, outliers); saving the cleaned dataset as clean_data.csv in the outputs directory. Does NOT do feature engineering.
When to use: always first if data needs to be acquired or cleaned. Also use if the analyst encounters data quality issues.

**feature_engineer** — Feature Engineer
Responsibilities: loading clean_data.csv and enriching it with new features — log/power transforms on skewed columns, interaction terms, ratios between related columns, polynomial terms for top predictors, categorical encoding. Saves the result as engineered_data.csv. Does NOT apply scaling (leaves that inside ML pipelines to prevent CV leakage).
When to use: after the engineer, before the analyst. Skip only for very simple problems (< 4 features, already well-behaved distributions) or when the task explicitly requests no feature engineering.

**analyst** — Data Analyst
Responsibilities: exploratory data analysis (distributions, correlations, class balance); statistical hypothesis testing; machine learning (train/evaluate classifiers and regressors with sklearn); visualization (histograms, scatter plots, heatmaps, confusion matrices, feature importance plots); dimensionality reduction.
When to use: after the engineer has prepared data. The analyst does the first-pass modeling and EDA.

**scientist** — Senior Data Scientist
Responsibilities: scrutinizes the analyst's approach, identifies weaknesses, and implements concrete improvements that attempt to beat the analyst's metrics. Tries alternative algorithms (XGBoost, ensembles, stacking), better feature engineering, hyperparameter tuning, or alternative problem framings. Always produces an explicit before/after comparison (Analyst vs Scientist metrics). Saves plots with a "scientist_" prefix.
When to use: after the analyst has completed initial modeling. Always route analyst → scientist before reviewer. Skip only for very simple problems where analyst results are already strong and well-validated.

**reviewer** — Senior Reviewer
Responsibilities: auditing the full code history (analyst + scientist) for data leakage, incorrect metric selection, overfitting signals, improper train-test splitting, missing random seeds, statistical assumption violations, and reproducibility issues.
When to use: after the scientist has run (or after the analyst if scientist was skipped). Always before reporter.

**reporter** — Report Writer
Responsibilities: compiling all findings, code outputs, metrics from both analyst and scientist, reviewer notes, and plots into a structured markdown report saved to outputs/.
When to use: MANDATORY — you MUST route to reporter after reviewer completes. Never set status="done" until the reporter has run. The dashboard builder runs automatically after reporter.

**dashboarder** — Dashboard Builder
Responsibilities: creating a polished static HTML dashboard from the report, metrics, reviewer notes, datasets, and generated plots. Optimizes for audience readability, scan-friendly hierarchy, caveats, and visual evidence.
When to use: the graph runs this automatically after reporter. You normally do not need to route to it directly.

## Standard Workflow

engineer → feature_engineer → analyst → scientist → reviewer → reporter → dashboarder → done

Deviations:
- Skip engineer if data is already clean or synthetic
- Skip feature_engineer only for very simple problems (< 4 features, already well-behaved) or if explicitly told not to engineer features
- Skip scientist only if the analyst's first pass is trivially simple (e.g., dataset with 2 features)
- If any agent errors, route back to the same agent with a corrected instruction
- NEVER set status="done" unless report_path is already set in state (reporter has already run)

## Task Writing Guidelines

Good tasks are specific and self-contained:
- For scientist: explicitly state the analyst's best metric (e.g., "the analyst achieved RMSE=0.617 — try to improve it by...") and name 1-2 specific approaches to try
- Include column names, model types, metric names, and plot file names
- Reference prior outputs when relevant (e.g., "load clean_data.csv, then...")

## Response Format

Respond with a JSON object and no markdown fences:
{
  "reasoning": "<your analysis of what has been done and what is needed next — be specific>",
  "status": "continuing" | "done",
  "agent": "engineer" | "feature_engineer" | "analyst" | "scientist" | "reviewer" | "reporter" | "dashboarder" | null,
  "task": "<the specific instruction for the chosen agent — null only if done>",
    "summary": "<brief final summary for the user — null if not done>"
}
"""


def _tail(text: str, chars: int) -> str:
    """Return the last `chars` characters of text — metrics are usually at the end."""
    return text[-chars:] if len(text) > chars else text


def build_context(state: AgentState) -> str:
    parts = [f"PROBLEM:\n{state['problem']}"]

    if state.get("plan"):
        parts.append(f"INITIAL PLAN:\n{state['plan']}")

    history = state.get("code_history", [])
    if history:
        parts.append("CODE HISTORY:")
        for i, entry in enumerate(history, 1):
            label = entry.get("agent", "agent").upper()
            is_latest = (i == len(history))
            parts.append(f"--- Step {i} [{label}]: {entry['task']} ---")

            if is_latest:
                # Latest step: full code (capped) + full output tail (where metrics live)
                code = entry["code"]
                if len(code) > 3000:
                    code = code[:3000] + "\n... [truncated for brevity]"
                parts.append(f"Code:\n{code}")
                if entry["output"]:
                    parts.append(f"Output:\n{_tail(entry['output'], 2000)}")
                if entry["error"]:
                    parts.append(f"Error:\n{entry['error']}")
            else:
                # Older steps: output summary only — no code, keeps context lean
                if entry["output"]:
                    parts.append(f"Output summary:\n{_tail(entry['output'], 500)}")
                if entry["error"]:
                    parts.append(f"Error:\n{entry['error'][:300]}")

    reviews = state.get("reviews", [])
    if reviews:
        parts.append("REVIEWS:")
        for i, r in enumerate(reviews, 1):
            parts.append(f"--- Review {i} ---\n{r}")

    if state.get("report_path"):
        parts.append(f"REPORT GENERATED: {state['report_path']}")

    return "\n\n".join(parts)


REPORTER_FALLBACK_TASK = (
    "Compile all findings, metrics, code outputs, reviewer notes, "
    "and generated plots into a structured markdown report saved to outputs/."
)

REVIEW_CORRECTION_TASK = (
    "Address the latest reviewer findings before reporting. Focus only on HIGH "
    "severity issues. Re-run the corrected modeling/evaluation code, print the "
    "before/after metrics, and save any updated plots with a corrected_ prefix."
)

MAX_REVIEWER_CORRECTIONS = 1


def _must_run_reporter(state: AgentState) -> bool:
    return not state.get("report_path")


def _latest_review_has_high_issue(state: AgentState) -> bool:
    reviews = state.get("reviews", [])
    if not reviews:
        return False
    latest = reviews[-1]
    markers = ("[HIGH]", "**[HIGH]**", "- [HIGH]")
    return any(marker in latest for marker in markers)


def _needs_review_correction(state: AgentState) -> bool:
    return (
        _latest_review_has_high_issue(state)
        and state.get("reviewer_corrections", 0) < MAX_REVIEWER_CORRECTIONS
    )


def manager_node(state: AgentState) -> dict:
    iteration = state.get("iteration", 0)

    try:
        raw = complete(SYSTEM_PROMPT, build_context(state), max_tokens=4096)
        decision = parse_model(raw, ManagerDecision)
    except (ValueError, ValidationError):
        # Malformed response — preserve the correction/reporting guarantees.
        if _needs_review_correction(state):
            return {
                "iteration": iteration + 1,
                "next": "scientist",
                "current_task": REVIEW_CORRECTION_TASK,
                "reviewer_corrections": state.get("reviewer_corrections", 0) + 1,
            }
        if _must_run_reporter(state):
            return {
                "iteration": iteration + 1,
                "next": "reporter",
                "current_task": REPORTER_FALLBACK_TASK,
            }
        return {
            "iteration": iteration + 1,
            "next": "done",
            "final_summary": "Manager response was malformed. Partial results saved to outputs/.",
        }

    status = decision.status

    if iteration >= MAX_ITERATIONS:
        status = "done"
        decision.summary = (
            decision.summary or
            "Reached maximum iterations. See outputs/ for results."
        )

    # If the reviewer found a HIGH issue, give the scientist one correction pass
    # before allowing the final report.
    if _needs_review_correction(state) and (
        status == "done" or decision.agent == "reporter"
    ):
        return {
            "iteration": iteration + 1,
            "next": "scientist",
            "current_task": REVIEW_CORRECTION_TASK,
            "reviewer_corrections": state.get("reviewer_corrections", 0) + 1,
        }

    # Guard: never allow "done" before the reporter has run
    if status == "done" and _must_run_reporter(state):
        status = "continuing"
        decision.agent = "reporter"
        decision.task = REPORTER_FALLBACK_TASK

    updates: dict = {"iteration": iteration + 1}

    if status == "continuing":
        updates["next"] = decision.agent or "analyst"
        updates["current_task"] = decision.task or "Continue the analysis."
        if not state.get("plan"):
            updates["plan"] = decision.reasoning
    else:
        updates["next"] = "done"
        updates["final_summary"] = decision.summary or ""

    return updates
