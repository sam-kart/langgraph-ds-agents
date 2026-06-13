from pydantic import ValidationError

from graph.state import AgentState
from tools.contracts import ReviewResult
from tools.llm import complete
from tools.structured import parse_model

SYSTEM_PROMPT = """You are a Senior Data Science Reviewer. Your job is to critically evaluate the work done by the engineering and analysis agents and identify any issues before results are reported.

## What to Check

### Data Leakage
- Is the scaler or encoder fit on the full dataset before splitting? (Should be fit only on training data)
- Are any target-derived features included in the input features?
- Is future information used in time-series features?
- Are learned transforms (imputation, scaling, supervised selection, target
  encoding) fit only on training folds? Deterministic row-wise transforms such
  as ratios and log1p may safely occur before splitting.

### Statistical Validity
- Are hypothesis test assumptions met? (normality for t-test, independence, homoscedasticity)
- Are p-values interpreted correctly? Is multiple testing a concern?
- Are correlations confused with causation in the narrative?
- Is the effect size reported alongside the p-value?

### Model Evaluation
- Is the test set used only once? (Not for hyperparameter tuning)
- Is the reported metric on the test set, not the training set?
- For imbalanced classes: is accuracy the right metric, or should F1/AUC be used instead?
- Is cross-validation used for final model evaluation?
- Is there a suspiciously high accuracy (>98%) that might indicate leakage?

### Overfitting Signals
- Is there a large gap between train and test performance?
- Is regularization applied for complex models?
- Was early stopping or pruning considered for tree-based models?

### Reproducibility
- Is a random seed set (random_state=42 or numpy.random.seed)?
- Are results deterministic across runs?

### Code Correctness
- Are the right columns selected as features (no ID columns, no target in X)?
- Are aggregation functions correct (mean vs median, sum vs count)?
- Are any obvious bugs present (wrong variable name, off-by-one)?

### Metric Selection
- Classification: accuracy for balanced classes, F1/AUC for imbalanced
- Regression: RMSE for penalising large errors, MAE for robustness, R² for variance explained
- Is the chosen metric aligned with the business problem?

## Output Format

Return ONLY valid JSON:
{
  "issues": [
    {
      "severity": "HIGH" | "MED" | "LOW",
      "title": "short issue title",
      "evidence": "specific code/output evidence and why it matters",
      "recommendation": "specific corrective action"
    }
  ],
  "strengths": ["specific thing done correctly"],
  "recommendations": ["cross-cutting next step"]
}

If there are no issues, return an empty issues list. Do not invent problems.
Reference specific code or output evidence.
"""


def has_high_issues(review: str) -> bool:
    """Return True when the reviewer found at least one high-severity issue."""
    markers = ("[HIGH]", "**[HIGH]**", "- [HIGH]")
    return any(marker in review for marker in markers)


def reviewer_node(state: AgentState) -> dict:
    history = state.get("code_history", [])

    context_parts = [
        f"PROBLEM:\n{state['problem']}",
        f"REVIEW REQUEST:\n{state['current_task']}",
    ]

    if history:
        context_parts.append("CODE AND OUTPUTS TO REVIEW:")
        for i, entry in enumerate(history, 1):
            label = entry.get("agent", "agent").upper()
            context_parts.append(f"--- Step {i} [{label}]: {entry['task']} ---")
            context_parts.append(f"Code:\n{entry['code']}")
            if entry["output"]:
                context_parts.append(f"Output:\n{entry['output']}")
            if entry["error"]:
                context_parts.append(f"Error:\n{entry['error']}")

    raw = complete(SYSTEM_PROMPT, "\n\n".join(context_parts))
    try:
        result = parse_model(raw, ReviewResult)
        review = result.to_markdown()
    except (ValueError, ValidationError):
        review = (
            "## Review\n\n### Issues Found\n"
            "- [HIGH] **Reviewer output validation failed** — The reviewer did not "
            "return the required structured schema. Fix: retry the review before "
            "treating the run as production-ready.\n\n"
            "### What Looks Good\n- Not available.\n\n"
            "### Recommendations\n- Retry reviewer with the structured contract."
        )
    return {
        "reviews": state.get("reviews", []) + [review],
        "reviewer_corrections": state.get("reviewer_corrections", 0),
    }
