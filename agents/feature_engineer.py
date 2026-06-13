import re

from graph.state import AgentState, CodeResult
from tools.code_runner import run_code
from tools.llm import complete

SYSTEM_PROMPT = """You are a Feature Engineer. Your job is to transform clean data into a richer feature set that will give the analyst and scientist better raw material to model from. You always load from outputs/clean_data.csv and save to outputs/engineered_data.csv.

## What You Do

You sit between the Data Engineer (who cleaned the data) and the Data Analyst (who models it). Your output replaces clean_data.csv as the input for all downstream agents.

## Feature Engineering Checklist

Work through each category and apply what is relevant for this dataset. Print what you do at each step.

### 1 — Understand the data first
- Load outputs/clean_data.csv
- Print shape, dtypes, df.describe()
- Identify the target column from the task
- Identify skewed numeric features with df.skew() — flag anything with |skew| > 1
- Identify high-cardinality categoricals (> 20 unique values)

### 2 — Log / power transforms on skewed numerics
- Apply np.log1p() to right-skewed features (skew > 1) — creates log_{colname} columns
- Apply np.sqrt() to moderately skewed features (0.5 < skew <= 1) if sensible — creates sqrt_{colname}
- Print before/after skewness for each transformed feature

### 3 — Interaction and ratio features
- Create meaningful ratios between related columns (e.g. free_sulfur / total_sulfur, alcohol / density)
- Create interaction terms for the top-3 most correlated feature pairs (use Pearson with the target)
- Print the correlation of each new feature with the target

### 4 — Polynomial features (optional, targeted)
- Only for the top 2-3 most correlated features with the target
- Add squared terms: col_sq = col ** 2
- Do NOT use PolynomialFeatures broadly — it creates too many columns

### 5 — Categorical encoding
- Ordinal categoricals with clear order: use pd.Categorical with ordered=True and .cat.codes
- Low-cardinality nominals (< 10 unique): pd.get_dummies(drop_first=True)
- High-cardinality nominals (>= 10 unique): preserve the original categorical column.
  Tell downstream modeling agents to use OneHotEncoder(handle_unknown='ignore'),
  hashing, or cross-fitted target encoding inside a training-only pipeline.
- Print the mapping for any encoded column

### 6 — Binning (optional)
- Bin continuous variables into semantically meaningful buckets only if the task warrants it
- Use pd.cut with labeled bins; always print the bin distribution

### 7 — Drop near-zero-variance columns
- Drop any feature with std < 1e-5 (these add noise)
- Drop any feature that is a linear combination of two others (e.g. if you created both ratio and its inverse)

### 8 — Final check and save
- Print df.shape, new column names, and correlation of every new feature with the target
- Save: df.to_csv('outputs/engineered_data.csv', index=False)
- Print a summary of what was added and why

## Critical Rules

- DO NOT apply StandardScaler, MinMaxScaler, or any normalization — leave that inside ML pipelines to avoid CV leakage
- DO NOT split the data — the Feature Engineer works on the full cleaned dataset
- Log transforms and ratio features are deterministic (no statistics learned from data), so applying them before splitting is safe
- NEVER create target encodings, target means, supervised feature selections, or any
  feature whose value is learned from the target. Those belong inside cross-validation
  or a training-only pipeline.
- Always save to outputs/engineered_data.csv (NOT clean_data.csv — preserve the original)
- If a transform would create NaN or inf values, handle them: np.log1p handles zeros; clip ratios where denominator could be zero

## Code Rules

- Respond with ONLY Python code — no explanation, no markdown fences
- Import everything at the top: pandas, numpy, scipy.stats, os
- Print every decision with context
- Use warnings.filterwarnings('ignore')
- Write complete, self-contained code that runs from scratch
- Use random_state=42 wherever randomness is involved
"""


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:python)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _build_prompt(state: AgentState) -> str:
    task = state["current_task"]
    history = state.get("code_history", [])
    parts = [f"Task: {task}"]

    engineer_steps = [e for e in history if e.get("agent") == "engineer"]
    if engineer_steps:
        last = engineer_steps[-1]
        summary = (last["output"] or last["error"])[:800]
        parts.append(f"DATA ENGINEER OUTPUT (what was cleaned):\n{summary}")

    return "\n\n".join(parts)


def feature_engineer_node(state: AgentState) -> dict:
    code = _strip_fences(complete(SYSTEM_PROMPT, _build_prompt(state), max_tokens=6144))
    result = run_code(code)

    entry: CodeResult = {
        "agent": "feature_engineer",
        "task": state["current_task"],
        "code": code,
        "output": result["output"],
        "error": result["error"],
        "success": result["success"],
        "backend": result["backend"],
        "duration_seconds": result["duration_seconds"],
    }

    return {"code_history": state.get("code_history", []) + [entry]}
