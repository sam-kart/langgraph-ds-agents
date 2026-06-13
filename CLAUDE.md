# The Pog — DS Multi-Agent System

## What This Project Is

A multi-agent AI system built with LangGraph that solves Data Science problems end-to-end from a natural language prompt. Eight agents collaborate in a directed graph: a **Manager** that orchestrates everything, and seven specialist agents covering the full DS pipeline.

## Agent Roles

| Agent | File | Responsibility |
|-------|------|----------------|
| Manager | `agents/manager.py` | Plans, reviews results, routes to the right agent each turn |
| Data Engineer | `agents/engineer.py` | Scrapes/loads/cleans data, saves `clean_data.csv` |
| Feature Engineer | `agents/feature_engineer.py` | Log transforms, interactions, ratios, encoding — saves `engineered_data.csv` |
| Data Analyst | `agents/analyst.py` | EDA, statistical tests, ML modeling, visualization (first pass) |
| Senior Scientist | `agents/scientist.py` | Critiques analyst's approach, implements improvements, produces before/after comparison |
| Reviewer | `agents/reviewer.py` | Validates methodology across all agents, catches data leakage |
| Report Writer | `agents/reporter.py` | Compiles all findings into `outputs/report_TIMESTAMP.md` |
| Dashboard Builder | `agents/dashboarder.py` | Builds a static `dashboard.html` optimized for readability and stakeholder review |

## Graph Flow

```
START → Manager
          ├──► Data Engineer      → Manager
          ├──► Feature Engineer   → Manager
          ├──► Data Analyst       → Manager
          ├──► Senior Scientist   → Manager
          ├──► Reviewer           → Manager
          ├──► Report Writer      → Dashboard Builder → END
          └──► done               → END
```

Standard workflow: engineer → feature_engineer → analyst → scientist → reviewer → reporter → dashboarder

## Agent Output Convention

- Engineer saves `outputs/clean_data.csv`
- Feature Engineer saves `outputs/engineered_data.csv` (analyst/scientist load this if it exists)
- Scientist saves plots with `scientist_` prefix to avoid overwriting analyst's plots
- Dashboard Builder saves `outputs/dashboard.html`

## Project Structure

```
The Pog/
├── agents/
│   ├── manager.py          # Orchestrator — routes between all agents
│   ├── engineer.py         # Data acquisition + cleaning (writes clean_data.csv)
│   ├── feature_engineer.py # Feature engineering (writes engineered_data.csv)
│   ├── analyst.py          # EDA + ML + stats + visualization
│   ├── scientist.py        # Improvement agent — runs code to beat analyst metrics
│   ├── reviewer.py         # Methodology review (text output, no code execution)
│   ├── reporter.py         # Final markdown report writer
│   └── dashboarder.py      # Static HTML dashboard builder
├── graph/
│   ├── state.py            # AgentState TypedDict — shared across all nodes
│   └── workflow.py         # LangGraph StateGraph wiring
├── tools/
│   ├── llm.py              # Shared LLM helper with prompt caching
│   └── code_runner.py      # Subprocess executor (60s timeout, plots → outputs/)
├── outputs/                # All generated plots (.png) and reports (.md) land here
├── main.py                 # Rich-formatted CLI entrypoint
├── requirements.txt
├── .venv/                  # Python 3.10 virtualenv
└── .vscode/settings.json   # Points VS Code to .venv interpreter
```

## How to Run

```bash
source .venv/bin/activate
export ANTHROPIC_API_KEY=your-api-key-here

# Interactive
python main.py

# One-liner
python main.py "Scrape the Titanic dataset, clean it, train a classifier, and generate a report"
```

## State Fields

```python
AgentState:
  problem        # Original user prompt
  plan           # Manager's reasoning from first turn
  current_task   # Instruction passed to the active agent
  code_history   # List[CodeResult] — all code runs (agent field tags each entry)
  reviews        # List[str] — reviewer outputs
  reviewer_corrections # Number of reviewer-triggered correction passes used
  iteration      # Loop counter (max 20)
  next           # Routing key: "engineer"|"feature_engineer"|"analyst"|"scientist"|"reviewer"|"reporter"|"done"
  final_summary  # Set on completion
  report_path    # Path to generated .md report
  dashboard_path # Path to generated dashboard.html
```

## Key Design Decisions

- Engineer saves `clean_data.csv`; Feature Engineer saves `engineered_data.csv` — clean original is preserved
- Feature Engineer does NOT scale data — scaling stays inside ML pipelines to prevent CV leakage
- Feature Engineer never performs full-dataset target encoding or supervised feature selection
- Reviewer generates text only (no subprocess) — pure LLM analysis of code history
- Reporter hands off to Dashboard Builder; manager guard forces reporter before "done"
- Dashboard Builder runs after Reporter and writes a static HTML dashboard
- If reviewer finds a HIGH issue, manager gives scientist one correction pass before reporting
- `matplotlib.use('Agg')` forced in code runner — no display windows, all plots saved as PNG
- Production execution uses a hardened Docker sandbox; local mode is an explicit unsafe development fallback
- LangGraph state is persisted in per-run SQLite checkpoints
- Manager and Reviewer outputs are validated with Pydantic contracts
- Max 20 iterations to prevent runaway loops while allowing correction retries
- LLM: `claude-sonnet-4-6` for all agents via direct Anthropic SDK (prompt caching on system prompts)

## Adding a New Agent

1. Create `agents/<name>.py` with `SYSTEM_PROMPT` (>1024 tokens for caching) and `<name>_node(state) -> dict`
2. Add node to `graph/workflow.py`: `g.add_node(...)`, routing edge, `g.add_edge(..., "manager")`
3. Add routing key to `conditional_edges` map in `workflow.py`
4. Add agent description to manager's `SYSTEM_PROMPT` and standard workflow
5. Add routing key to manager's response format JSON
6. Add `AGENT_STYLES` entry in `main.py`

## Scaling Roadmap

Next agents to consider:
- **Statistician** — deep statistical modeling (time series, Bayesian, survival analysis)
- **ML Tuner** — hyperparameter optimization with Optuna/GridSearch
- **Presenter** — converts report to slides or interactive HTML dashboard
