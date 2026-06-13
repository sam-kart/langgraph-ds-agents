#!/usr/bin/env python3
import json
import os
import sqlite3
import sys
from datetime import datetime

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from graph.workflow import build_graph
from tools.llm import MODEL, get_usage_summary

load_dotenv()  # loads ANTHROPIC_API_KEY from .env if present

# Create a per-run output directory before any agent imports so code_runner picks it up
_run_tag = os.environ.get("DS_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
_run_dir = os.path.join(os.path.dirname(__file__), "outputs", f"run_{_run_tag}")
os.makedirs(_run_dir, exist_ok=True)
os.environ["DS_OUTPUTS_DIR"] = _run_dir
_checkpoint_path = os.path.join(_run_dir, "checkpoints.sqlite")
_resuming = os.path.exists(_checkpoint_path)

console = Console()

AGENT_STYLES = {
    "manager":          ("Manager",            "bold blue",       "blue"),
    "engineer":         ("Data Engineer",      "bold cyan",       "cyan"),
    "feature_engineer": ("Feature Engineer",   "bold orange3",    "orange3"),
    "analyst":          ("Data Analyst",       "bold green",      "green"),
    "scientist":        ("Senior Scientist",   "bold red",        "red"),
    "reviewer":         ("Reviewer",           "bold yellow",     "yellow"),
    "reporter":         ("Report Writer",      "bold magenta",    "magenta"),
    "dashboarder":      ("Dashboard Builder",  "bold bright_white", "white"),
}


def run(problem: str):
    checkpoint_conn = sqlite3.connect(_checkpoint_path, check_same_thread=False)
    graph = build_graph(checkpointer=SqliteSaver(checkpoint_conn))
    started_at = datetime.now()
    run_metadata = {
        "problem": problem,
        "started_at": started_at.isoformat(timespec="seconds"),
        "run_dir": _run_dir,
        "artifacts_dir": os.path.join(_run_dir, "outputs"),
        "model": MODEL,
        "agents_called": [],
        "events": [],
        "success": False,
        "resumed": _resuming,
        "checkpoint_path": _checkpoint_path,
    }

    initial_state = {
        "problem": problem,
        "plan": "",
        "current_task": "",
        "code_history": [],
        "reviews": [],
        "reviewer_corrections": 0,
        "iteration": 0,
        "next": "engineer",
        "final_summary": "",
        "report_path": "",
        "dashboard_path": "",
    }
    final_state = dict(initial_state)

    console.print(Panel(problem, title="[bold blue]Problem[/bold blue]", border_style="blue"))
    console.print()

    graph_input = None if _resuming else initial_state
    config = {"configurable": {"thread_id": _run_tag}}
    try:
        for event in graph.stream(graph_input, config=config, stream_mode="updates"):
            for node, updates in event.items():
                final_state.update(updates)
                run_metadata["agents_called"].append(node)
                run_metadata["events"].append(
                    {
                        "node": node,
                        "next": updates.get("next"),
                        "task": updates.get("current_task", "")[:500],
                        "report_path": updates.get("report_path", ""),
                    }
                )
                label, rich_style, border = AGENT_STYLES.get(node, (node, "white", "white"))
                console.rule(f"[{rich_style}]{label}[/{rich_style}]", style=border)

                if node == "manager":
                    task = updates.get("current_task")
                    if task:
                        console.print(f"[dim]Routing to:[/dim] [bold]{updates.get('next','?').upper()}[/bold]")
                        console.print(f"[dim]Task:[/dim] {task}\n")

                elif node in ("engineer", "feature_engineer", "analyst", "scientist"):
                    history = updates.get("code_history", [])
                    if history:
                        entry = history[-1]
                        console.print(Syntax(entry["code"], "python", theme="monokai", line_numbers=True))
                        console.print(f"[dim]Execution backend: {entry.get('backend', 'unknown')}[/dim]")
                        if entry["output"]:
                            console.print(Panel(entry["output"], title="Output", border_style="dim"))
                        if entry["error"]:
                            console.print(Panel(entry["error"], title="[red]Error[/red]", border_style="red"))

                elif node == "reviewer":
                    reviews = updates.get("reviews", [])
                    if reviews:
                        console.print(Markdown(reviews[-1]))

                elif node == "reporter":
                    path = updates.get("report_path", "")
                    if path:
                        console.print(f"[magenta]Report saved:[/magenta] {path}")

                elif node == "dashboarder":
                    path = updates.get("dashboard_path", "")
                    if path:
                        console.print(f"[white]Dashboard saved:[/white] {path}")
        snapshot = graph.get_state(config)
        if snapshot.values:
            final_state.update(snapshot.values)
    finally:
        checkpoint_conn.close()

    console.print()
    console.rule("[bold]Done[/bold]")

    _artifacts_dir = os.path.join(_run_dir, "outputs")
    console.print(f"[dim]Run dir: {_run_dir}[/dim]")
    if os.path.isdir(_artifacts_dir):
        plots = sorted(f for f in os.listdir(_artifacts_dir) if f.endswith(".png"))
        reports = sorted(f for f in os.listdir(_artifacts_dir) if f.endswith(".md"))
        dashboards = sorted(f for f in os.listdir(_artifacts_dir) if f.endswith(".html"))
        if plots:
            console.print(f"[dim]Plots:   {', '.join(plots)}[/dim]")
        if reports:
            console.print(f"[dim]Reports: {', '.join(reports)}[/dim]")
        if dashboards:
            console.print(f"[dim]Dashboards: {', '.join(dashboards)}[/dim]")

    ended_at = datetime.now()
    artifacts = []
    if os.path.isdir(_artifacts_dir):
        for root, _, files in os.walk(_artifacts_dir):
            for file_name in files:
                path = os.path.join(root, file_name)
                artifacts.append(os.path.relpath(path, _run_dir))

    run_metadata.update(
        {
            "ended_at": ended_at.isoformat(timespec="seconds"),
            "duration_seconds": round((ended_at - started_at).total_seconds(), 2),
            "total_iterations": final_state.get("iteration", 0),
            "report_path": final_state.get("report_path", ""),
            "dashboard_path": final_state.get("dashboard_path", ""),
            "final_summary": final_state.get("final_summary", ""),
            "reviewer_corrections": final_state.get("reviewer_corrections", 0),
            "artifacts": sorted(artifacts),
            "llm_usage": get_usage_summary(),
            "success": bool(final_state.get("report_path") and final_state.get("dashboard_path")),
        }
    )
    code_history = final_state.get("code_history", [])
    failed_attempts = [entry for entry in code_history if not entry.get("success", False)]
    backends = sorted({entry.get("backend", "unknown") for entry in code_history})
    latest_review = (final_state.get("reviews") or [""])[-1]
    run_metadata["quality_metrics"] = {
        "code_attempts": len(code_history),
        "failed_code_attempts": len(failed_attempts),
        "runtime_error_rate": (
            round(len(failed_attempts) / len(code_history), 4)
            if code_history else 0.0
        ),
        "execution_backends": backends,
        "total_code_execution_seconds": round(
            sum(entry.get("duration_seconds", 0.0) for entry in code_history),
            3,
        ),
        "high_severity_findings": latest_review.count("[HIGH]"),
        "agent_counts": {
            agent: run_metadata["agents_called"].count(agent)
            for agent in sorted(set(run_metadata["agents_called"]))
        },
        "artifact_count": len(artifacts),
    }
    metadata_path = os.path.join(_run_dir, "run.json")
    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
    console.print(f"[dim]Metadata: {metadata_path}[/dim]")
    console.print(f"[dim]Checkpoint: {_checkpoint_path}[/dim]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])
    else:
        console.print("[bold]DS Agent System[/bold] — describe a data science problem.\n")
        problem = console.input("[bold blue]Problem:[/bold blue] ").strip()
        if not problem:
            sys.exit(0)

    run(problem)
