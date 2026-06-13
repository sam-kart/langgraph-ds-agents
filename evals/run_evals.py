#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "evals" / "tasks.json"
RESULTS_DIR = ROOT / "evals" / "results"
RUN_DIR_RE = re.compile(r"Run dir:\s+(.+)")


def load_tasks() -> list[dict]:
    with TASKS_PATH.open() as f:
        return json.load(f)


def parse_run_dir(output: str) -> str:
    for line in output.splitlines():
        match = RUN_DIR_RE.search(line)
        if match:
            return match.group(1).strip()
    return ""


def validate_run(task: dict, run_dir: str) -> dict:
    artifacts_dir = Path(run_dir) / "outputs"
    metadata_path = Path(run_dir) / "run.json"
    artifacts = []
    if artifacts_dir.exists():
        artifacts = sorted(p.name for p in artifacts_dir.iterdir() if p.is_file())

    missing = [
        name for name in task.get("required_artifacts", [])
        if name not in artifacts
    ]
    reports = [name for name in artifacts if name.endswith(".md")]
    if task.get("required_report") and not reports:
        missing.append("report_*.md")
    dashboards = [name for name in artifacts if name.endswith(".html")]
    if task.get("required_dashboard") and "dashboard.html" not in dashboards:
        missing.append("dashboard.html")

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open() as f:
            metadata = json.load(f)

    agents_called = set(metadata.get("agents_called", []))
    missing_agents = [
        agent for agent in task.get("required_agents", [])
        if agent not in agents_called
    ]
    runtime_error_rate = metadata.get("quality_metrics", {}).get("runtime_error_rate", 1.0)
    max_error_rate = task.get("max_runtime_error_rate", 1.0)

    checks = {
        "metadata": metadata_path.exists(),
        "artifacts": not missing,
        "required_agents": not missing_agents,
        "runtime_error_rate": runtime_error_rate <= max_error_rate,
        "completed": metadata.get("success", False),
    }
    score = round(sum(checks.values()) / len(checks), 3)

    return {
        "run_dir": run_dir,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "missing_agents": missing_agents,
        "metadata_found": metadata_path.exists(),
        "runtime_error_rate": runtime_error_rate,
        "max_runtime_error_rate": max_error_rate,
        "checks": checks,
        "score": score,
        "success": all(checks.values()),
        "metadata": metadata,
    }


def run_task(task: dict, timeout: int) -> dict:
    cmd = [sys.executable, "main.py", task["prompt"]]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = completed.stdout + "\n" + completed.stderr
    run_dir = parse_run_dir(output)
    validation = validate_run(task, run_dir) if run_dir else {
        "run_dir": "",
        "artifacts": [],
        "missing_artifacts": task.get("required_artifacts", []),
        "metadata_found": False,
        "success": False,
        "metadata": {},
    }
    return {
        "task_id": task["id"],
        "returncode": completed.returncode,
        "run_dir": run_dir,
        "stdout_tail": output[-4000:],
        "validation": validation,
        "success": completed.returncode == 0 and validation["success"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark prompts for the DS multi-agent system.")
    parser.add_argument("--run", action="store_true", help="Actually call the LLM and run selected evals.")
    parser.add_argument("--task", help="Only run one task id.")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout per eval task in seconds.")
    args = parser.parse_args()

    tasks = load_tasks()
    if args.task:
        tasks = [task for task in tasks if task["id"] == args.task]
        if not tasks:
            raise SystemExit(f"Unknown task id: {args.task}")

    if not args.run:
        print("Eval tasks loaded. Dry run only; use --run to spend API credits.")
        for task in tasks:
            print(f"- {task['id']}: {task['name']}")
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "tasks": [],
    }

    for task in tasks:
        print(f"Running {task['id']}...")
        results["tasks"].append(run_task(task, args.timeout))

    results["ended_at"] = datetime.now().isoformat(timespec="seconds")
    results["success"] = all(task["success"] for task in results["tasks"])

    path = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved eval results: {path}")
    return 0 if results["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
