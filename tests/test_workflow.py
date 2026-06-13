import os
import sqlite3
import tempfile
import unittest

from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from agents.manager import (
    MAX_ITERATIONS,
    _latest_review_has_high_issue,
    _needs_review_correction,
)
from graph.workflow import build_graph
from tools.code_runner import run_code
from tools.contracts import ManagerDecision, ReviewResult
from tools.structured import parse_model


class WorkflowTests(unittest.TestCase):
    def test_graph_compiles_with_expected_nodes(self):
        graph = build_graph()
        self.assertEqual(
            set(graph.nodes),
            {
                "__start__",
                "manager",
                "engineer",
                "feature_engineer",
                "analyst",
                "scientist",
                "reviewer",
                "reporter",
                "dashboarder",
            },
        )

    def test_graph_compiles_with_sqlite_checkpoint(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as file:
            connection = sqlite3.connect(file.name, check_same_thread=False)
            try:
                graph = build_graph(checkpointer=SqliteSaver(connection))
                self.assertIn("manager", graph.nodes)
            finally:
                connection.close()

    def test_iteration_cap_allows_full_pipeline_with_retries(self):
        self.assertGreaterEqual(MAX_ITERATIONS, 20)

    def test_high_review_detection(self):
        state = {"reviews": ["## Review\n- [HIGH] Leakage found"]}
        self.assertTrue(_latest_review_has_high_issue(state))

    def test_review_correction_is_limited(self):
        state = {
            "reviews": ["## Review\n- [HIGH] Leakage found"],
            "reviewer_corrections": 0,
        }
        self.assertTrue(_needs_review_correction(state))

        state["reviewer_corrections"] = 1
        self.assertFalse(_needs_review_correction(state))

    def test_manager_contract_rejects_unknown_agent(self):
        with self.assertRaises(ValidationError):
            ManagerDecision.model_validate(
                {
                    "status": "continuing",
                    "agent": "mystery_agent",
                    "task": "Do work",
                }
            )

    def test_structured_review_renders_high_issue(self):
        result = parse_model(
            """
            ```json
            {
              "issues": [{
                "severity": "HIGH",
                "title": "Leakage",
                "evidence": "Scaler fit before CV.",
                "recommendation": "Move scaling into Pipeline."
              }],
              "strengths": ["Seeded split."],
              "recommendations": ["Re-run CV."]
            }
            ```
            """,
            ReviewResult,
        )
        self.assertTrue(result.has_high_issues)
        self.assertIn("[HIGH]", result.to_markdown())

    def test_local_execution_is_labeled_unsafe(self):
        previous_backend = os.environ.get("DS_EXECUTION_BACKEND")
        previous_outputs = os.environ.get("DS_OUTPUTS_DIR")
        with tempfile.TemporaryDirectory() as directory:
            os.environ["DS_EXECUTION_BACKEND"] = "local"
            os.environ["DS_OUTPUTS_DIR"] = directory
            try:
                result = run_code("print('ok')")
            finally:
                if previous_backend is None:
                    os.environ.pop("DS_EXECUTION_BACKEND", None)
                else:
                    os.environ["DS_EXECUTION_BACKEND"] = previous_backend
                if previous_outputs is None:
                    os.environ.pop("DS_OUTPUTS_DIR", None)
                else:
                    os.environ["DS_OUTPUTS_DIR"] = previous_outputs

        self.assertTrue(result["success"])
        self.assertEqual(result["backend"], "local")
        self.assertIn("without container isolation", result["output"])
        self.assertEqual(result["error"], "")


if __name__ == "__main__":
    unittest.main()
