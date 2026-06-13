from typing import Literal

from pydantic import BaseModel, Field

AgentName = Literal[
    "engineer",
    "feature_engineer",
    "analyst",
    "scientist",
    "reviewer",
    "reporter",
    "dashboarder",
]


class ManagerDecision(BaseModel):
    reasoning: str = ""
    status: Literal["continuing", "done"]
    agent: AgentName | None = None
    task: str | None = None
    summary: str | None = None


class ReviewIssue(BaseModel):
    severity: Literal["HIGH", "MED", "LOW"]
    title: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)


class ReviewResult(BaseModel):
    issues: list[ReviewIssue] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    @property
    def has_high_issues(self) -> bool:
        return any(issue.severity == "HIGH" for issue in self.issues)

    def to_markdown(self) -> str:
        issue_lines = []
        for issue in self.issues:
            issue_lines.append(
                f"- [{issue.severity}] **{issue.title}** — {issue.evidence} "
                f"Fix: {issue.recommendation}"
            )
        if not issue_lines:
            issue_lines.append("- No issues found.")

        strength_lines = [f"- {item}" for item in self.strengths] or ["- None reported."]
        recommendation_lines = (
            [f"- {item}" for item in self.recommendations]
            or ["- No additional recommendations."]
        )
        return "\n".join(
            [
                "## Review",
                "",
                "### Issues Found",
                *issue_lines,
                "",
                "### What Looks Good",
                *strength_lines,
                "",
                "### Recommendations",
                *recommendation_lines,
            ]
        )
