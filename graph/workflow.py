from langgraph.graph import END, START, StateGraph

from agents.analyst import analyst_node
from agents.dashboarder import dashboarder_node
from agents.engineer import engineer_node
from agents.feature_engineer import feature_engineer_node
from agents.manager import manager_node
from agents.reporter import reporter_node
from agents.reviewer import reviewer_node
from agents.scientist import scientist_node
from graph.state import AgentState


def route(state: AgentState) -> str:
    return state["next"]


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("manager", manager_node)
    g.add_node("engineer", engineer_node)
    g.add_node("feature_engineer", feature_engineer_node)
    g.add_node("analyst", analyst_node)
    g.add_node("scientist", scientist_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("reporter", reporter_node)
    g.add_node("dashboarder", dashboarder_node)

    g.add_edge(START, "manager")
    g.add_conditional_edges(
        "manager",
        route,
        {
            "engineer":         "engineer",
            "feature_engineer": "feature_engineer",
            "analyst":          "analyst",
            "scientist":        "scientist",
            "reviewer":         "reviewer",
            "reporter":         "reporter",
            "dashboarder":      "dashboarder",
            "done":             END,
        },
    )

    # All worker agents loop back to manager except final artifact builders.
    g.add_edge("engineer",         "manager")
    g.add_edge("feature_engineer", "manager")
    g.add_edge("analyst",          "manager")
    g.add_edge("scientist",        "manager")
    g.add_edge("reviewer",         "manager")
    g.add_edge("reporter",         "dashboarder")
    g.add_edge("dashboarder",      END)

    return g.compile(checkpointer=checkpointer)
