"""LangGraph workflow definition for the multi-agent system."""

from typing import Any

from langgraph.graph import END, StateGraph

from agents.dev_agent import dev_node
from agents.pm_agent import pm_node
from agents.qa_agent import qa_node
from state import AgentState


def create_workflow() -> Any:  # type: ignore[no-any-return]
    """
    Create the multi-agent workflow graph.

    Flow:
    1. PM analyzes requirements
    2. Dev writes code
    3. QA tests code
    4. If QA fails and iterations < max: go back to Dev
    5. If QA passes or max iterations reached: END

    Returns:
        Compiled StateGraph workflow
    """
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes for each agent
    workflow.add_node("pm", pm_node)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("dev", dev_node)  # pyright: ignore[reportUnknownMemberType]
    workflow.add_node("qa", qa_node)  # pyright: ignore[reportUnknownMemberType]

    # Define the flow
    # Start -> PM
    workflow.set_entry_point("pm")

    # PM -> Dev (always)
    workflow.add_edge("pm", "dev")

    # Dev -> QA (always)
    workflow.add_edge("dev", "qa")

    # QA -> conditional routing
    workflow.add_conditional_edges(
        "qa",
        should_continue,
        {
            "dev": "dev",  # If failed and can retry, go back to dev
            "end": END,  # If passed or max iterations, end
        },
    )

    # Compile the graph
    return workflow.compile()  # type: ignore[no-any-return]


def should_continue(state: AgentState) -> str:
    """
    Determine if we should continue iterating or end the workflow.

    Logic:
    - If QA passed: end
    - If max iterations reached: end
    - Otherwise: continue (go back to dev)

    Args:
        state: Current agent state

    Returns:
        "end" or "dev" to indicate next step
    """
    qa_status = state.get("qa_status", "fail")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    # If QA passed, we're done
    if qa_status == "pass":
        return "end"

    # If we've hit max iterations, stop even if failing
    if iteration >= max_iterations:
        print(f"\n⚠️  Max iterations ({max_iterations}) reached. Stopping.")
        return "end"

    # Otherwise, go back to dev for fixes
    print(f"\n🔄 QA failed. Going back to Dev (iteration {iteration}/{max_iterations})")
    return "dev"
