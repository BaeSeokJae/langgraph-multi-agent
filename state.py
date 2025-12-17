"""State definition for the multi-agent workflow."""

from operator import add
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    """Shared state between all agents in the workflow.

    This state is passed between PM -> Dev -> QA agents.
    """

    # Input
    task_description: str  # User's initial request

    # PM Agent outputs
    requirements: str | None  # Analyzed requirements from PM
    acceptance_criteria: list[str] | None  # List of criteria for QA

    # Dev Agent outputs
    code: str | None  # Generated code
    implementation_notes: str | None  # Notes about the implementation

    # QA Agent outputs
    test_results: str | None  # Test results and findings
    issues_found: list[str] | None  # List of issues found
    qa_status: str | None  # "pass" or "fail"

    # Workflow control
    iteration: int  # Track how many times we've gone through the loop
    max_iterations: int  # Maximum allowed iterations

    # Message history (optional, for debugging/logging)
    messages: Annotated[list[str], add]  # Accumulates messages from agents
