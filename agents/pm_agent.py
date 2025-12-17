"""PM (Product Manager) Agent.

Analyzes requirements and defines acceptance criteria.
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config import Config
from state import AgentState
from tools.ui_utils import console, show_agent_thinking, show_agent_thinking_prompt


class PMAgent:
    """PM Agent analyzes user requests and creates structured requirements."""

    def __init__(self):
        model = Config.PM_MODEL or Config.LLM_MODEL
        temperature = Config.AGENT_TEMPERATURES.get("pm", Config.DEFAULT_TEMPERATURE)

        api_key: SecretStr | None = None
        if Config.OPENAI_API_KEY:
            api_key = SecretStr(Config.OPENAI_API_KEY)

        # ChatOpenAI constructor accepts these parameters, but pyright doesn't recognize them
        self.llm = ChatOpenAI(  # pyright: ignore[reportCallIssue]
            model=model,  # pyright: ignore[reportArgumentType]
            temperature=temperature,  # pyright: ignore[reportArgumentType]
            api_key=api_key,  # pyright: ignore[reportArgumentType]
        )

    def analyze_requirements(self, state: AgentState) -> AgentState:
        """
        Analyze the task description and produce:
        1. Structured requirements
        2. Acceptance criteria for QA

        Args:
            state: Current agent state

        Returns:
            Updated state with requirements and acceptance criteria
        """
        task_description = state["task_description"]

        system_prompt = """You are a Product Manager who has worked at the world's smartest companies, including Meta, Apple, and AWS.
        Your role is to analyze user requests and create clear, structured requirements.

        Given a task description, you should:
        1. Break down the requirements into clear, actionable items
        2. Define acceptance criteria that can be used for testing
        3. Identify edge cases and constraints
        Be specific and thorough."""

        user_prompt = f"""Task Description:
        {task_description}

        Please provide:
        1. **Requirements**: A detailed breakdown of what needs to be built
        2. **Acceptance Criteria**: Specific, testable criteria (as a bulleted list)"""

        # Show what the agent is thinking about
        show_agent_thinking_prompt("PM", system_prompt, user_prompt)

        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Collect response chunks while showing progress spinner
        response_chunks: list[str] = []
        try:
            with show_agent_thinking() as progress:
                progress.add_task(
                    "[cyan]PM Agent가 요구사항을 분석하고 있습니다...[/cyan]",
                    total=None,
                )
                for chunk in self.llm.stream(messages):
                    content = chunk.content  # type: ignore[assignment]
                    # Type guard: content can be str | list[str | dict] | None
                    # We only process string content for streaming
                    if isinstance(content, str) and content:
                        response_chunks.append(content)
        except KeyboardInterrupt:
            console.print("\n\n⚠️  [bold yellow]사용자에 의해 중단되었습니다.[/bold yellow]")
            raise  # Re-raise to propagate to workflow

        # Combine all chunks to get full content
        content = "".join(response_chunks)

        # Parse the response to extract requirements and acceptance criteria
        # Simple parsing - you could make this more sophisticated
        requirements = content
        acceptance_criteria = self._extract_acceptance_criteria(content)

        # Update state
        state["requirements"] = requirements
        state["acceptance_criteria"] = acceptance_criteria
        # Handle Annotated[list[str], add] type
        current_messages = state.get("messages", [])
        new_messages: list[str] = []
        if current_messages:
            for msg in current_messages:
                if isinstance(msg, str):
                    new_messages.append(msg)  # pyright: ignore[reportArgumentType]
        message_text: str = f"[PM] Requirements analyzed:\n{requirements}"
        new_messages.append(message_text)  # pyright: ignore[reportArgumentType]
        state["messages"] = new_messages  # type: ignore[assignment]

        return state

    def _extract_acceptance_criteria(self, content: str) -> list[str]:
        """Extract acceptance criteria from the PM's response."""
        criteria: list[str] = []
        in_criteria_section = False

        for raw_line in content.split("\n"):
            line = raw_line.strip()

            # Look for acceptance criteria section
            if "acceptance criteria" in line.lower():
                in_criteria_section = True
                continue

            # Stop if we hit another section
            if in_criteria_section and line.startswith("**") and line.endswith("**"):
                break

            # Extract bullet points
            if in_criteria_section and (line.startswith("-") or line.startswith("*")):
                criteria.append(line.lstrip("-*").strip())

        return criteria if criteria else ["Code should execute without errors"]


# Node function for LangGraph
def pm_node(state: AgentState) -> AgentState:
    """PM node for the LangGraph workflow."""
    agent = PMAgent()
    return agent.analyze_requirements(state)
