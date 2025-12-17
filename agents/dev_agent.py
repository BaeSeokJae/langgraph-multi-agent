"""Dev (Developer) Agent - Writes code based on requirements."""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config import Config
from state import AgentState
from tools.ui_utils import console, show_agent_thinking, show_agent_thinking_prompt


class DevAgent:
    """Developer Agent writes code based on PM requirements."""

    def __init__(self):
        model = Config.DEV_MODEL or Config.LLM_MODEL
        temperature = Config.AGENT_TEMPERATURES.get("dev", Config.DEFAULT_TEMPERATURE)

        api_key: SecretStr | None = None
        if Config.OPENAI_API_KEY:
            api_key = SecretStr(Config.OPENAI_API_KEY)

        # ChatOpenAI constructor accepts these parameters, but pyright doesn't recognize them
        self.llm = ChatOpenAI(  # pyright: ignore[reportCallIssue]
            model=model,  # pyright: ignore[reportArgumentType]
            temperature=temperature,  # pyright: ignore[reportArgumentType]
            api_key=api_key,  # pyright: ignore[reportArgumentType]
        )

    def write_code(self, state: AgentState) -> AgentState:
        """
        Write code based on requirements.
        If there are QA issues from a previous iteration, fix them.

        Args:
            state: Current agent state

        Returns:
            Updated state with generated code
        """
        requirements = state.get("requirements", "")
        issues_found = state.get("issues_found") or []
        previous_code = state.get("code", "")

        # Determine if this is a fix iteration or initial development
        is_fix_iteration = len(issues_found) > 0 and previous_code

        if is_fix_iteration:
            system_prompt = (
                "You are a Senior developer who has worked at the world's "
                "smartest companies, including Meta, Apple, and AWS.\n"
                "QA has identified an issue in code you wrote during a "
                "previous iteration.\n"
                "Fix the issue while preserving the code's functionality."
            )

            user_prompt = f"""Previous Code:
            ```python
            {previous_code}
            ```

            Issues Found by QA:
            {chr(10).join(f"- {issue}" for issue in issues_found)}

            Requirements:
            {requirements}

            Please provide the FIXED code that addresses all the issues."""

        else:
            system_prompt = """You are a Senior developer who has worked at the world's smartest companies, including Meta, Apple, and AWS.
            Your role is to write clean, efficient, well-documented code based on requirements.

            Follow best practices
            - Write clear, readable cod
            - Include docstrings and comment
            - Handle edge case
            - Use proper error handling"""

            user_prompt = f"""Requirements:\n{requirements}

            Please write Python czode that implements these requirements.
            Provide ONLY the code, properly formatted."""

        # Show what the agent is thinking about
        show_agent_thinking_prompt("Dev", system_prompt, user_prompt)

        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Collect response chunks while showing progress spinner
        action_text = "수정하고 있습니다" if is_fix_iteration else "작성하고 있습니다"
        response_chunks: list[str] = []
        try:
            with show_agent_thinking() as progress:
                progress.add_task(
                    f"[cyan]Dev Agent가 코드를 {action_text}...[/cyan]",
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
        full_content = "".join(response_chunks)
        code = self._extract_code(full_content)

        # Update state
        state["code"] = code
        action = "fixed" if is_fix_iteration else "created"
        state["implementation_notes"] = f"Implementation {action} based on requirements"
        # Handle Annotated[list[str], add] type
        current_messages = state.get("messages", [])
        new_messages: list[str] = []
        if current_messages:
            for msg in current_messages:
                if isinstance(msg, str):
                    new_messages.append(msg)
        message_text: str = f"[Dev] Code {'fixed' if is_fix_iteration else 'written'}:\n{code[:200]}..."
        new_messages.append(message_text)
        state["messages"] = new_messages  # type: ignore[assignment]

        return state

    def _extract_code(self, content: str) -> str:
        """Extract code from the response, removing markdown code blocks if present."""
        # Remove markdown code block markers
        lines = content.split("\n")
        code_lines: list[str] = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block or (not content.startswith("```")):
                code_lines.append(line)

        return "\n".join(code_lines).strip()


# Node function for LangGraph
def dev_node(state: AgentState) -> AgentState:
    """Dev node for the LangGraph workflow."""
    agent = DevAgent()
    return agent.write_code(state)
