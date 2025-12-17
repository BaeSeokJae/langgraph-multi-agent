"""QA (Quality Assurance) Agent - Tests code against acceptance criteria."""

import sys
from io import StringIO

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from config import Config
from state import AgentState
from tools.ui_utils import console, show_agent_thinking, show_agent_thinking_prompt


class QAAgent:
    """QA Agent tests code and verifies it meets acceptance criteria."""

    def __init__(self):
        model = Config.QA_MODEL or Config.LLM_MODEL
        temperature = Config.AGENT_TEMPERATURES.get("qa", Config.DEFAULT_TEMPERATURE)

        api_key: SecretStr | None = None
        if Config.OPENAI_API_KEY:
            api_key = SecretStr(Config.OPENAI_API_KEY)

        # ChatOpenAI constructor accepts these parameters, but pyright doesn't recognize them
        self.llm = ChatOpenAI(  # pyright: ignore[reportCallIssue]
            model=model,  # pyright: ignore[reportArgumentType]
            temperature=temperature,  # pyright: ignore[reportArgumentType]
            api_key=api_key,  # pyright: ignore[reportArgumentType]
        )

    def test_code(self, state: AgentState) -> AgentState:
        """
        Test the code against acceptance criteria.
        Performs both static analysis (via LLM) and attempts to run the code.

        Args:
            state: Current agent state

        Returns:
            Updated state with test results and QA status
        """
        code = state.get("code", "")
        acceptance_criteria = state.get("acceptance_criteria") or []
        requirements = state.get("requirements", "")

        # First, try to execute the code to check for runtime errors
        if not code:
            execution_result = "✗ No code provided"
        else:
            execution_result = self._execute_code(code)

        # Then, use LLM to analyze code quality and criteria compliance
        system_prompt = """You are a QA Engineer who has worked at the world's smartest companies, including Meta, Apple, and AWS.
Your role is to review code and verify it meets all acceptance criteria.

Check for:
1. Correctness against requirements
2. Code quality and best practices
3. Potential bugs or edge cases
4. Whether all acceptance criteria are met

Provide a clear assessment with specific issues if any are found."""

        criteria_list = "\n".join(f"{i + 1}. {criterion}" for i, criterion in enumerate(acceptance_criteria))

        user_prompt = f"""Code to Review:
```python
{code}
```

Acceptance Criteria:
{criteria_list}

Requirements:
{requirements}

Code Execution Result:
{execution_result}

Please evaluate:
1. Does the code meet all acceptance criteria?
2. Are there any bugs or issues?
3. Overall assessment (PASS or FAIL)

Format your response with:
**Status**: PASS or FAIL
**Issues**: List any issues found (or "None" if passing)
**Summary**: Brief summary of findings"""

        # Show what the agent is thinking about
        show_agent_thinking_prompt("QA", system_prompt, user_prompt)

        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Collect response chunks while showing progress spinner
        response_chunks: list[str] = []
        try:
            with show_agent_thinking() as progress:
                progress.add_task(
                    "[cyan]QA Agent가 코드를 테스트하고 분석하고 있습니다...[/cyan]",
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

        # Parse the response
        qa_status, issues = self._parse_qa_response(content)

        # Update state
        state["test_results"] = content
        state["issues_found"] = issues
        state["qa_status"] = qa_status
        state["iteration"] = state.get("iteration", 0) + 1
        # Handle Annotated[list[str], add] type
        current_messages = state.get("messages", [])
        new_messages: list[str] = []
        if current_messages:
            for msg in current_messages:
                if isinstance(msg, str):
                    new_messages.append(msg)
        message_text: str = f"[QA] Testing complete. Status: {qa_status}"
        new_messages.append(message_text)
        state["messages"] = new_messages  # type: ignore[assignment]

        return state

    def _execute_code(self, code: str) -> str:
        """
        Attempt to execute the code and capture any errors.
        This is a simple execution - in production you'd want sandboxing.
        """
        try:
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            # Create a namespace for execution
            namespace = {}

            # Execute the code
            exec(code, namespace)

            # Restore stdout
            sys.stdout = old_stdout
            output = captured_output.getvalue()

            output_text = output if output else "(no output)"
            return f"✓ Code executed successfully.\nOutput: {output_text}"

        except SyntaxError as e:
            sys.stdout = old_stdout
            return f"✗ Syntax Error: {str(e)}"
        except Exception as e:
            sys.stdout = old_stdout
            return f"✗ Runtime Error: {type(e).__name__}: {str(e)}"

    def _parse_qa_response(self, content: str) -> tuple[str, list[str]]:
        """Parse the QA response to extract status and issues."""
        status: str = "fail"
        issues: list[str] = []

        for raw_line in content.split("\n"):
            line = raw_line.strip()

            # Extract status
            if line.lower().startswith("**status**"):
                if "pass" in line.lower():
                    status = "pass"
                else:
                    status = "fail"

            # Extract issues
            if line.lower().startswith("**issues**"):
                # Get the content after the label
                issues_text = line.split(":", 1)[1].strip() if ":" in line else ""
                if issues_text.lower() not in ["none", "none found", ""]:
                    issues.append(issues_text)

            # Also capture bulleted issues
            if (line.startswith("-") or line.startswith("*")) and "issue" in content.lower():
                issue = line.lstrip("-*").strip()
                if issue and issue not in issues:
                    issues.append(issue)

        # If no specific issues were found but status is fail, add a generic one
        if status == "fail" and not issues:
            issues.append("Code did not meet acceptance criteria")

        return status, issues


# Node function for LangGraph
def qa_node(state: AgentState) -> AgentState:
    """QA node for the LangGraph workflow."""
    agent = QAAgent()
    return agent.test_code(state)
