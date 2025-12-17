"""UI utilities for better terminal interaction."""

import difflib
import re
from pathlib import Path
from time import sleep
from types import TracebackType
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from prompt_toolkit import Application, prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from pydantic import SecretStr
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.text import Text

from config import Config

console = Console()


def get_multiline_input(
    message: str = "Enter your input",
    placeholder: str = "",
    enable_history: bool = True,
) -> str:
    """
    Get multiline input from user with better Korean character support.

    Features:
    - Proper Korean character handling (backspace works correctly)
    - Multiline support (Alt+Enter or Esc+Enter for new line)
    - History support
    - Syntax highlighting
    - Auto-completion ready

    Args:
        message: Prompt message to display
        placeholder: Placeholder text
        enable_history: Enable command history

    Returns:
        User input string
    """
    # Custom style
    style = Style.from_dict(
        {
            "prompt": "#00aa00 bold",
            "message": "#ffffff",
            "placeholder": "#888888 italic",
        }
    )

    # Key bindings for better UX
    kb = KeyBindings()

    @kb.add("escape", "enter")
    def _(event: KeyPressEvent) -> None:
        """Insert newline with Esc+Enter."""
        event.current_buffer.insert_text("\n")

    # Prompt message with formatting
    prompt_message = HTML(f"<prompt>{message}</prompt>\n<message>&gt; </message>")

    try:
        result = prompt(
            prompt_message,
            multiline=False,  # Single line by default
            style=style,
            key_bindings=kb,
            placeholder=placeholder if placeholder else None,
            enable_history_search=enable_history,
        )
        return result.strip()

    except EOFError:
        return ""
    except KeyboardInterrupt:
        # Propagate KeyboardInterrupt to allow proper exit handling
        raise


def get_task_description() -> str:
    """
    Get task description from user with enhanced input.

    Returns:
        Task description string
    """
    return get_multiline_input(
        message="작업 설명을 입력하세요 (Enter로 제출)",
        placeholder="예: 피보나치 수열을 계산하는 함수를 만들어줘",
        enable_history=True,
    )


def get_confirmation(message: str, default: bool = False) -> bool:
    """
    Get yes/no confirmation from user.

    Args:
        message: Question to ask
        default: Default value if user just presses Enter

    Returns:
        True for yes, False for no
    """
    default_str = "Y/n" if default else "y/N"
    prompt_message = HTML(f"<prompt>{message}</prompt> <message>[{default_str}]</message>\n&gt; ")

    style = Style.from_dict(
        {
            "prompt": "#00aa00",
            "message": "#888888",
        }
    )

    try:
        result = prompt(prompt_message, style=style)
        result = result.strip().lower()

        if not result:
            return default

        return result in ["y", "yes", "예", "ㅇ"]

    except EOFError:
        return default
    except KeyboardInterrupt:
        # Propagate KeyboardInterrupt to allow immediate exit
        raise


def generate_filename_from_task(task_description: str) -> str:
    """
    Generate filename from task description by analyzing the task content.

    Uses LLM to intelligently extract the core concept and generate
    an appropriate Python filename.

    Args:
        task_description: Task description text

    Returns:
        Generated filename with .py extension
    """
    try:
        # Use LLM to generate appropriate filename
        model = Config.LLM_MODEL
        temperature = 0.3  # Lower temperature for more consistent naming

        api_key: SecretStr | None = None
        if Config.OPENAI_API_KEY:
            api_key = SecretStr(Config.OPENAI_API_KEY)

        llm = ChatOpenAI(  # pyright: ignore[reportCallIssue]
            model=model,  # pyright: ignore[reportArgumentType]
            temperature=temperature,  # pyright: ignore[reportArgumentType]
            api_key=api_key,  # pyright: ignore[reportArgumentType]
        )

        system_prompt = """You are a helpful assistant that generates appropriate Python filenames based on task descriptions.

Your task is to analyze the task description and generate a concise, descriptive Python filename.

Rules:
1. Use snake_case (lowercase with underscores)
2. Be concise but descriptive (2-4 words max)
3. Use English only (translate Korean/other languages to English)
4. Focus on the main concept/functionality
5. Do NOT include .py extension in your response
6. Do NOT include any explanation, just the filename

Examples:
- "피보나치 수열을 계산하는 함수를 만들어줘" -> "fibonacci"
- "계산기 클래스를 만들어줘" -> "calculator"
- "사용자 인증 시스템 구현" -> "authentication"
- "Create a web scraper" -> "web_scraper"
- "데이터베이스 연결 관리" -> "database_connection"
"""

        user_prompt = f"Task description: {task_description}\n\nGenerate an appropriate Python filename:"

        messages: list[SystemMessage | HumanMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)  # type: ignore[assignment]
        filename_str: str
        if hasattr(response, "content"):
            content = response.content  # type: ignore[attr-defined]
            filename_str = str(content).strip() if not isinstance(content, str) else content.strip()
        else:
            filename_str = str(response).strip()

        # Clean up the filename
        # Remove .py if present
        filename_str = filename_str.replace(".py", "").strip()

        # Remove any quotes
        filename_str = filename_str.strip('"').strip("'").strip()

        # Ensure it's valid (only alphanumeric, underscores, hyphens)
        filename_str = re.sub(r"[^\w-]", "", filename_str)

        # Replace spaces and hyphens with underscores
        filename_str = filename_str.replace(" ", "_").replace("-", "_")

        # Remove multiple underscores
        filename_str = re.sub(r"_+", "_", filename_str)

        # Remove leading/trailing underscores
        filename_str = filename_str.strip("_")

        # If empty or too short, use fallback
        if not filename_str or len(filename_str) < 2:
            filename_str = _fallback_filename_generation(task_description).replace(".py", "")

        # Add .py extension
        if not filename_str.endswith(".py"):
            filename_str += ".py"

        return filename_str

    except Exception as e:
        # Fallback to simple generation if LLM fails
        console.print(f"[dim]파일명 생성 중 오류 발생, 기본 방식 사용: {e}[/dim]")
        return _fallback_filename_generation(task_description)


def _fallback_filename_generation(task_description: str) -> str:
    """
    Fallback filename generation when LLM is unavailable.

    Args:
        task_description: Task description text

    Returns:
        Generated filename with .py extension
    """
    # Simple keyword mapping for common Korean terms
    keyword_map = {
        "인증": "authentication",
        "사용자": "user",
        "데이터베이스": "database",
        "웹": "web",
        "스크래퍼": "scraper",
        "파서": "parser",
        "API": "api",
        "서버": "server",
        "클라이언트": "client",
        "함수": "function",
        "클래스": "class",
        "유틸리티": "utility",
        "도구": "tool",
    }

    task = task_description.strip().lower()

    # Try to find keywords
    found_keywords: list[str] = []
    for korean, english in keyword_map.items():
        if korean in task:
            found_keywords.append(english)

    filename: str
    if found_keywords:
        filename = "_".join(found_keywords[:3])  # Max 3 keywords
    else:
        # Extract English words if present
        english_words = re.findall(r"\b[a-z]{3,}\b", task)
        if english_words:
            filename = "_".join(english_words[:3])
        else:
            # Last resort: use first few characters
            task_clean = re.sub(r"[^\w\s가-힣]", "", task)[:30]
            task_clean = re.sub(r"\s+", "_", task_clean)
            filename = task_clean if task_clean else "output"

    # Clean up
    filename = re.sub(r"[^\w-]", "", filename)
    filename = filename.replace("-", "_")
    filename = re.sub(r"_+", "_", filename)
    filename = filename.strip("_")

    if not filename or len(filename) < 2:
        filename = "output"

    return filename + ".py"


def get_filename_input(default: str = "output.py") -> str:
    """
    Get filename from user.

    Args:
        default: Default filename

    Returns:
        Filename string
    """
    prompt_message = HTML(f"<prompt>파일명을 입력하세요</prompt> <message>[기본값: {default}]</message>\n&gt; ")

    style = Style.from_dict(
        {
            "prompt": "#00aa00",
            "message": "#888888",
        }
    )

    try:
        result = prompt(prompt_message, style=style, default="")
        return result.strip() if result.strip() else default

    except EOFError:
        return default
    except KeyboardInterrupt:
        # Propagate KeyboardInterrupt to allow immediate exit
        raise


def display_section_header(title: str, width: int = 60):
    """
    Display a formatted section header.

    Args:
        title: Section title
        width: Width of the header line
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def display_subsection(title: str, width: int = 60):
    """
    Display a formatted subsection header.

    Args:
        title: Subsection title
        width: Width of the header line
    """
    print("\n" + title)
    print("-" * width)


# Rich-based progress display


def show_agent_start(agent_name: str, description: str):
    """
    Display agent start notification.

    Args:
        agent_name: Name of the agent
        description: What the agent is doing
    """
    emoji_map = {
        "PM": "📋",
        "Dev": "💻",
        "QA": "🧪",
    }

    emoji = emoji_map.get(agent_name, "🤖")
    text = Text()
    text.append(f"{emoji} {agent_name} Agent", style="bold cyan")
    text.append(" | ", style="dim")
    text.append(description, style="yellow")

    console.print(Panel(text, border_style="cyan", padding=(0, 2)))


def show_agent_complete(agent_name: str, summary: str = ""):
    """
    Display agent completion notification.

    Args:
        agent_name: Name of the agent
        summary: Brief summary of what was accomplished
    """
    emoji_map = {
        "PM": "✅",
        "Dev": "✅",
        "QA": "✅",
    }

    emoji = emoji_map.get(agent_name, "✅")
    text = Text()
    text.append(f"{emoji} {agent_name} 완료", style="bold green")

    if summary:
        text.append(" | ", style="dim")
        text.append(summary, style="white")

    console.print(text)


def show_agent_thinking():
    """
    Show a spinner while agent is thinking.

    Returns:
        Progress context manager
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


def show_code_preview(code: str, max_lines: int = 10, show_full: bool = False):
    """
    Display generated code in full.

    Args:
        code: Code to display
        max_lines: Deprecated, kept for compatibility
        show_full: Deprecated, kept for compatibility
    """
    lines = code.split("\n")
    total_lines = len(lines)

    # Always show full code
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    title = f"💻 생성된 코드 (전체 {total_lines}줄)"
    console.print(Panel(syntax, title=title, border_style="blue"))


def show_requirements_summary(requirements: str, max_chars: int = 300, show_full: bool = False):
    """
    Display requirements in full.

    Args:
        requirements: Requirements text
        max_chars: Deprecated, kept for compatibility
        show_full: Deprecated, kept for compatibility
    """
    total_chars = len(requirements)

    # Always show full requirements
    title = f"📋 요구사항 분석 결과 (전체 {total_chars}자)"
    console.print(
        Panel(
            requirements,
            title=title,
            border_style="green",
            padding=(1, 2),
        )
    )


def show_qa_result(status: str, issues: list[str] | None = None):
    """
    Display QA test results.

    Args:
        status: "pass" or "fail"
        issues: List of issues if failed
    """
    if status == "pass":
        console.print(
            Panel(
                "✅ 모든 테스트를 통과했습니다!",
                title="🧪 QA 결과",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        issue_text = "\n".join(f"• {issue}" for issue in (issues or []))
        console.print(
            Panel(
                f"❌ 테스트 실패\n\n발견된 문제:\n{issue_text}",
                title="🧪 QA 결과",
                border_style="red",
                padding=(1, 2),
            )
        )


def show_iteration_info(current: int, max_iter: int):
    """
    Display iteration information.

    Args:
        current: Current iteration number
        max_iter: Maximum iterations
    """
    console.print(
        f"\n🔄 [yellow]재시도 중[/yellow] (반복: {current}/{max_iter})",
        style="bold",
    )


def _summarize_prompt(system_prompt: str, user_prompt: str, max_length: int = 800) -> str:
    """
    Intelligently summarize a prompt, keeping important parts.

    Args:
        prompt: Full prompt text
        max_length: Maximum length for summary

    Returns:
        Summarized prompt
    """
    if len(system_prompt) + len(user_prompt) <= max_length:
        return system_prompt + "\n\n" + user_prompt

    # Summarize system prompt (keep first sentence and key points)
    if system_prompt:
        system_lines = system_prompt.split("\n")
        if len(system_lines) > 3:
            # Keep first line and last few lines
            summarized_system = "\n".join(system_lines[:2] + ["..."] + system_lines[-2:])
        else:
            summarized_system = system_prompt
    else:
        summarized_system = ""

    # Process user prompt - keep task description, summarize code blocks
    if user_prompt:
        # Check for code blocks
        if "```" in user_prompt:
            lines = user_prompt.split("\n")
            result_lines: list[str] = []
            in_code_block = False
            code_lines: list[str] = []

            for line in lines:
                if line.strip().startswith("```"):
                    if not in_code_block:
                        # Start of code block
                        in_code_block = True
                        code_lines = []
                        result_lines.append(line)
                    else:
                        # End of code block
                        in_code_block = False
                        if len(code_lines) > 20:
                            # Code block was too long, show summary
                            # Keep first 10 lines and last 5 lines
                            summary_lines = code_lines[:10] + ["\n... (코드 일부 생략, 전체 코드는 에이전트가 확인 중) ...\n"] + code_lines[-5:]
                            result_lines.extend(summary_lines)
                        else:
                            result_lines.extend(code_lines)
                        result_lines.append(line)
                        code_lines = []
                elif in_code_block:
                    code_lines.append(line)
                else:
                    result_lines.append(line)

            user_prompt = "\n".join(result_lines)
        else:
            # No code blocks, just truncate if too long
            if len(user_prompt) > max_length - len(summarized_system):
                user_prompt = user_prompt[: max_length - len(summarized_system) - 50] + "\n... (일부 생략)"

    # Combine
    if summarized_system:
        result = f"{summarized_system}\n\n{user_prompt}"
    else:
        result = user_prompt

    # Final check
    if len(result) > max_length:
        result = result[:max_length] + "\n... (프롬프트 일부 생략)"

    return result


def show_agent_thinking_prompt(agent_name: str, system_prompt: str, user_prompt: str):
    """
    Display the prompt that the agent is using for thinking with animation.

    Args:
        agent_name: Name of the agent
        prompt: The prompt being sent to the LLM
    """
    # Show loading animation
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"[cyan]{agent_name} Agent[/cyan] 프롬프트 분석 중...", total=None)
            sleep(0.2)  # Brief pause for visual effect
    except KeyboardInterrupt:
        console.print("\n\n⚠️  [bold yellow]사용자에 의해 중단되었습니다.[/bold yellow]")
        raise

    # Summarize the prompt intelligently
    summarized_prompt = _summarize_prompt(system_prompt, user_prompt)

    # Calculate full prompt length for comparison
    full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

    # Show summarized prompt
    panel_content = summarized_prompt
    if len(full_prompt) > len(summarized_prompt):
        panel_content += f"\n\n[dim]📝 프롬프트 요약: 전체 {len(full_prompt)}자 중 {len(summarized_prompt)}자 표시 (핵심 내용 위주)[/dim]"

    console.print(
        Panel(
            user_prompt,
            title=f"💭 {agent_name} Agent의 생각 과정 (프롬프트 요약)",
            border_style="dim",
            padding=(1, 2),
        )
    )


def show_agent_thinking_stream(agent_name: str):
    """
    Create a context manager for streaming agent thinking process.

    Args:
        agent_name: Name of the agent

    Returns:
        A context manager that displays streaming text
    """

    class ThinkingStream:
        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            self.content = Text()
            self.live = None

        def __enter__(self):
            self.content = Text()
            panel = Panel(
                self.content,
                title=f"💭 {self.agent_name} Agent의 생각 중...",
                border_style="yellow",
                padding=(1, 2),
            )
            self.live = Live(panel, console=console, refresh_per_second=10)
            self.live.__enter__()
            return self

        def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: TracebackType | None) -> None:
            if self.live:
                self.live.__exit__(exc_type, exc_val, exc_tb)

        def append(self, text: str):
            """Append text to the stream."""
            self.content.append(text)
            if self.live:
                panel = Panel(
                    self.content,
                    title=f"💭 {self.agent_name} Agent의 생각 중...",
                    border_style="yellow",
                    padding=(1, 2),
                )
                self.live.update(panel)

    return ThinkingStream(agent_name)


def show_workflow_summary(final_state: Any, filename: str | None = None) -> None:
    """
    Display a comprehensive summary of the completed workflow.

    Args:
        final_state: Final state from the workflow (AgentState dict-like object)
        filename: Filename where code was saved (if any)
    """
    task_description = final_state.get("task_description", "")
    requirements = final_state.get("requirements", "")
    qa_status = final_state.get("qa_status", "unknown")
    issues_found = final_state.get("issues_found", [])
    iteration = final_state.get("iteration", 0)
    code = final_state.get("code", "")

    # Build summary content
    summary_parts: list[str] = []

    # Task description
    summary_parts.append(f"[bold cyan]📝 작업 내용:[/bold cyan]\n{task_description}\n")

    # Process summary
    summary_parts.append("[bold cyan]🔄 처리 과정:[/bold cyan]")
    summary_parts.append("  1. [green]PM Agent[/green]: 요구사항 분석 및 수용 기준 정의")
    if requirements:
        # Show first 200 chars of requirements as summary
        req_summary = requirements[:200] + ("..." if len(requirements) > 200 else "")
        summary_parts.append(f"     → {req_summary}")
    summary_parts.append("  2. [blue]Dev Agent[/blue]: 코드 작성 및 구현")
    if code:
        code_lines = code.split("\n")
        summary_parts.append(f"     → {len(code_lines)}줄의 코드 생성")
        if iteration > 0:
            summary_parts.append(f"     → {iteration}회 반복 개선 수행")
    summary_parts.append("  3. [yellow]QA Agent[/yellow]: 코드 테스트 및 검증")
    if qa_status == "pass":
        summary_parts.append("     → ✅ 모든 테스트 통과")
    elif qa_status == "fail":
        summary_parts.append(f"     → ❌ 테스트 실패 ({len(issues_found)}개 이슈 발견)")
        if issues_found:
            for issue in issues_found[:3]:  # Show first 3 issues
                summary_parts.append(f"       • {issue}")
            if len(issues_found) > 3:
                summary_parts.append(f"       ... 외 {len(issues_found) - 3}개 이슈")

    # Results
    summary_parts.append("\n[bold cyan]📊 결과:[/bold cyan]")
    if code:
        code_lines = code.split("\n")
        summary_parts.append(f"  • 생성된 코드: {len(code_lines)}줄")
    if filename:
        summary_parts.append(f"  • 저장된 파일: [bold green]{filename}[/bold green]")
    if qa_status == "pass":
        summary_parts.append("  • 상태: [bold green]✅ 성공적으로 완료[/bold green]")
    elif qa_status == "fail":
        summary_parts.append("  • 상태: [bold yellow]⚠️  완료 (일부 이슈 존재)[/bold yellow]")

    summary_text = "\n".join(summary_parts)

    console.print("\n")
    console.print(
        Panel(
            summary_text,
            title="✅ 작업 완료 요약",
            border_style="bold green",
            padding=(1, 2),
        )
    )
    console.print()


def show_code_diff(new_code: str, filename: str) -> tuple[str, bool]:
    """
    Show diff between existing file and new code, and get user's choice.

    Args:
        new_code: New code to compare
        filename: Filename to compare against

    Returns:
        Tuple of (action, auto_apply)
        - action: "apply" (반영하기), "apply_once" (이번 세션에는 반영), "apply_all" (반영 계속 허용), "skip" (skip)
        - auto_apply: Whether to auto-apply for future files in this session
    """
    file_path = Path(filename)
    existing_code = ""

    # Read existing file if it exists
    if file_path.exists():
        try:
            existing_code = file_path.read_text(encoding="utf-8")
        except Exception:
            existing_code = ""

    # Show diff if file exists
    if existing_code:
        console.print("\n")
        console.print(
            Panel(
                f"📄 기존 파일: {filename}",
                border_style="yellow",
                padding=(0, 1),
            )
        )

        # Create diff
        diff_lines = list(
            difflib.unified_diff(
                existing_code.splitlines(keepends=True),
                new_code.splitlines(keepends=True),
                fromfile=f"기존 {filename}",
                tofile=f"새로운 {filename}",
                lineterm="",
            )
        )

        if diff_lines:
            # Show diff with syntax highlighting
            diff_text = "".join(diff_lines)
            syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
            console.print(
                Panel(
                    syntax,
                    title="📊 코드 변경사항 (Diff)",
                    border_style="cyan",
                )
            )
        else:
            console.print(
                Panel(
                    "변경사항이 없습니다.",
                    title="📊 코드 변경사항",
                    border_style="green",
                )
            )
    else:
        # New file
        console.print("\n")
        syntax = Syntax(new_code, "python", theme="monokai", line_numbers=True)
        console.print(
            Panel(
                syntax,
                title=f"📝 새 파일: {filename}",
                border_style="blue",
            )
        )

    # Define options
    options = [
        ("apply", False, "반영하기 (이번 파일만 저장)"),
        ("apply_once", False, "이번 세션에는 반영 (이번 파일만 저장)"),
        ("apply_all", True, "반영 계속 허용 (이번 세션의 모든 파일 자동 저장)"),
        ("skip", False, "Skip (저장하지 않음)"),
    ]

    selected_index = [0]  # Use list to allow modification in nested functions

    def get_formatted_text() -> Any:
        """Get formatted text with current selection highlighted."""
        formatted: list[tuple[str, str]] = []
        formatted.append(("", "선택하세요 (↑↓ 화살표로 이동, Enter로 선택):\n\n"))
        for i, (_, _, description) in enumerate(options):
            if i == selected_index[0]:
                formatted.append(("bold cyan", f"  ▶ {description}\n"))
            else:
                formatted.append(("", f"    {description}\n"))
        return formatted

    # Create key bindings
    kb = KeyBindings()

    @kb.add("up")
    def _(event: KeyPressEvent) -> None:
        """Move selection up."""
        selected_index[0] = max(0, selected_index[0] - 1)
        # Force redraw
        event.app.invalidate()

    @kb.add("down")
    def _(event: KeyPressEvent) -> None:
        """Move selection down."""
        selected_index[0] = min(len(options) - 1, selected_index[0] + 1)
        # Force redraw
        event.app.invalidate()

    @kb.add("enter")
    def _(event: KeyPressEvent) -> None:
        """Confirm selection."""
        event.app.exit(result=selected_index[0])

    @kb.add("c-c")
    def _(event: KeyPressEvent) -> None:
        """Cancel with Ctrl+C."""
        event.app.exit(exception=KeyboardInterrupt())

    try:
        # Create control with formatted text
        control = FormattedTextControl(get_formatted_text, key_bindings=kb, focusable=True)  # type: ignore[arg-type]
        layout = Layout(Window(control))

        # Create and run application
        app: Application[Any] = Application(layout=layout, key_bindings=kb, full_screen=False)
        result_index: int | None = app.run()  # type: ignore[assignment]

        if result_index is None:
            raise KeyboardInterrupt()

        # Return selected option
        action, auto_apply, _ = options[result_index]  # type: ignore[misc]
        return (action, auto_apply)

    except KeyboardInterrupt:
        console.print("\n\n⚠️  [bold yellow]사용자에 의해 중단되었습니다.[/bold yellow]")
        raise
