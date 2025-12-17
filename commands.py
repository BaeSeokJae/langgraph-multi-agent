"""Command handler for slash commands."""

from collections.abc import Callable

from rich.table import Table

from config import Config
from tools.ui_utils import console


class UsageTracker:
    """Track API usage statistics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all usage statistics."""
        self.total_requests: int = 0
        self.pm_requests: int = 0
        self.dev_requests: int = 0
        self.qa_requests: int = 0
        self.total_tokens: int = 0  # Approximate
        self.workflows_completed: int = 0
        self.workflows_failed: int = 0

    def record_agent_call(self, agent_name: str) -> None:
        """Record an agent API call."""
        self.total_requests += 1
        if agent_name == "pm":
            self.pm_requests += 1
        elif agent_name == "dev":
            self.dev_requests += 1
        elif agent_name == "qa":
            self.qa_requests += 1

    def record_workflow_complete(self, success: bool = True) -> None:
        """Record a completed workflow."""
        if success:
            self.workflows_completed += 1
        else:
            self.workflows_failed += 1

    def display(self) -> None:
        """Display usage statistics in a table."""
        table = Table(title="📊 사용량 통계", show_header=True, header_style="bold")
        table.add_column("항목", style="cyan")
        table.add_column("값", justify="right", style="green")

        table.add_row("총 API 요청", str(self.total_requests))
        table.add_row("PM Agent 요청", str(self.pm_requests))
        table.add_row("Dev Agent 요청", str(self.dev_requests))
        table.add_row("QA Agent 요청", str(self.qa_requests))
        table.add_row("", "")  # Separator
        table.add_row("완료된 워크플로우", str(self.workflows_completed))
        table.add_row("실패한 워크플로우", str(self.workflows_failed))

        console.print(table)


class CommandHandler:
    """Handle slash commands for configuration and control."""

    def __init__(self, usage_tracker: UsageTracker):
        self.usage_tracker = usage_tracker
        self.commands: dict[str, Callable[[list[str]], None]] = {
            "/help": self.show_help,
            "/usage": self.show_usage,
            "/config": self.show_config,
            "/model": self.set_model,
            "/reset": self.reset_usage,
            "/temp": self.set_temperature,
            "/max-iter": self.set_max_iterations,
        }

    def handle(self, command: str) -> bool:
        """
        Handle a slash command.

        Args:
            command: The command string (e.g., "/help" or "/model pm gpt-4")

        Returns:
            True if command was handled, False if not a command
        """
        if not command.startswith("/"):
            return False

        parts = command.strip().split()
        cmd = parts[0].lower()

        if cmd not in self.commands:
            console.print(
                f"❌ 알 수 없는 명령어: {cmd}\n사용 가능한 명령어를 보려면 /help를 입력하세요.",
                style="red",
            )
            return True

        # Execute command with arguments
        try:
            self.commands[cmd](parts[1:])
        except Exception as e:
            console.print(f"❌ 명령어 실행 오류: {e}", style="red")

        return True

    def show_help(self, args: list[str]) -> None:
        """Show available commands."""
        table = Table(title="📚 사용 가능한 명령어", show_header=True, header_style="bold")
        table.add_column("명령어", style="cyan", width=30)
        table.add_column("설명", style="white")

        table.add_row("/help", "명령어 도움말 표시")
        table.add_row("/usage", "API 사용량 통계 표시")
        table.add_row("/config", "현재 설정 표시")
        table.add_row("/model <agent> <model>", "에이전트별 모델 설정\n예: /model pm gpt-4")
        table.add_row("/temp <agent> <value>", "에이전트별 온도 설정\n예: /temp dev 0.2")
        table.add_row("/max-iter <n>", "최대 반복 횟수 설정\n예: /max-iter 5")
        table.add_row("/reset", "사용량 통계 초기화")

        console.print(table)
        console.print("\n💡 [dim]팁: 에이전트는 'pm', 'dev', 'qa', 'all' 중 하나입니다.[/dim]")

    def show_usage(self, args: list[str]) -> None:
        """Show usage statistics."""
        self.usage_tracker.display()

    def show_config(self, args: list[str]) -> None:
        """Show current configuration."""
        table = Table(title="⚙️  현재 설정", show_header=True, header_style="bold")
        table.add_column("항목", style="cyan", width=25)
        table.add_column("값", style="yellow")

        # Models
        table.add_row("PM Model", Config.PM_MODEL or Config.LLM_MODEL)
        table.add_row("Dev Model", Config.DEV_MODEL or Config.LLM_MODEL)
        table.add_row("QA Model", Config.QA_MODEL or Config.LLM_MODEL)
        table.add_row("", "")  # Separator

        # Temperatures
        table.add_row(
            "PM Temperature",
            str(Config.AGENT_TEMPERATURES.get("pm", Config.DEFAULT_TEMPERATURE)),
        )
        table.add_row(
            "Dev Temperature",
            str(Config.AGENT_TEMPERATURES.get("dev", Config.DEFAULT_TEMPERATURE)),
        )
        table.add_row(
            "QA Temperature",
            str(Config.AGENT_TEMPERATURES.get("qa", Config.DEFAULT_TEMPERATURE)),
        )
        table.add_row("", "")  # Separator

        # Other settings
        table.add_row("최대 반복 횟수", str(Config.MAX_ITERATIONS))

        console.print(table)

    def set_model(self, args: list[str]) -> None:
        """Set model for specific agent."""
        if len(args) < 2:
            console.print(
                "❌ 사용법: /model <agent> <model>\n예: /model pm gpt-4\n에이전트: pm, dev, qa, all",
                style="red",
            )
            return

        agent = args[0].lower()
        model = args[1]

        valid_agents = ["pm", "dev", "qa", "all"]
        if agent not in valid_agents:
            console.print(
                f"❌ 잘못된 에이전트: {agent}\n사용 가능: {', '.join(valid_agents)}",
                style="red",
            )
            return

        if agent == "all":
            Config.PM_MODEL = model
            Config.DEV_MODEL = model
            Config.QA_MODEL = model
            console.print(f"✅ 모든 에이전트의 모델을 {model}로 설정했습니다.", style="green")
        elif agent == "pm":
            Config.PM_MODEL = model
            console.print(f"✅ PM 에이전트의 모델을 {model}로 설정했습니다.", style="green")
        elif agent == "dev":
            Config.DEV_MODEL = model
            console.print(f"✅ Dev 에이전트의 모델을 {model}로 설정했습니다.", style="green")
        elif agent == "qa":
            Config.QA_MODEL = model
            console.print(f"✅ QA 에이전트의 모델을 {model}로 설정했습니다.", style="green")

    def set_temperature(self, args: list[str]) -> None:
        """Set temperature for specific agent."""
        if len(args) < 2:
            console.print(
                "❌ 사용법: /temp <agent> <value>\n예: /temp dev 0.2\n에이전트: pm, dev, qa, all",
                style="red",
            )
            return

        agent = args[0].lower()
        try:
            temp = float(args[1])
            if not 0 <= temp <= 2:
                raise ValueError("온도는 0과 2 사이여야 합니다.")
        except ValueError as e:
            console.print(f"❌ 잘못된 온도 값: {e}", style="red")
            return

        valid_agents = ["pm", "dev", "qa", "all"]
        if agent not in valid_agents:
            console.print(
                f"❌ 잘못된 에이전트: {agent}\n사용 가능: {', '.join(valid_agents)}",
                style="red",
            )
            return

        if agent == "all":
            Config.AGENT_TEMPERATURES["pm"] = temp
            Config.AGENT_TEMPERATURES["dev"] = temp
            Config.AGENT_TEMPERATURES["qa"] = temp
            console.print(f"✅ 모든 에이전트의 온도를 {temp}로 설정했습니다.", style="green")
        else:
            Config.AGENT_TEMPERATURES[agent] = temp
            console.print(
                f"✅ {agent.upper()} 에이전트의 온도를 {temp}로 설정했습니다.",
                style="green",
            )

    def set_max_iterations(self, args: list[str]) -> None:
        """Set maximum iterations for workflow."""
        if len(args) < 1:
            console.print("❌ 사용법: /max-iter <숫자>\n예: /max-iter 5", style="red")
            return

        try:
            max_iter = int(args[0])
            if max_iter < 1:
                raise ValueError("Must be at least 1")
            Config.MAX_ITERATIONS = max_iter
            console.print(f"✅ 최대 반복 횟수를 {max_iter}로 설정했습니다.", style="green")
        except ValueError as e:
            console.print(f"❌ 잘못된 값: {e}", style="red")

    def reset_usage(self, _: list[str]) -> None:
        """Reset usage statistics."""
        self.usage_tracker.reset()
        console.print("✅ 사용량 통계를 초기화했습니다.", style="green")
