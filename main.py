"""Main entry point for the multi-agent system."""

import traceback

from rich.panel import Panel

from commands import CommandHandler, UsageTracker
from config import Config
from graph import create_workflow  # type: ignore[assignment]
from state import AgentState
from tools.ui_utils import (
    console,
    display_section_header,
    generate_filename_from_task,
    get_task_description,
    show_agent_complete,
    show_agent_start,
    show_code_diff,
    show_code_preview,
    show_iteration_info,
    show_qa_result,
    show_requirements_summary,
    show_workflow_summary,
)

# Global usage tracker
usage_tracker = UsageTracker()


def run_workflow(task: str, tracker: UsageTracker, auto_apply: bool = False) -> tuple[AgentState | None, bool]:
    """
    Run the workflow for a given task.

    Args:
        task: Task description

    Returns:
        Final state or None if failed
    """
    # Initialize state
    initial_state: AgentState = {
        "task_description": task,
        "requirements": None,
        "acceptance_criteria": None,
        "code": None,
        "implementation_notes": None,
        "test_results": None,
        "issues_found": None,
        "qa_status": None,
        "iteration": 0,
        "max_iterations": Config.MAX_ITERATIONS,
        "messages": [],
    }

    # Create and run the workflow
    console.print("\n🚀 워크플로우를 시작합니다...\n", style="bold blue")
    workflow = create_workflow()  # type: ignore[assignment]

    try:
        # Run the workflow with streaming
        final_state: AgentState | None = None
        previous_node: str | None = None

        try:
            stream_result = workflow.stream(initial_state)  # type: ignore[attr-defined]
            for chunk in stream_result:  # type: ignore[assignment]
                # chunk is a dict with node name as key
                # Convert chunk to dict if needed
                if isinstance(chunk, dict):
                    chunk_dict: dict[str, AgentState] = chunk  # type: ignore[assignment]
                else:
                    # Handle other iterable types
                    chunk_dict = {str(k): v for k, v in chunk.items()}  # type: ignore[assignment, arg-type]
                for node_name, node_state in chunk_dict.items():
                    # Show agent start notification and track usage
                    if node_name != previous_node:
                        if node_name == "pm":
                            show_agent_start("PM", "요구사항 분석 중...")
                            tracker.record_agent_call("pm")
                        elif node_name == "dev":
                            iteration = node_state.get("iteration", 0)
                            if iteration > 0:
                                show_iteration_info(iteration, Config.MAX_ITERATIONS)
                                show_agent_start("Dev", "코드 수정 중...")
                            else:
                                show_agent_start("Dev", "코드 작성 중...")
                            tracker.record_agent_call("dev")
                        elif node_name == "qa":
                            show_agent_start("QA", "코드 테스트 중...")
                            tracker.record_agent_call("qa")

                        previous_node = node_name

                    # Update final state
                    final_state = node_state

                    # Show completion and results
                    if node_name == "pm" and node_state.get("requirements"):
                        show_agent_complete("PM", "요구사항 분석 완료")
                        requirements = node_state.get("requirements")
                        if requirements:
                            show_requirements_summary(requirements)

                    elif node_name == "dev" and node_state.get("code"):
                        show_agent_complete("Dev", "코드 생성 완료")
                        code = node_state.get("code")
                        if code:
                            show_code_preview(code)

                    elif node_name == "qa" and node_state.get("qa_status"):
                        qa_status = node_state.get("qa_status")
                        if qa_status:
                            issues = node_state.get("issues_found", [])
                            show_agent_complete("QA", "테스트 완료")
                            show_qa_result(qa_status, issues or [])
        except KeyboardInterrupt:
            # KeyboardInterrupt during workflow execution
            console.print("\n\n⚠️  [bold yellow]워크플로우 실행 중 사용자에 의해 중단되었습니다.[/bold yellow]")
            raise  # Re-raise to be caught by outer try-except

        if final_state is None:
            console.print("\n❌ 워크플로우가 완료되지 않았습니다.", style="bold red")
            tracker.record_workflow_complete(success=False)
            return None, auto_apply

        # Record successful workflow
        qa_status = final_state.get("qa_status", "unknown")
        tracker.record_workflow_complete(success=(qa_status == "pass"))

        # Display code preview
        code = final_state.get("code", "N/A")
        if isinstance(code, str) and code != "N/A":
            show_code_preview(code)
        else:
            console.print(
                Panel(
                    "코드를 생성하지 못했습니다.",
                    title="💻 FINAL CODE",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )

        # Save code to file if successful
        code_to_save = final_state.get("code")
        new_auto_apply = auto_apply
        saved_filename: str | None = None

        if code_to_save:
            print()
            try:
                # Generate filename from task description
                task_description = final_state.get("task_description", "")
                filename = generate_filename_from_task(task_description)

                if auto_apply:
                    # Auto-apply mode: save with generated filename
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(code_to_save)
                    saved_filename = filename
                else:
                    # Show diff and get user choice
                    action, should_auto_apply = show_code_diff(code_to_save, filename)

                    if action == "skip":
                        pass  # Don't save, don't show filename in summary
                    else:
                        # Save the file
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(code_to_save)
                        saved_filename = filename

                        # Update auto_apply flag if user chose "apply_all"
                        if should_auto_apply:
                            new_auto_apply = True

            except KeyboardInterrupt:
                console.print("\n\n⚠️  [bold yellow]사용자에 의해 중단되었습니다.[/bold yellow]")
                raise

        # Show workflow summary
        show_workflow_summary(final_state, saved_filename)

        return final_state, new_auto_apply

    except KeyboardInterrupt:
        console.print("\n\n⚠️  사용자에 의해 중단되었습니다.", style="bold yellow")
        return None, auto_apply
    except Exception as e:
        console.print(f"\n❌ Error running workflow: {e}", style="bold red")

        traceback.print_exc()
        return None, auto_apply


def main():
    """Run the multi-agent system in a loop."""
    display_section_header("Multi-Agent System: PM -> Dev -> QA")

    # Initialize command handler
    cmd_handler = CommandHandler(usage_tracker)

    # Auto-apply flag for the session
    auto_apply = False

    console.print("\n💡 [dim]팁: 'exit'로 종료 | 슬래시 명령어는 /help 참고[/dim]\n")

    while True:
        try:
            # Get task from user with better input handling
            task = get_task_description()

            # Skip empty input and continue to next iteration
            if not task.strip():
                continue

            # Check for exit commands
            if task.lower() in ["exit"]:
                console.print("\n👋 프로그램을 종료합니다. 감사합니다!", style="bold cyan")
                break

            # Handle slash commands
            if cmd_handler.handle(task):
                # Command was handled, continue to next iteration
                continue

            # Run the workflow
            _, auto_apply = run_workflow(task, usage_tracker, auto_apply)

            # Clear screen for next task
            print("\n" * 2)
            console.print("─" * 60, style="dim")
            print()

        except KeyboardInterrupt:
            console.print("\n\n👋 프로그램을 종료합니다. 감사합니다!", style="bold cyan")
            break
        except Exception as e:
            console.print(f"\n❌ 예상치 못한 오류: {e}", style="bold red")
            console.print("[dim]다음 작업을 계속합니다...[/dim]\n")


if __name__ == "__main__":
    main()
