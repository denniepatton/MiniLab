"""
Console output formatting for MiniLab.

Provides styled terminal output similar to VS Code agent activity display.
"""

import sys
from enum import Enum
from typing import Optional
from datetime import datetime


class Style:
    """ANSI escape codes for terminal styling."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class StatusIcon:
    """Status icons for different operations."""
    SUCCESS = "✓"
    FAILURE = "✗"
    WARNING = "⚠"
    INFO = "ℹ"
    RUNNING = "●"
    PENDING = "○"
    ARROW = "→"
    BULLET = "•"
    SEARCH = "🔍"
    FILE = "📄"
    FOLDER = "📁"
    CODE = "💻"
    BRAIN = "🧠"
    CHAT = "💬"
    SAVE = "💾"
    LOAD = "📂"


class Console:
    """
    Styled console output for MiniLab.
    
    Mimics VS Code agent activity display with:
    - Colored status indicators
    - Tool operation messages
    - Agent activity markers
    - Progress indicators
    """
    
    _enabled = True  # Can disable colors for non-TTY
    _verbose = False
    
    @classmethod
    def enable_colors(cls, enabled: bool = True) -> None:
        """Enable or disable colored output."""
        cls._enabled = enabled
    
    @classmethod
    def set_verbose(cls, verbose: bool = True) -> None:
        """Enable verbose output mode."""
        cls._verbose = verbose
    
    @classmethod
    def _style(cls, text: str, *styles: str) -> str:
        """Apply styles to text if colors are enabled."""
        if not cls._enabled or not sys.stdout.isatty():
            return text
        style_str = "".join(styles)
        return f"{style_str}{text}{Style.RESET}"
    
    # === Status Messages ===
    
    @classmethod
    def success(cls, message: str, detail: Optional[str] = None) -> None:
        """Print a success message."""
        icon = cls._style(StatusIcon.SUCCESS, Style.GREEN, Style.BOLD)
        msg = cls._style(message, Style.GREEN)
        if detail:
            detail_text = cls._style(f"({detail})", Style.DIM)
            print(f"{icon} {msg} {detail_text}")
        else:
            print(f"{icon} {msg}")
    
    @classmethod
    def error(cls, message: str, detail: Optional[str] = None) -> None:
        """Print an error message."""
        icon = cls._style(StatusIcon.FAILURE, Style.RED, Style.BOLD)
        msg = cls._style(message, Style.RED)
        if detail:
            detail_text = cls._style(f"({detail})", Style.DIM)
            print(f"{icon} {msg} {detail_text}")
        else:
            print(f"{icon} {msg}")
    
    @classmethod
    def warning(cls, message: str, detail: Optional[str] = None) -> None:
        """Print a warning message."""
        icon = cls._style(StatusIcon.WARNING, Style.YELLOW)
        msg = cls._style(message, Style.YELLOW)
        if detail:
            detail_text = cls._style(f"({detail})", Style.DIM)
            print(f"{icon} {msg} {detail_text}")
        else:
            print(f"{icon} {msg}")
    
    @classmethod
    def info(cls, message: str, detail: Optional[str] = None) -> None:
        """Print an info message."""
        icon = cls._style(StatusIcon.INFO, Style.BLUE)
        msg = message
        if detail:
            detail_text = cls._style(f"({detail})", Style.DIM)
            print(f"{icon} {msg} {detail_text}")
        else:
            print(f"{icon} {msg}")
    
    # === Tool Operations ===
    
    @classmethod
    def tool_start(cls, tool_name: str, operation: str) -> None:
        """Indicate a tool operation is starting."""
        if cls._verbose:
            icon = cls._style(StatusIcon.RUNNING, Style.CYAN)
            tool = cls._style(tool_name, Style.CYAN, Style.BOLD)
            print(f"{icon} {tool}: {operation}...")
    
    @classmethod
    def tool_success(cls, operation: str, target: Optional[str] = None) -> None:
        """Indicate a tool operation succeeded."""
        icon = cls._style(StatusIcon.SUCCESS, Style.GREEN)
        if target:
            target_styled = cls._style(f"'{target}'", Style.BRIGHT_WHITE)
            print(f"{icon} {operation} {target_styled}")
        else:
            print(f"{icon} {operation}")
    
    @classmethod
    def tool_error(cls, operation: str, error: str) -> None:
        """Indicate a tool operation failed."""
        icon = cls._style(StatusIcon.FAILURE, Style.RED)
        err = cls._style(error, Style.DIM)
        print(f"{icon} {operation}: {err}")
    
    # === File Operations ===
    
    @classmethod
    def file_read(cls, path: str) -> None:
        """Log a file read operation."""
        icon = cls._style(StatusIcon.SUCCESS, Style.GREEN)
        print(f"{icon} Read '{path}'")
    
    @classmethod
    def file_write(cls, path: str) -> None:
        """Log a file write operation."""
        icon = cls._style(StatusIcon.SUCCESS, Style.GREEN)
        print(f"{icon} Created '{path}'")
    
    @classmethod
    def file_edit(cls, path: str) -> None:
        """Log a file edit operation."""
        icon = cls._style(StatusIcon.SUCCESS, Style.GREEN)
        print(f"{icon} Edited '{path}'")
    
    @classmethod
    def file_delete(cls, path: str) -> None:
        """Log a file delete operation."""
        icon = cls._style(StatusIcon.SUCCESS, Style.YELLOW)
        print(f"{icon} Deleted '{path}'")
    
    @classmethod
    def file_list(cls, path: str, count: int) -> None:
        """Log a directory listing."""
        icon = cls._style(StatusIcon.SUCCESS, Style.GREEN)
        print(f"{icon} Listed '{path}' ({count} items)")
    
    # === Search Operations ===
    
    @classmethod
    def search_start(cls, query: str, source: str) -> None:
        """Log search initiation."""
        icon = cls._style(StatusIcon.SEARCH, Style.CYAN)
        print(f"{icon} Searching {source} for: {query[:50]}{'...' if len(query) > 50 else ''}")
    
    @classmethod
    def search_complete(cls, source: str, count: int) -> None:
        """Log search completion."""
        icon = cls._style(StatusIcon.SUCCESS, Style.GREEN)
        print(f"{icon} Found {count} results from {source}")
    
    # === Agent Activity ===
    
    @classmethod
    def agent_start(cls, agent_name: str, task: Optional[str] = None) -> None:
        """Log agent starting work."""
        icon = cls._style(StatusIcon.BRAIN, Style.MAGENTA)
        name = cls._style(agent_name.upper(), Style.MAGENTA, Style.BOLD)
        if task:
            task_preview = task[:60] + "..." if len(task) > 60 else task
            print(f"\n{icon} {name}: {task_preview}")
        else:
            print(f"\n{icon} {name} is thinking...")
    
    @classmethod
    def agent_response(cls, agent_name: str, summary: Optional[str] = None) -> None:
        """Log agent completing work."""
        icon = cls._style(StatusIcon.SUCCESS, Style.GREEN)
        name = cls._style(agent_name.upper(), Style.GREEN)
        if summary:
            print(f"{icon} {name} completed: {summary[:60]}...")
        else:
            print(f"{icon} {name} completed")
    
    @classmethod
    def agent_handoff(cls, from_agent: str, to_agent: str, reason: str = "") -> None:
        """Log agent handoff."""
        icon = cls._style(StatusIcon.ARROW, Style.CYAN)
        from_name = cls._style(from_agent.upper(), Style.DIM)
        to_name = cls._style(to_agent.upper(), Style.CYAN, Style.BOLD)
        if reason:
            print(f"{icon} {from_name} → {to_name}: {reason}")
        else:
            print(f"{icon} {from_name} → {to_name}")
    
    # === Workflow Status ===
    
    @classmethod
    def workflow_start(cls, workflow_name: str) -> None:
        """Log workflow starting."""
        line = cls._style("─" * 50, Style.DIM)
        name = cls._style(workflow_name.upper(), Style.BOLD, Style.CYAN)
        print(f"\n{line}")
        print(f"  {StatusIcon.RUNNING} Starting workflow: {name}")
        print(f"{line}")
    
    @classmethod
    def workflow_step(cls, step_num: int, description: str) -> None:
        """Log workflow step."""
        step = cls._style(f"Step {step_num}", Style.BOLD)
        print(f"\n  {StatusIcon.BULLET} {step}: {description}")
    
    @classmethod
    def workflow_complete(cls, workflow_name: str, summary: str = "") -> None:
        """Log workflow completion."""
        line = cls._style("─" * 50, Style.DIM)
        name = cls._style(workflow_name.upper(), Style.BOLD, Style.GREEN)
        icon = cls._style(StatusIcon.SUCCESS, Style.GREEN, Style.BOLD)
        print(f"\n{line}")
        print(f"  {icon} Completed workflow: {name}")
        if summary:
            print(f"  {summary}")
        print(f"{line}")
    
    @classmethod
    def workflow_failed(cls, workflow_name: str, error: str) -> None:
        """Log workflow failure."""
        line = cls._style("─" * 50, Style.DIM)
        name = cls._style(workflow_name.upper(), Style.BOLD, Style.RED)
        icon = cls._style(StatusIcon.FAILURE, Style.RED, Style.BOLD)
        print(f"\n{line}")
        print(f"  {icon} Failed workflow: {name}")
        print(f"  Error: {error}")
        print(f"{line}")
    
    # === User Interaction ===
    
    @classmethod
    def user_prompt(cls, prompt: str = "YOU") -> str:
        """Show user input prompt and get input."""
        styled_prompt = cls._style(f"[{prompt}]: ", Style.BOLD, Style.WHITE)
        return input(styled_prompt)
    
    @classmethod
    def agent_message(cls, agent_name: str, message: str) -> None:
        """Display a message from an agent to the user."""
        name = cls._style(f"[{agent_name.upper()}]:", Style.BOLD, Style.CYAN)
        print(f"\n{name} {message}")
    
    # === Progress ===
    
    @classmethod
    def progress(cls, message: str) -> None:
        """Show a real-time progress/status update on a single line."""
        # Clear line and show status
        status = cls._style(f"  ↳ {message}", Style.DIM)
        print(f"\r\033[K{status}", end="", flush=True)
    
    @classmethod
    def progress_bar(cls, current: int, total: int, label: str = "") -> None:
        """Show progress bar indicator."""
        pct = int((current / total) * 100) if total > 0 else 0
        bar_width = 30
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        progress_text = cls._style(f"[{bar}] {pct}%", Style.CYAN)
        if label:
            print(f"\r  {progress_text} {label}", end="", flush=True)
        else:
            print(f"\r  {progress_text}", end="", flush=True)
        
        if current >= total:
            print()  # Newline when complete
    
    # === Separators and Headers ===
    
    @classmethod
    def header(cls, text: str, width: int = 60) -> None:
        """Print a section header."""
        line = cls._style("=" * width, Style.DIM)
        centered = text.center(width)
        header_text = cls._style(centered, Style.BOLD)
        print(f"\n{line}")
        print(header_text)
        print(line)
    
    @classmethod
    def subheader(cls, text: str) -> None:
        """Print a subsection header."""
        line = cls._style("─" * 40, Style.DIM)
        print(f"\n{line}")
        print(f"  {text}")
        print(line)
    
    @classmethod
    def separator(cls, char: str = "─", width: int = 50) -> None:
        """Print a separator line."""
        line = cls._style(char * width, Style.DIM)
        print(line)
    
    # === Timestamps ===
    
    @classmethod
    def timestamp(cls) -> None:
        """Print current timestamp."""
        ts = datetime.now().strftime("%H:%M:%S")
        styled = cls._style(f"[{ts}]", Style.DIM)
        print(styled, end=" ")


class ActivityLog:
    """
    Scrolling activity log for agent operations.
    
    Replaces spinner-based display with a clean scrolling log that shows:
    - Full agent activity descriptions (not truncated)
    - Tool operations as they complete
    - Progress through tasks
    
    Design principles:
    - Each activity gets its own line
    - No truncation - full context is shown
    - Maintains a "current phase" header that can be updated
    - Tool results and activities scroll naturally
    """
    
    _current_log: Optional["ActivityLog"] = None
    _current_phase: str = ""
    _current_agent: str = ""
    _last_activity: str = ""
    _running: bool = False
    
    def __init__(self, phase: str = "Working"):
        self.phase = phase
        self._paused = False
    
    @classmethod
    def current(cls) -> Optional["ActivityLog"]:
        """Get the currently active log, if any."""
        return cls._current_log
    
    @classmethod
    def set_global_activity(cls, activity: str) -> None:
        """
        Log an activity. Activities that are different from the last are printed.
        
        This is the main method called by agents to report what they're doing.
        Format: "[AGENT] is doing something specific..."
        """
        if not cls._running:
            return
        
        # Skip duplicate consecutive activities
        if activity == cls._last_activity:
            return
        cls._last_activity = activity
        
        # Parse agent name from activity if present (format: "[AGENT] activity")
        agent = ""
        msg = activity
        if activity.startswith("[") and "]" in activity:
            bracket_end = activity.index("]")
            agent = activity[1:bracket_end]
            msg = activity[bracket_end + 1:].strip()
        
        # Print the activity as a log line (with dim styling for context)
        if agent:
            agent_styled = f"{Style.CYAN}{Style.BOLD}{agent}{Style.RESET}"
            print(f"  {Style.DIM}↳{Style.RESET} {agent_styled} {msg}")
        else:
            print(f"  {Style.DIM}↳ {activity}{Style.RESET}")
    
    @classmethod
    def log_phase(cls, phase: str, agent: str = "") -> None:
        """
        Log a new phase/task. This is a more prominent header line.
        """
        cls._current_phase = phase
        cls._current_agent = agent
        
        if agent:
            agent_styled = f"{Style.MAGENTA}{Style.BOLD}{agent.upper()}{Style.RESET}"
            print(f"\n  {StatusIcon.RUNNING} {phase} ({agent_styled})")
        else:
            print(f"\n  {StatusIcon.RUNNING} {phase}")
    
    @classmethod
    def pause_for_input(cls) -> bool:
        """Pause logging for user input. Returns True if was running."""
        if cls._running:
            cls._current_log._paused = True if cls._current_log else False
            return True
        return False
    
    @classmethod
    def resume_after_input(cls) -> None:
        """Resume logging after user input."""
        if cls._current_log:
            cls._current_log._paused = False
    
    def start(self):
        """Start the activity log."""
        self._paused = False
        ActivityLog._running = True
        ActivityLog._current_log = self
        ActivityLog._current_phase = self.phase
        ActivityLog._last_activity = ""
    
    def pause(self):
        """Pause logging (for user input)."""
        self._paused = True
    
    def resume(self):
        """Resume logging after pause."""
        self._paused = False
    
    def update(self, message: str):
        """Update the current phase."""
        self.phase = message
        ActivityLog._current_phase = message
    
    def stop(self, final_message: str = None):
        """Stop the activity log with optional final message."""
        ActivityLog._running = False
        if final_message:
            print(f"  {Style.GREEN}{StatusIcon.SUCCESS}{Style.RESET} {final_message}")
        ActivityLog._current_log = None
        ActivityLog._last_activity = ""
    
    def stop_error(self, error_message: str):
        """Stop the activity log with an error message."""
        ActivityLog._running = False
        print(f"  {Style.RED}{StatusIcon.FAILURE}{Style.RESET} {error_message}")
        ActivityLog._current_log = None
        ActivityLog._last_activity = ""
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.stop_error(str(exc_val) if exc_val else "Error occurred")
        else:
            self.stop()
        return False


# Backwards compatibility alias
Spinner = ActivityLog


# Convenience singleton
console = Console()


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    This is the ONLY way agents should get timestamps - never hallucinate dates.
    Returns format: YYYY-MM-DD HH:MM:SS
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_current_date() -> str:
    """
    Get current date for documentation.
    
    This is the ONLY way agents should get dates - never hallucinate dates.
    Returns format: YYYY-MM-DD
    """
    return datetime.now().strftime("%Y-%m-%d")


# Import timing utilities
from .timing import (
    TimingCollector,
    TimingContext,
    TimingRecord,
    timing,
    timed,
    timed_operation,
    async_timed_operation,
    enable_timing,
    disable_timing,
    print_timing_summary,
)


def extract_json_from_text(text: str, fallback: dict = None) -> dict:
    """
    Extract JSON object from text that may contain markdown code blocks or prose.
    
    This is a utility for parsing LLM responses that contain JSON.
    It handles:
    - ```json ... ``` code blocks
    - Raw JSON objects
    - JSON embedded in prose
    
    Args:
        text: Text that may contain JSON
        fallback: Default value if no valid JSON found
        
    Returns:
        Parsed JSON dict, or fallback if parsing fails
    """
    import json
    
    if not text:
        return fallback or {}
    
    # Try to find JSON in code block first
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    # Try generic code block
    if "```" in text:
        start = text.find("```") + 3
        # Skip language identifier if present
        newline = text.find("\n", start)
        if newline > start:
            start = newline + 1
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()
            if json_str.startswith("{"):
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
    
    # Try to find raw JSON object by matching braces
    if "{" in text:
        start = text.find("{")
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    json_str = text[start:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        break
    
    return fallback or {}


__all__ = [
    # Console
    "Style",
    "StatusIcon",
    "Console",
    "console",
    "Spinner",
    # Utilities
    "get_current_timestamp",
    "get_current_date",
    "extract_json_from_text",
    # Timing
    "TimingCollector",
    "TimingContext",
    "TimingRecord",
    "timing",
    "timed",
    "timed_operation",
    "async_timed_operation",
    "enable_timing",
    "disable_timing",
    "print_timing_summary",
]
