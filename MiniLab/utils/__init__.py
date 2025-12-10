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
    def progress(cls, current: int, total: int, label: str = "") -> None:
        """Show progress indicator."""
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


class Spinner:
    """
    Animated spinner for showing activity.
    
    Usage:
        with Spinner("Processing..."):
            do_work()
        
        # Or manually:
        spinner = Spinner("Working")
        spinner.start()
        do_work()
        spinner.stop("Done!")
    """
    
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self, message: str = "Working", color: str = Style.CYAN):
        self.message = message
        self.color = color
        self._running = False
        self._thread = None
        self._frame_idx = 0
        self._lock = None
    
    def _animate(self):
        """Animation loop running in background thread."""
        import time
        try:
            while self._running:
                frame = self.FRAMES[self._frame_idx % len(self.FRAMES)]
                styled_frame = f"{self.color}{frame}{Style.RESET}"
                # Write spinner, message, then return cursor to start of line
                with self._lock:
                    sys.stdout.write(f"\r  {styled_frame} {self.message}".ljust(60))
                    sys.stdout.flush()
                self._frame_idx += 1
                time.sleep(0.1)
        except Exception:
            pass  # Silently exit if stdout is closed
    
    def start(self):
        """Start the spinner."""
        import threading
        if self._running:
            return
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
    
    def update(self, message: str):
        """Update the spinner message."""
        self.message = message
    
    def stop(self, final_message: str = None):
        """Stop the spinner with optional final message."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.3)
            self._thread = None
        # Clear the spinner line
        try:
            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.flush()
            if final_message:
                print(f"  {Style.GREEN}{StatusIcon.SUCCESS}{Style.RESET} {final_message}")
        except Exception:
            pass
    
    def stop_error(self, error_message: str):
        """Stop the spinner with an error message."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.3)
            self._thread = None
        # Clear the spinner line
        try:
            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.flush()
            print(f"  {Style.RED}{StatusIcon.FAILURE}{Style.RESET} {error_message}")
        except Exception:
            pass
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.stop_error(str(exc_val) if exc_val else "Error occurred")
        else:
            self.stop()
        return False


# Convenience singleton
console = Console()
