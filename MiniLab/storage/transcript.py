"""
MiniLab Transcript - Human-Readable Narrative Only.

The transcript is a human-readable lab notebook. It captures
the narrative of what happened for human understanding.

Design philosophy:
- Human-readable Markdown only (no JSONL, no artifacts dir)
- ProjectSSOT is the machine-queryable state (not the transcript)
- Lightweight, minimal overhead
- Integrates with SSOT for authoritative state
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.project_ssot import ProjectSSOT


class TranscriptWriter:
    """
    Human-readable session transcript.
    
    This is a write-once narrative log. Machine-queryable state
    lives in ProjectSSOT, so the transcript can focus purely
    on being readable by humans.
    
    Output: Single transcript.md file in project directory.
    
    Usage:
        # Create with deferred initialization
        transcript = TranscriptWriter(project_path)
        transcript.start_session(project_name)
        
        # Or create fully initialized
        transcript = TranscriptWriter(project_path, project_name)
    """

    def __init__(self, project_path: Path, project_name: Optional[str] = None):
        """
        Initialize transcript writer.
        
        Args:
            project_path: Path to project directory
            project_name: Human-readable project name (optional, can set via start_session)
        """
        self.project_path = Path(project_path)
        self.project_name = project_name or "MiniLab Session"
        self.start_time = datetime.now()
        self._lines: list[str] = []
        self._initialized = False
        self._token_budget: Optional[int] = None
        
    @property
    def transcript_path(self) -> Path:
        """Path to the transcript file."""
        return self.project_path / "transcript.md"
    
    def _ensure_initialized(self) -> None:
        """Initialize transcript file with header if needed."""
        if self._initialized:
            return
        
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        # Write header
        lines = [
            f"# {self.project_name}",
            "",
            f"**Date:** {self.start_time.strftime('%B %d, %Y')}",
            f"**Started:** {self.start_time.strftime('%H:%M:%S')}",
            "",
            "---",
            "",
        ]
        
        self.transcript_path.write_text("\n".join(lines))
        self._initialized = True
    
    def _timestamp(self) -> str:
        """Get current timestamp for entries."""
        return datetime.now().strftime("%H:%M:%S")
    
    def _append(self, content: str) -> None:
        """Append content to transcript."""
        self._ensure_initialized()
        
        with open(self.transcript_path, "a") as f:
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")
    
    # ========== Session Lifecycle (deferred init) ==========
    
    def start_session(self, project_name: str) -> None:
        """
        Start or reinitialize session with project name.
        
        Call this after construction if project_name wasn't provided
        to the constructor, or to reset the session.
        
        Args:
            project_name: Human-readable project name
        """
        self.project_name = project_name
        self.start_time = datetime.now()
        self._initialized = False  # Force re-initialization with new name
    
    def set_token_budget(self, budget: int) -> None:
        """Set the token budget for budget tracking."""
        self._token_budget = budget
    
    # ========== User Communication ==========
    
    def log_user_message(self, message: str) -> None:
        """Log message from user."""
        self._append(f"\n### [{self._timestamp()}] USER\n\n{message}\n")
    
    def log_user_response(self, prompt: str, response: str) -> None:
        """Log user response to a question."""
        self._append(f"\n**[{self._timestamp()}] User Response:**\n> {response}\n")
    
    # ========== Agent Communication ==========
    
    def log_agent_message(
        self,
        agent_name: str,
        message: str,
        tokens: int = 0,
    ) -> None:
        """Log agent message/response."""
        token_str = f" ({tokens:,} tokens)" if tokens else ""
        self._append(f"\n### [{self._timestamp()}] {agent_name.upper()}{token_str}\n\n{message}\n")
    
    def log_agent_reasoning(self, agent_name: str, reasoning: str) -> None:
        """Log agent reasoning (optional, for verbose mode)."""
        self._append(f"\n**[{self._timestamp()}] {agent_name.upper()} (reasoning):**\n\n*{reasoning}*\n")
    
    # ========== Streaming ==========
    
    def log_stream_chunk(self, agent_id: str, chunk: str) -> None:
        """Log a streaming chunk (no-op in markdown transcript, for API compatibility)."""
        # Streaming chunks are transient - we don't log them to the narrative transcript
        # Full message will be logged via log_agent_message when complete
        pass
    
    # ========== Consultations ==========
    
    def log_consultation(
        self,
        from_agent: str,
        to_agent: str,
        question: str,
        response: str,
        tokens: int = 0,
    ) -> None:
        """Log a complete agent consultation."""
        token_str = f" ({tokens:,} tokens)" if tokens else ""
        self._append(f"""
**[{self._timestamp()}] Consultation: {from_agent.upper()} → {to_agent.upper()}**

**Question:** {question}

**Response{token_str}:**

{response}
""")
    
    def log_consultation_start(
        self,
        from_agent: str,
        to_agent: str,
        question: str,
    ) -> None:
        """Log the start of a consultation."""
        self._append(f"\n**[{self._timestamp()}] {from_agent.upper()} → {to_agent.upper()}** (consultation started)\n")
        self._append(f"> {question[:200]}{'...' if len(question) > 200 else ''}\n")
    
    def log_consultation_response(
        self,
        from_agent: str,
        to_agent: str,
        response: str,
    ) -> None:
        """Log the response to a consultation."""
        self._append(f"\n**[{self._timestamp()}] {to_agent.upper()} → {from_agent.upper()}** (response)\n")
        # Truncate very long responses for readability
        if len(response) > 500:
            self._append(f"> {response[:500]}...[truncated]\n")
        else:
            self._append(f"> {response}\n")
    
    # ========== Tool Operations ==========
    
    _tool_call_counter: int = 0  # Class-level counter for call IDs
    
    def log_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        action: str,
        params: dict[str, Any],
    ) -> str:
        """
        Log start of a tool call.
        
        Returns a call_id for correlation with log_tool_result.
        """
        TranscriptWriter._tool_call_counter += 1
        call_id = f"tool_{TranscriptWriter._tool_call_counter}"
        
        # Summarize params for readability
        params_str = ", ".join(f"{k}={repr(v)[:30]}" for k, v in list(params.items())[:3])
        if len(params) > 3:
            params_str += f", ... ({len(params)} total)"
        
        self._append(f"\n**[{self._timestamp()}] {agent_name.upper()}** → `{tool_name}.{action}({params_str})`\n")
        return call_id
    
    def log_tool_result(
        self,
        call_id: str,
        agent_name: str,
        tool_name: str,
        action: str,
        success: bool,
        duration_ms: int,
        artifact_path: Optional[str] = None,
        result_snippet: str = "",
        error: Optional[str] = None,
        raw_bytes: int = 0,
        injected_chars: int = 0,
    ) -> None:
        """Log the result of a tool call."""
        status = "✓" if success else "✗"
        duration_str = f"{duration_ms}ms" if duration_ms < 1000 else f"{duration_ms/1000:.1f}s"
        
        if success:
            snippet = result_snippet[:100] + "..." if len(result_snippet) > 100 else result_snippet
            self._append(f"  {status} {duration_str} | {snippet}\n")
        else:
            self._append(f"  {status} {duration_str} | Error: {error or 'Unknown error'}\n")
    
    def save_tool_artifact(
        self,
        call_id: str,
        content: str,
        suffix: str = ".txt",
    ) -> Optional[str]:
        """
        Save a tool artifact to disk.
        
        Returns the relative path to the artifact, or None if saving failed.
        
        Note: In the simplified transcript, we don't save artifacts separately.
        Full tool outputs should go in ProjectSSOT for replay capability.
        This method is provided for API compatibility.
        """
        # For the simplified transcript, we just return None
        # Full artifact persistence is handled by ProjectSSOT
        return None
    
    def log_tool_use(
        self,
        agent_name: str,
        tool_name: str,
        action: str,
        summary: str,
        success: bool = True,
    ) -> None:
        """
        Log a tool operation (summary only).
        
        Full tool outputs go in SSOT if needed for replay.
        The transcript just needs the narrative.
        """
        status = "✓" if success else "✗"
        self._append(f"\n**[{self._timestamp()}] {agent_name.upper()}** used `{tool_name}.{action}` {status}\n> {summary}\n")
    
    # ========== Workflow Events ==========
    
    def log_module_start(self, module_name: str, description: str = "") -> None:
        """Log start of a workflow module."""
        title = module_name.replace("_", " ").title()
        desc = f"\n*{description}*" if description else ""
        self._append(f"\n## [{self._timestamp()}] {title}{desc}\n")
    
    def log_module_complete(self, module_name: str, summary: str = "") -> None:
        """Log completion of a workflow module."""
        title = module_name.replace("_", " ").title()
        summ = f"\n> {summary}" if summary else ""
        self._append(f"\n**[{self._timestamp()}] ✓ {title} completed**{summ}\n")
    
    def log_task_start(self, task_title: str) -> None:
        """Log start of a task."""
        self._append(f"\n### [{self._timestamp()}] Task: {task_title}\n")
    
    def log_task_complete(self, task_title: str, summary: str = "") -> None:
        """Log task completion."""
        summ = f" - {summary}" if summary else ""
        self._append(f"\n**[{self._timestamp()}] ✓ {task_title}{summ}**\n")
    
    # ========== Budget Updates ==========
    
    def log_budget_status(
        self,
        used: int,
        budget: int,
        message: str = "",
    ) -> None:
        """Log budget status update."""
        pct = (used / budget * 100) if budget > 0 else 0
        msg = f" - {message}" if message else ""
        self._append(f"\n*[{self._timestamp()}] Budget: {used:,}/{budget:,} tokens ({pct:.1f}% used){msg}*\n")
    
    def log_budget_guidance(self, message: str) -> None:
        """Log budget guidance for agents."""
        self._append(f"\n**[{self._timestamp()}] 💡 Budget Guidance:**\n> {message}\n")
    
    def log_budget_warning(self, percentage: float, message: str) -> None:
        """Log budget warning when threshold crossed."""
        self._append(f"\n**[{self._timestamp()}] ⚠️ Budget Warning ({percentage:.0f}%):**\n> {message}\n")
    
    # ========== Stage/Phase Transitions ==========
    
    def log_stage_transition(self, stage: str, description: str = "") -> None:
        """Log transition to a new stage/phase."""
        title = stage.replace("_", " ").title()
        desc = f" - {description}" if description else ""
        self._append(f"\n## [{self._timestamp()}] Stage: {title}{desc}\n")
    
    # ========== System Events ==========
    
    def log_system_event(
        self,
        event_type: str,
        message: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log a system event.
        
        Args:
            event_type: Type of event (e.g., "session_resumed", "task_failed")
            message: Human-readable message (optional)
            metadata: Additional structured data (optional)
        """
        msg_part = f": {message}" if message else ""
        self._append(f"\n*[{self._timestamp()}] {event_type}{msg_part}*\n")
        
        # Log metadata if present (as indented details)
        if metadata:
            for key, value in metadata.items():
                self._append(f"  - {key}: {value}\n")
    
    def log_error(self, message: str) -> None:
        """Log an error."""
        self._append(f"\n**[{self._timestamp()}] ❌ Error:**\n> {message}\n")
    
    # ========== Session Lifecycle ==========
    
    # ========== Save & Finalize ==========
    
    def save_transcript(self) -> Path:
        """
        Save/flush the transcript to disk.
        
        This is a lightweight save that just ensures content is written.
        Use finalize() for a complete session summary.
        
        Returns:
            Path to the transcript file
        """
        self._ensure_initialized()
        return self.transcript_path
    
    def finalize(self, ssot: Optional["ProjectSSOT"] = None) -> Path:
        """
        Finalize the transcript with summary.
        
        Pulls authoritative token usage from TokenAccount and
        project state from SSOT.
        """
        self._ensure_initialized()
        
        lines = [
            "",
            "---",
            "",
            "## Session Summary",
            "",
            f"**Completed:** {datetime.now().strftime('%H:%M:%S')}",
            "",
        ]
        
        # Token summary from TokenAccount (authoritative)
        try:
            from ..core import get_token_account
            account = get_token_account()
            summary = account.usage_summary
            
            lines.extend([
                "### Resource Usage",
                "",
                f"- **Input Tokens:** {summary['total_input']:,}",
                f"- **Output Tokens:** {summary['total_output']:,}",
                f"- **Total Tokens:** {summary['total_used']:,}",
            ])
            
            if summary.get('budget'):
                lines.append(f"- **Budget:** {summary['budget']:,} ({summary['percentage_used']:.1f}% used)")
            
            lines.append(f"- **Estimated Cost:** ${summary['estimated_cost']:.2f}")
            lines.append("")
            
            # Per-module breakdown (if taxonomy available)
            by_module = account.aggregate_by_module()
            if by_module:
                lines.append("### Per-Module Usage")
                lines.append("")
                for module, tokens in sorted(by_module.items(), key=lambda x: -x[1]):
                    lines.append(f"- **{module}:** {tokens:,} tokens")
                lines.append("")
                
        except Exception:
            lines.extend([
                "### Resource Usage",
                "",
                "*Token data unavailable*",
                "",
            ])
        
        # SSOT summary (if available)
        if ssot:
            lines.extend([
                "### Project Status",
                "",
                f"- **Status:** {ssot.status.value}",
            ])
            
            if ssot.tasks:
                completed = sum(1 for t in ssot.tasks if t.status.value == "completed")
                lines.append(f"- **Tasks Completed:** {completed}/{len(ssot.tasks)}")
            
            if ssot.deliverables:
                lines.extend([
                    "",
                    "### Deliverables",
                    "",
                ])
                for d in ssot.deliverables:
                    lines.append(f"- {d}")
            
            lines.append("")
        
        lines.extend([
            "---",
            "",
            "*End of Transcript*",
        ])
        
        self._append("\n".join(lines))
        
        return self.transcript_path


# Legacy compatibility: alias the old class name
TranscriptLogger = TranscriptWriter


def get_transcript_writer(
    project_path: Path,
    project_name: Optional[str] = None,
) -> TranscriptWriter:
    """
    Factory function to get transcript writer.
    
    Args:
        project_path: Path to project directory
        project_name: Human-readable project name (optional)
    """
    return TranscriptWriter(project_path, project_name)
