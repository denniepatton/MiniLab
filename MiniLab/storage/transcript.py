"""
MiniLab Transcript System - Complete Lab Notebook.

The transcript is the SINGLE SOURCE OF TRUTH for a MiniLab session.
It captures EVERYTHING that happens, creating a scientific lab notebook
that could be published as supplementary material.

What gets logged:
- All user prompts and responses (complete text)
- All agent reasoning and responses (complete text)
- All cross-agent consultations (full dialogue)
- Tool operations (summarized: tool, action, key params, success/fail)
- Budget updates and warnings
- Workflow transitions
- Decision points and rationale

What does NOT get logged verbatim:
- Raw API payloads
- Full file contents (just paths and sizes)
- Every ReAct loop iteration (just tool summaries)

Format: Markdown for human readability and publishability.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class TranscriptLogger:
    """
    Complete session transcript capturing all substantive communication.
    
    Design philosophy: If a human would want to understand what happened
    in a session, it should be in the transcript. This includes:
    - Full conversations (not summaries)
    - Agent reasoning and decisions
    - Cross-agent consultations
    - Tool operations (what was done, not raw data)
    
    The transcript serves as:
    1. Audit trail for reproducibility
    2. Scientific documentation
    3. Debugging resource
    4. Publication supplementary material
    """

    def __init__(self, transcripts_dir: str | Path):
        self.transcripts_dir = Path(transcripts_dir)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Session state
        self.events: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.conversation_name: Optional[str] = None
        
        # Token tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.token_budget: Optional[int] = None
    
    def start_session(self, conversation_name: str, token_budget: Optional[int] = None) -> None:
        """Start a new conversation session."""
        self.start_time = datetime.now()
        self.conversation_name = conversation_name
        self.events = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.token_budget = token_budget
        
        # Opening entry
        self._add_event("session_start", {
            "project": conversation_name,
            "date": self.start_time.strftime("%B %d, %Y"),
            "time": self.start_time.strftime("%H:%M:%S"),
        })
    
    def update_session_name(self, new_name: str) -> None:
        """Update the session name."""
        self.conversation_name = new_name
    
    def set_token_budget(self, budget: int) -> None:
        """Set the token budget for this session."""
        self.token_budget = budget
        self._add_event("budget_set", {"budget": budget})
    
    # ========== User Communication ==========
    
    def log_user_message(self, message: str) -> None:
        """Log a message from the user (complete text)."""
        self._add_event("user_message", {"content": message})
    
    def log_user_response(self, prompt: str, response: str) -> None:
        """Log a user response to a question."""
        self._add_event("user_response", {
            "prompt": prompt,
            "response": response,
        })
    
    # ========== Agent Communication ==========
    
    def log_agent_message(
        self, 
        agent_name: str, 
        message: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        is_reasoning: bool = False,
    ) -> None:
        """
        Log complete agent message/response.
        
        This captures the FULL text of what an agent said, not a summary.
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        self._add_event("agent_message", {
            "agent": agent_name.upper(),
            "content": message,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "is_reasoning": is_reasoning,
<<<<<<< Updated upstream
        })
    
    def log_agent_reasoning(
        self,
        agent_name: str,
        reasoning: str,
        tokens: int = 0,
    ) -> None:
        """Log agent's reasoning/thinking process."""
        self.total_output_tokens += tokens
        
        self._add_event("agent_reasoning", {
            "agent": agent_name.upper(),
            "reasoning": reasoning,
            "tokens": tokens,
        })
    
=======
        })
    
    def log_agent_reasoning(
        self,
        agent_name: str,
        reasoning: str,
        tokens: int = 0,
    ) -> None:
        """Log agent's reasoning/thinking process."""
        self.total_output_tokens += tokens
        
        self._add_event("agent_reasoning", {
            "agent": agent_name.upper(),
            "reasoning": reasoning,
            "tokens": tokens,
        })
    
>>>>>>> Stashed changes
    # ========== Agent Consultations ==========
    
    def log_consultation_start(
        self,
        from_agent: str,
        to_agent: str,
        question: str,
    ) -> None:
        """Log the start of an agent consultation."""
        self._add_event("consultation_start", {
            "from": from_agent.upper(),
            "to": to_agent.upper(),
            "question": question,
        })
    
    def log_consultation_response(
        self,
        from_agent: str,
        to_agent: str,
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Log a consultation response (complete text)."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        self._add_event("consultation_response", {
            "from": to_agent.upper(),  # Response comes FROM the consulted agent
            "to": from_agent.upper(),  # Back TO the requesting agent
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })
    
    # ========== Tool Operations ==========
    
    def log_tool_use(
        self,
        agent_name: str,
        tool_name: str,
        action: str,
        params: Dict[str, Any],
        success: bool,
        result_summary: str = "",
        error: Optional[str] = None,
    ) -> None:
        """
        Log a tool operation (summarized, not raw data).
        
        For file operations: logs path and size, not content.
        For searches: logs query and result count, not all results.
        """
        # Build concise param summary
        param_summary = self._summarize_params(tool_name, action, params)
        
        self._add_event("tool_use", {
            "agent": agent_name.upper(),
            "tool": tool_name,
            "action": action,
            "params": param_summary,
            "success": success,
            "result": result_summary,
            "error": error,
        })
    
    def _summarize_params(
        self, 
        tool_name: str, 
        action: str, 
        params: Dict[str, Any]
    ) -> str:
        """Create concise param summary for transcript."""
        if tool_name == "filesystem":
            path = params.get("path", params.get("directory", ""))
            if action == "write":
                content_len = len(params.get("content", ""))
                return f"path={path} ({content_len} bytes)"
            elif action == "read":
                return f"path={path}"
            elif action == "list":
                return f"directory={path}"
            else:
                return f"path={path}"
        
        elif tool_name == "pubmed":
            query = params.get("query", "")
            max_results = params.get("max_results", 10)
            return f"query=\"{query}\" max={max_results}"
        
        elif tool_name == "arxiv":
            query = params.get("query", "")
            return f"query=\"{query}\""
        
        elif tool_name == "terminal":
            cmd = params.get("command", "")
            # Truncate long commands
            if len(cmd) > 80:
                cmd = cmd[:77] + "..."
            return f"command=\"{cmd}\""
        
        elif tool_name == "code_editor":
            path = params.get("path", "")
            return f"path={path}"
        
        elif tool_name == "web_search":
            query = params.get("query", "")
            return f"query=\"{query}\""
        
        else:
            # Generic: show first few params
            parts = []
            for k, v in list(params.items())[:3]:
                v_str = str(v)
                if len(v_str) > 40:
                    v_str = v_str[:37] + "..."
                parts.append(f"{k}={v_str}")
            return ", ".join(parts)
    
    # ========== Workflow Events ==========
    
    def log_workflow_start(self, workflow_name: str, phase: str = "") -> None:
        """Log workflow/phase start."""
        self._add_event("workflow_start", {
            "workflow": workflow_name,
            "phase": phase,
        })
    
    def log_workflow_complete(
        self, 
        workflow_name: str, 
        summary: str = "",
        outputs: List[str] = None,
    ) -> None:
        """Log workflow completion."""
        self._add_event("workflow_complete", {
            "workflow": workflow_name,
            "summary": summary,
            "outputs": outputs or [],
        })
    
    def log_workflow_failed(self, workflow_name: str, error: str) -> None:
        """Log workflow failure."""
        self._add_event("workflow_failed", {
            "workflow": workflow_name,
            "error": error,
        })
    
    # ========== Budget Updates ==========
    
    def log_budget_update(
        self,
        used: int,
        budget: int,
        percentage: float,
        remaining_workflows: int = 0,
    ) -> None:
        """Log budget status update."""
        self._add_event("budget_update", {
            "used": used,
            "budget": budget,
            "percentage": percentage,
            "remaining_workflows": remaining_workflows,
        })
    
    def log_budget_warning(self, percentage: float, message: str) -> None:
        """Log budget warning."""
        self._add_event("budget_warning", {
            "percentage": percentage,
            "message": message,
        })
    
    # ========== System Events ==========
    
    def log_system_event(
        self, 
        event_type: str, 
        message: str, 
        details: Dict[str, Any] = None
    ) -> None:
        """Log system event (internal operations)."""
        self._add_event("system", {
            "event_type": event_type,
            "message": message,
            "details": details or {},
        })
    
    def log_stage_transition(self, stage_name: str, description: str) -> None:
        """Log workflow stage transition."""
        self._add_event("stage_transition", {
            "stage": stage_name,
            "description": description,
        })
    
    # ========== Internal ==========
    
    def _add_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add an event to the transcript."""
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            **data,
        })
    
    # ========== Output ==========
    
    def save_transcript(self, override_name: str = None) -> Path:
        """
        Save the complete transcript to a Markdown file.
        
        Returns the path to the saved transcript.
        """
        if not self.start_time:
            raise ValueError("No active session to save")
        
        # Generate filename
        timestamp = self.start_time.strftime("%Y-%m-%d_%H%M")
        name = override_name or self.conversation_name or "conversation"
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        filename = f"{timestamp}_{safe_name}.md"
        
        filepath = self.transcripts_dir / filename
        
        # Generate Markdown content
        content = self._generate_markdown()
        filepath.write_text(content)
        
        return filepath
    
    def _generate_markdown(self) -> str:
        """Generate complete Markdown transcript."""
        lines = []
        
        # Header
        lines.append(f"# MiniLab Session Transcript")
        lines.append("")
        lines.append(f"**Project:** {self.conversation_name or 'Untitled'}")
        lines.append(f"**Date:** {self.start_time.strftime('%B %d, %Y')}")
        lines.append(f"**Started:** {self.start_time.strftime('%H:%M:%S')}")
        lines.append(f"**Ended:** {datetime.now().strftime('%H:%M:%S')}")
        lines.append("")
        
        # Token summary - query TokenAccount for authoritative data
<<<<<<< Updated upstream
=======
        lines.append("## Resource Usage")
        lines.append("")
        
>>>>>>> Stashed changes
        try:
            from ..core import get_token_account
            account = get_token_account()
            summary = account.usage_summary
            
<<<<<<< Updated upstream
            lines.append("## Resource Usage")
            lines.append("")
=======
>>>>>>> Stashed changes
            lines.append(f"- **Input Tokens:** {summary['total_input']:,}")
            lines.append(f"- **Output Tokens:** {summary['total_output']:,}")
            lines.append(f"- **Total Tokens:** {summary['total_used']:,}")
            if summary['budget']:
                lines.append(f"- **Budget:** {summary['budget']:,} ({summary['percentage_used']:.1f}% used)")
            lines.append(f"- **Estimated Cost:** ${summary['estimated_cost']:.2f}")
            
<<<<<<< Updated upstream
            # Per-agent breakdown if available
            if account._agent_usage:
                lines.append("")
                lines.append("### Per-Agent Usage")
                lines.append("")
                for agent_id in sorted(account._agent_usage.keys()):
=======
            # Per-agent breakdown from transactions
            agent_ids = set(t.agent_id for t in account._transactions)
            if agent_ids:
                lines.append("")
                lines.append("### Per-Agent Usage")
                lines.append("")
                for agent_id in sorted(agent_ids):
>>>>>>> Stashed changes
                    agent_data = account.get_agent_usage(agent_id)
                    lines.append(f"- **{agent_id.upper()}:** {agent_data['total_tokens']:,} tokens ({agent_data['call_count']} calls)")
        except Exception:
            # Fallback to local tracking if TokenAccount not available
            total = self.total_input_tokens + self.total_output_tokens
<<<<<<< Updated upstream
            lines.append("## Resource Usage")
            lines.append("")
=======
>>>>>>> Stashed changes
            lines.append(f"- **Input Tokens:** {self.total_input_tokens:,}")
            lines.append(f"- **Output Tokens:** {self.total_output_tokens:,}")
            lines.append(f"- **Total Tokens:** {total:,}")
            if self.token_budget:
                pct = (total / self.token_budget) * 100
                lines.append(f"- **Budget:** {self.token_budget:,} ({pct:.1f}% used)")
            lines.append(f"- **Estimated Cost:** ${(total / 1_000_000) * 5:.2f}")
        
        lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("## Session Log")
        lines.append("")
        
        # Events
        for event in self.events:
            lines.extend(self._format_event(event))
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*End of Transcript*")
        
        return "\n".join(lines)
<<<<<<< Updated upstream
    
    def _format_event(self, event: Dict[str, Any]) -> List[str]:
        """Format a single event as Markdown."""
        lines = []
        ts = event["timestamp"].split("T")[1][:8]  # HH:MM:SS
        event_type = event["type"]
        
        if event_type == "session_start":
            lines.append(f"### Session Started")
            lines.append(f"*{event.get('date', '')} at {event.get('time', '')}*")
        
        elif event_type == "user_message":
            lines.append(f"### [{ts}] USER")
            lines.append("")
            lines.append(event.get("content", ""))
        
        elif event_type == "user_response":
            lines.append(f"**[{ts}] User Response:**")
            lines.append(f"> {event.get('response', '')}")
        
        elif event_type == "agent_message":
            agent = event.get("agent", "AGENT")
            tokens = event.get("input_tokens", 0) + event.get("output_tokens", 0)
            token_str = f" ({tokens:,} tokens)" if tokens else ""
            
            lines.append(f"### [{ts}] {agent}{token_str}")
            lines.append("")
            lines.append(event.get("content", ""))
        
        elif event_type == "agent_reasoning":
            agent = event.get("agent", "AGENT")
            lines.append(f"**[{ts}] {agent} (reasoning):**")
            lines.append("")
            lines.append(f"*{event.get('reasoning', '')}*")
        
        elif event_type == "consultation_start":
            lines.append(f"**[{ts}] Consultation: {event.get('from', '')} → {event.get('to', '')}**")
            lines.append("")
            lines.append(f"**Question:** {event.get('question', '')}")
        
        elif event_type == "consultation_response":
            tokens = event.get("input_tokens", 0) + event.get("output_tokens", 0)
            token_str = f" ({tokens:,} tokens)" if tokens else ""
            lines.append(f"**[{ts}] Response from {event.get('from', '')}{token_str}:**")
            lines.append("")
            lines.append(event.get("response", ""))
        
        elif event_type == "tool_use":
            agent = event.get("agent", "")
            tool = event.get("tool", "")
            action = event.get("action", "")
            params = event.get("params", "")
            success = "✓" if event.get("success") else "✗"
            
            lines.append(f"**[{ts}] {agent}** used `{tool}.{action}` {success}")
            if params:
                lines.append(f"> {params}")
            if event.get("result"):
                lines.append(f"> Result: {event.get('result')}")
            if event.get("error"):
                lines.append(f"> Error: {event.get('error')}")
        
        elif event_type == "workflow_start":
            workflow = event.get("workflow", "").replace("_", " ").title()
            phase = event.get("phase", "")
            lines.append(f"### [{ts}] Workflow: {workflow}")
            if phase:
                lines.append(f"*{phase}*")
        
        elif event_type == "workflow_complete":
            workflow = event.get("workflow", "").replace("_", " ").title()
            lines.append(f"**[{ts}] ✓ {workflow} completed**")
            if event.get("summary"):
                lines.append(f"> {event.get('summary')}")
        
        elif event_type == "workflow_failed":
            workflow = event.get("workflow", "").replace("_", " ").title()
            lines.append(f"**[{ts}] ✗ {workflow} failed**")
            lines.append(f"> Error: {event.get('error', '')}")
        
        elif event_type == "stage_transition":
            lines.append(f"---")
            lines.append(f"**[{ts}] Stage: {event.get('stage', '')}**")
            lines.append(f"*{event.get('description', '')}*")
        
        elif event_type == "budget_update":
            pct = event.get("percentage", 0)
            used = event.get("used", 0)
            budget = event.get("budget", 0)
            lines.append(f"*[{ts}] Budget: {used:,}/{budget:,} tokens ({pct:.1f}% used)*")
        
        elif event_type == "budget_warning":
            lines.append(f"**[{ts}] ⚠ Budget Warning ({event.get('percentage', 0):.0f}%)**")
            lines.append(f"> {event.get('message', '')}")
        
        elif event_type == "budget_set":
            lines.append(f"*[{ts}] Token budget set: {event.get('budget', 0):,}*")
        
        elif event_type == "system":
            event_subtype = event.get("event_type", "info")
            lines.append(f"*[{ts}] [{event_subtype}] {event.get('message', '')}*")
        
        return lines
    
=======
    
    def _format_event(self, event: Dict[str, Any]) -> List[str]:
        """Format a single event as Markdown."""
        lines = []
        ts = event["timestamp"].split("T")[1][:8]  # HH:MM:SS
        event_type = event["type"]
        
        if event_type == "session_start":
            lines.append(f"### Session Started")
            lines.append(f"*{event.get('date', '')} at {event.get('time', '')}*")
        
        elif event_type == "user_message":
            lines.append(f"### [{ts}] USER")
            lines.append("")
            lines.append(event.get("content", ""))
        
        elif event_type == "user_response":
            lines.append(f"**[{ts}] User Response:**")
            lines.append(f"> {event.get('response', '')}")
        
        elif event_type == "agent_message":
            agent = event.get("agent", "AGENT")
            tokens = event.get("input_tokens", 0) + event.get("output_tokens", 0)
            token_str = f" ({tokens:,} tokens)" if tokens else ""
            
            lines.append(f"### [{ts}] {agent}{token_str}")
            lines.append("")
            lines.append(event.get("content", ""))
        
        elif event_type == "agent_reasoning":
            agent = event.get("agent", "AGENT")
            lines.append(f"**[{ts}] {agent} (reasoning):**")
            lines.append("")
            lines.append(f"*{event.get('reasoning', '')}*")
        
        elif event_type == "consultation_start":
            lines.append(f"**[{ts}] Consultation: {event.get('from', '')} → {event.get('to', '')}**")
            lines.append("")
            lines.append(f"**Question:** {event.get('question', '')}")
        
        elif event_type == "consultation_response":
            tokens = event.get("input_tokens", 0) + event.get("output_tokens", 0)
            token_str = f" ({tokens:,} tokens)" if tokens else ""
            lines.append(f"**[{ts}] Response from {event.get('from', '')}{token_str}:**")
            lines.append("")
            lines.append(event.get("response", ""))
        
        elif event_type == "tool_use":
            agent = event.get("agent", "")
            tool = event.get("tool", "")
            action = event.get("action", "")
            params = event.get("params", "")
            success = "✓" if event.get("success") else "✗"
            
            lines.append(f"**[{ts}] {agent}** used `{tool}.{action}` {success}")
            if params:
                lines.append(f"> {params}")
            if event.get("result"):
                lines.append(f"> Result: {event.get('result')}")
            if event.get("error"):
                lines.append(f"> Error: {event.get('error')}")
        
        elif event_type == "workflow_start":
            workflow = event.get("workflow", "").replace("_", " ").title()
            phase = event.get("phase", "")
            lines.append(f"### [{ts}] Workflow: {workflow}")
            if phase:
                lines.append(f"*{phase}*")
        
        elif event_type == "workflow_complete":
            workflow = event.get("workflow", "").replace("_", " ").title()
            lines.append(f"**[{ts}] ✓ {workflow} completed**")
            if event.get("summary"):
                lines.append(f"> {event.get('summary')}")
        
        elif event_type == "workflow_failed":
            workflow = event.get("workflow", "").replace("_", " ").title()
            lines.append(f"**[{ts}] ✗ {workflow} failed**")
            lines.append(f"> Error: {event.get('error', '')}")
        
        elif event_type == "stage_transition":
            lines.append(f"---")
            lines.append(f"**[{ts}] Stage: {event.get('stage', '')}**")
            lines.append(f"*{event.get('description', '')}*")
        
        elif event_type == "budget_update":
            pct = event.get("percentage", 0)
            used = event.get("used", 0)
            budget = event.get("budget", 0)
            lines.append(f"*[{ts}] Budget: {used:,}/{budget:,} tokens ({pct:.1f}% used)*")
        
        elif event_type == "budget_warning":
            lines.append(f"**[{ts}] ⚠ Budget Warning ({event.get('percentage', 0):.0f}%)**")
            lines.append(f"> {event.get('message', '')}")
        
        elif event_type == "budget_set":
            lines.append(f"*[{ts}] Token budget set: {event.get('budget', 0):,}*")
        
        elif event_type == "system":
            event_subtype = event.get("event_type", "info")
            lines.append(f"*[{ts}] [{event_subtype}] {event.get('message', '')}*")
        
        return lines
    
>>>>>>> Stashed changes
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary for programmatic use."""
        total = self.total_input_tokens + self.total_output_tokens
        return {
            "conversation_name": self.conversation_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "event_count": len(self.events),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": total,
            "token_budget": self.token_budget,
            "estimated_cost": (total / 1_000_000) * 5,
        }

