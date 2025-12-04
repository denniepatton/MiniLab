from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import json


class TranscriptLogger:
    """
    Captures and logs all conversations and tool operations.
    Saves transcripts to Transcripts/ directory with timestamp and conversation name.
    """

    def __init__(self, transcripts_dir: str | Path):
        self.transcripts_dir = Path(transcripts_dir)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session data
        self.events: List[Dict[str, Any]] = []
        self.start_time = None
        self.conversation_name = None
        self.total_tokens = 0
    
    def start_session(self, conversation_name: str):
        """Start a new conversation session."""
        self.start_time = datetime.now()
        self.conversation_name = conversation_name
        self.events = []
        self.total_tokens = 0
    
    def update_session_name(self, new_name: str):
        """Update the session name (useful when project name is determined later)."""
        self.conversation_name = new_name
    
    def log_user_message(self, message: str):
        """Log a message from the user."""
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "user_message",
            "content": message,
        })
    
    def log_agent_response(
        self, 
        agent_name: str, 
        agent_id: str,
        message: str, 
        tokens_used: int | None = None
    ):
        """Log a response from an agent."""
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "agent_response",
            "agent_name": agent_name,
            "agent_id": agent_id,
            "content": message,
            "tokens_used": tokens_used,
        })
        if tokens_used:
            self.total_tokens += tokens_used
    
    def log_agent_consultation(
        self,
        from_agent: str,
        to_agent: str,
        question: str,
        response: str,
        tokens_used: int | None = None
    ):
        """Log a consultation between agents."""
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "agent_consultation",
            "from_agent": from_agent,
            "to_agent": to_agent,
            "question": question,
            "response": response,
            "tokens_used": tokens_used,
        })
        if tokens_used:
            self.total_tokens += tokens_used
    
    def log_tool_operation(
        self,
        agent_name: str,
        tool_name: str,
        action: str,
        params: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """
        Log a tool operation (concisely - no full file contents).
        
        For file writes, logs filename and size, not content.
        For file reads, logs filename only.
        """
        # Summarize the operation
        summary = f"<{action}"
        
        if action == "write" and "path" in params:
            summary += f" {params['path']}"
            if "content" in params:
                summary += f" ({len(params['content'])} bytes)"
        elif action == "read" and "path" in params:
            summary += f" {params['path']}"
        elif action == "install_package" and "packages" in params:
            summary += f" {', '.join(params['packages'])}"
        elif "path" in params:
            summary += f" {params['path']}"
        
        if result.get("success"):
            summary += " ✓>"
        else:
            summary += f" ✗: {result.get('error', 'unknown error')}>"
        
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "tool_operation",
            "agent_name": agent_name,
            "tool_name": tool_name,
            "action": action,
            "summary": summary,
            "success": result.get("success", False),
            "error": result.get("error") if not result.get("success") else None,
        })
    
    def log_stage_transition(self, stage_name: str, description: str):
        """Log a workflow stage transition (for Single Analysis mode)."""
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "stage_transition",
            "stage": stage_name,
            "description": description,
        })
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] | None = None):
        """
        Log a system event (script generation, execution, etc.).
        Use this for progress tracking and debugging.
        """
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "system_event",
            "event_type": event_type,
            "message": message,
            "details": details or {},
        })
    
    def save_transcript(self, override_name: str | None = None) -> Path:
        """
        Save the current session transcript to a file.
        
        Returns the path to the saved transcript.
        """
        if not self.start_time:
            raise ValueError("No active session to save")
        
        # Generate filename
        timestamp = self.start_time.strftime("%Y-%m-%d_%H%M")
        name = override_name or self.conversation_name or "conversation"
        # Sanitize name for filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        filename = f"{timestamp}_{safe_name}.txt"
        
        filepath = self.transcripts_dir / filename
        
        # Generate transcript content
        lines = []
        lines.append("=" * 80)
        lines.append(f"MiniLab Conversation Transcript")
        lines.append(f"Session: {self.conversation_name or 'Untitled'}")
        lines.append(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Tokens Used: ~{self.total_tokens:,}")
        lines.append("=" * 80)
        lines.append("")
        
        # Write events
        for event in self.events:
            timestamp = event["timestamp"].split("T")[1][:8]  # HH:MM:SS
            
            if event["type"] == "user_message":
                lines.append(f"[{timestamp}] User:")
                lines.append(f"  {event['content']}")
                lines.append("")
                
            elif event["type"] == "agent_response":
                token_info = f" (~{event['tokens_used']:,} tokens)" if event.get("tokens_used") else ""
                lines.append(f"[{timestamp}] {event['agent_name']}{token_info}:")
                lines.append(f"  {event['content']}")
                lines.append("")
                
            elif event["type"] == "agent_consultation":
                token_info = f" (~{event['tokens_used']:,} tokens)" if event.get("tokens_used") else ""
                lines.append(f"[{timestamp}] {event['from_agent']} → {event['to_agent']}{token_info}:")
                lines.append(f"  Q: {event['question']}")
                lines.append(f"  A: {event['response']}")
                lines.append("")
                
            elif event["type"] == "tool_operation":
                lines.append(f"[{timestamp}] {event['agent_name']} {event['summary']}")
                if event.get("error"):
                    lines.append(f"  Error: {event['error']}")
                lines.append("")
            
            elif event["type"] == "system_event":
                event_type = event.get("event_type", "INFO")
                lines.append(f"[{timestamp}] [{event_type}] {event['message']}")
                if event.get("details"):
                    for key, value in event["details"].items():
                        lines.append(f"    {key}: {value}")
                lines.append("")
                
            elif event["type"] == "stage_transition":
                lines.append("")
                lines.append("─" * 80)
                lines.append(f"[{timestamp}] STAGE: {event['stage']}")
                lines.append(f"  {event['description']}")
                lines.append("─" * 80)
                lines.append("")
        
        lines.append("=" * 80)
        lines.append(f"End of Transcript")
        lines.append("=" * 80)
        
        # Write to file
        filepath.write_text("\n".join(lines))
        
        return filepath
    
    def get_current_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        return {
            "conversation_name": self.conversation_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "event_count": len(self.events),
            "total_tokens": self.total_tokens,
        }
