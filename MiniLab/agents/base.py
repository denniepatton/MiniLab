"""
Agent base class with ReAct loop and state management.

Implements:
- ReAct-style execution loop
- Persistent state per task
- Interrupt/pause capabilities
- Typed tool integration
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from ..context import ContextManager, ProjectContext, TaskState
from ..tools.base import Tool, ToolOutput


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentResponse:
    """Response from agent execution."""
    status: AgentStatus
    result: Optional[str] = None
    outputs: list[str] = field(default_factory=list)
    error: Optional[str] = None
    iterations: int = 0
    tool_calls: int = 0
    colleague_calls: int = 0
    
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "result": self.result,
            "outputs": self.outputs,
            "error": self.error,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "colleague_calls": self.colleague_calls,
        }


@dataclass
class AgentState:
    """
    Persistent state for an agent within a task.
    
    Allows resuming execution after interrupts.
    """
    agent_id: str
    task_id: str
    project_name: str
    
    # Execution state
    status: AgentStatus = AgentStatus.IDLE
    iteration: int = 0
    messages: list[dict] = field(default_factory=list)
    
    # Tool/colleague tracking
    tool_calls: list[dict] = field(default_factory=list)
    colleague_calls: list[dict] = field(default_factory=list)
    
    # Checkpoints
    last_checkpoint: Optional[datetime] = None
    checkpoint_data: dict = field(default_factory=dict)
    
    # Timestamps
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "project_name": self.project_name,
            "status": self.status.value,
            "iteration": self.iteration,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "colleague_calls": self.colleague_calls,
            "last_checkpoint": self.last_checkpoint.isoformat() if self.last_checkpoint else None,
            "checkpoint_data": self.checkpoint_data,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> AgentState:
        state = cls(
            agent_id=data["agent_id"],
            task_id=data["task_id"],
            project_name=data["project_name"],
        )
        state.status = AgentStatus(data.get("status", "idle"))
        state.iteration = data.get("iteration", 0)
        state.messages = data.get("messages", [])
        state.tool_calls = data.get("tool_calls", [])
        state.colleague_calls = data.get("colleague_calls", [])
        state.checkpoint_data = data.get("checkpoint_data", {})
        
        if data.get("last_checkpoint"):
            state.last_checkpoint = datetime.fromisoformat(data["last_checkpoint"])
        if data.get("started_at"):
            state.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("paused_at"):
            state.paused_at = datetime.fromisoformat(data["paused_at"])
        if data.get("completed_at"):
            state.completed_at = datetime.fromisoformat(data["completed_at"])
        
        return state
    
    def save(self, path: Path) -> None:
        """Save state to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> AgentState:
        """Load state from file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


class Agent:
    """
    Base agent class with ReAct execution loop.
    
    Features:
    - ReAct-style: Call LLM -> Run tool -> Append result -> Repeat
    - Self-critique before committing outputs (CellVoyager pattern)
    - Budget-scaled iterations (VS Code bounded tool calling)
    - Persistent state per task
    - Interrupt/pause capabilities
    - Typed tool integration
    """
    
    # Regex patterns for parsing agent output
    TOOL_PATTERN = re.compile(r"```tool\s*(.*?)\s*```", re.DOTALL)
    COLLEAGUE_PATTERN = re.compile(r"```colleague\s*(.*?)\s*```", re.DOTALL)
    DONE_PATTERN = re.compile(r"```done\s*(.*?)\s*```", re.DOTALL)
    EXTEND_PATTERN = re.compile(r"```extend\s*(.*?)\s*```", re.DOTALL)

    # Unified ordered parser for multiple action blocks
    ACTION_BLOCK_PATTERN = re.compile(
        r"```(tool|colleague|done|extend)\s*(.*?)\s*```",
        re.DOTALL,
    )
    
    # Budget-scaled iteration limits
    MIN_ITERATIONS = 10  # Floor for even the smallest budgets
    MAX_ITERATIONS_CAP = 100  # Ceiling for safety
    TOKENS_PER_ITERATION = 3000  # Rough estimate of tokens per ReAct iteration

    def _strip_action_blocks_for_transcript(self, text: str) -> str:
        """Remove tool/colleague/done/extend fenced blocks for transcript readability."""
        if not text:
            return ""
        cleaned = self.ACTION_BLOCK_PATTERN.sub("", text)
        # Also remove any now-empty triple-backtick remnants (defensive)
        cleaned = re.sub(r"```\s*```", "", cleaned)
        return cleaned.strip()
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        guild: str,
        system_prompt: str,
        llm_backend: Any,  # LLMBackend
        tools: dict[str, Tool],
        context_manager: ContextManager,
        max_iterations: int = 50,
        transcript_logger: Any = None,  # TranscriptLogger, optional
    ):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique agent identifier
            name: Display name
            guild: Guild membership
            system_prompt: Structured system prompt
            llm_backend: LLM backend for completions
            tools: Dict of tool name to Tool instance
            context_manager: Context manager for RAG
            max_iterations: Max ReAct iterations (can be scaled by budget)
            transcript_logger: Optional transcript logger for session logging
        """
        self.agent_id = agent_id
        self.name = name
        self.guild = guild
        self.system_prompt = system_prompt
        self.llm = llm_backend
        self.tools = tools
        self.context_manager = context_manager
        self._base_max_iterations = max_iterations
        self.max_iterations = max_iterations  # May be scaled by budget
        self.transcript = transcript_logger
        
        # Set agent_id on LLM backend for token tracking
        if hasattr(self.llm, 'agent_id'):
            self.llm.agent_id = agent_id
        
        # Colleagues (set by registry)
        self.colleagues: dict[str, Agent] = {}
        
        # State management
        self._current_state: Optional[AgentState] = None
        self._interrupt_requested = False
        self._pause_requested = False
        
        # Callbacks
        self.on_message: Optional[Callable[[str, str], None]] = None  # (agent_id, message)
        self.on_tool_call: Optional[Callable[[str, str, dict], None]] = None  # (tool, action, params)
        self.on_stream_chunk: Optional[Callable[[str, str], None]] = None  # (agent_id, chunk)
        
        # Streaming mode
        self.streaming_enabled: bool = False

        # Token attribution: what triggered the next LLM call
        self._next_llm_trigger: Optional[str] = None
        
        # Self-critique tracking
        self._critique_enabled: bool = True
        self._last_critique: Optional[str] = None
    
    def set_transcript(self, transcript_logger: Any) -> None:
        """Set the transcript logger for this agent."""
        self.transcript = transcript_logger
    
    def set_colleagues(self, colleagues: dict[str, Agent]) -> None:
        """Set colleague references."""
        self.colleagues = colleagues
    
    def scale_iterations_to_budget(self, workflow_budget: Optional[int] = None) -> int:
        """
        Scale max_iterations based on available budget.
        
        Implements VS Code-style bounded tool calling where iterations
        are proportional to allocated budget.
        
        Args:
            workflow_budget: Budget for current workflow (tokens)
        
        Returns:
            Scaled max_iterations
        """
        if not workflow_budget:
            # Try to get from BudgetManager
            try:
                from ..config.budget_manager import get_budget_manager
                from ..core.token_context import get_current_workflow
                
                workflow = get_current_workflow()
                if workflow:
                    wb = get_budget_manager().get_workflow_budget(workflow)
                    if wb:
                        workflow_budget = wb.remaining
            except Exception:
                pass
        
        if not workflow_budget or workflow_budget <= 0:
            return self._base_max_iterations
        
        # Calculate iterations based on budget
        # Each iteration costs ~TOKENS_PER_ITERATION tokens
        budget_iterations = workflow_budget // self.TOKENS_PER_ITERATION
        
        # Clamp between MIN and MAX
        scaled = max(self.MIN_ITERATIONS, min(self.MAX_ITERATIONS_CAP, budget_iterations))
        
        # Don't exceed configured max
        self.max_iterations = min(scaled, self._base_max_iterations)
        return self.max_iterations
    
    async def critique_output(
        self,
        output: str,
        task: str,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Self-critique an output before committing (CellVoyager pattern).
        
        Evaluates output quality and suggests improvements. Used before
        finalizing important outputs like analysis plans or reports.
        
        Args:
            output: The output to critique
            task: Original task/objective
            context: Additional context for evaluation
        
        Returns:
            {
                "approved": bool,  # Whether output passes critique
                "issues": list[str],  # List of issues found
                "suggestions": list[str],  # Improvement suggestions
                "revised_output": Optional[str],  # Revised version if needed
            }
        """
        if not self._critique_enabled:
            return {"approved": True, "issues": [], "suggestions": [], "revised_output": None}
        
        critique_prompt = f"""You are reviewing your own work for quality before finalizing.

## Original Task
{task}

{f"## Additional Context{chr(10)}{context}" if context else ""}

## Output to Review
{output[:8000]}{"..." if len(output) > 8000 else ""}

## Critique Instructions
Evaluate the output critically:
1. Does it fully address the task requirements?
2. Are there factual errors or inconsistencies?
3. Is it well-structured and clear?
4. Are there missing important elements?
5. Could it be improved significantly?

Respond with ONLY a JSON object:
{{
    "approved": true/false,
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "needs_revision": true/false
}}

Be honest but not overly critical. Approve if the output is good enough."""

        try:
            from ..core.token_context import token_context
            
            messages = [
                {"role": "system", "content": "You are a quality assurance critic. Be constructive but honest."},
                {"role": "user", "content": critique_prompt}
            ]
            
            with token_context(trigger="self_critique"):
                response = await self.llm.acomplete(messages)
            
            # Parse response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            critique = json.loads(response)
            self._last_critique = json.dumps(critique, indent=2)
            
            return {
                "approved": critique.get("approved", True),
                "issues": critique.get("issues", []),
                "suggestions": critique.get("suggestions", []),
                "revised_output": None,  # Revision handled separately if needed
            }
            
        except Exception as e:
            # On error, approve to avoid blocking
            return {
                "approved": True,
                "issues": [f"Critique failed: {e}"],
                "suggestions": [],
                "revised_output": None,
            }
    
    async def incorporate_critique(
        self,
        output: str,
        critique: dict[str, Any],
        task: str,
    ) -> str:
        """
        Revise output based on critique (CellVoyager pattern).
        
        Args:
            output: Original output
            critique: Critique results from critique_output()
            task: Original task
        
        Returns:
            Revised output
        """
        if critique.get("approved", True) and not critique.get("issues"):
            return output
        
        issues_text = "\n".join(f"- {i}" for i in critique.get("issues", []))
        suggestions_text = "\n".join(f"- {s}" for s in critique.get("suggestions", []))
        
        revision_prompt = f"""Revise your previous output based on self-critique feedback.

## Original Task
{task}

## Previous Output
{output[:6000]}{"..." if len(output) > 6000 else ""}

## Issues Found
{issues_text or "None"}

## Suggestions
{suggestions_text or "None"}

## Instructions
Revise the output to address the issues and incorporate suggestions.
Keep what was good, fix what was problematic.
Return ONLY the revised output, no explanation."""

        try:
            from ..core.token_context import token_context
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": revision_prompt}
            ]
            
            with token_context(trigger="incorporate_critique"):
                revised = await self.llm.acomplete(messages)
            
            return revised.strip()
            
        except Exception:
            # On error, return original
            return output
    
    def enable_streaming(
        self,
        on_chunk: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """
        Enable streaming output for this agent.
        
        Args:
            on_chunk: Callback receiving (agent_id, text_chunk) for each streamed chunk
        """
        self.streaming_enabled = True
        if on_chunk:
            self.on_stream_chunk = on_chunk
    
    def disable_streaming(self) -> None:
        """Disable streaming output."""
        self.streaming_enabled = False
    
    def _report_activity(self, activity: str) -> None:
        """Report current activity to the global spinner."""
        try:
            from ..utils import Spinner
            Spinner.set_global_activity(activity)
        except Exception:
            pass  # Spinner may not be active
    
    def request_interrupt(self) -> None:
        """Request agent to interrupt at next safe point."""
        self._interrupt_requested = True
    
    def request_pause(self) -> None:
        """Request agent to pause at next safe point."""
        self._pause_requested = True
    
    async def execute(
        self,
        task: str,
        project_name: str,
        task_id: Optional[str] = None,
        resume_state: Optional[AgentState] = None,
        workflow_budget: Optional[int] = None,
    ) -> AgentResponse:
        """
        Execute a task using the ReAct loop.
        
        Args:
            task: Task description/prompt
            project_name: Current project
            task_id: Unique task identifier (generated if not provided)
            resume_state: State to resume from (if paused previously)
            workflow_budget: Budget for this workflow (scales iterations)
            
        Returns:
            AgentResponse with results
        """
        # Report activity to spinner
        self._report_activity(f"[{self.agent_id.upper()}] is starting task")
        
        # Scale iterations based on budget
        self.scale_iterations_to_budget(workflow_budget)
        
        # Initialize or resume state
        if resume_state:
            self._current_state = resume_state
            self._current_state.status = AgentStatus.RUNNING
            self._current_state.paused_at = None
        else:
            task_id = task_id or f"{self.agent_id}_{int(time.time())}"
            self._current_state = AgentState(
                agent_id=self.agent_id,
                task_id=task_id,
                project_name=project_name,
                status=AgentStatus.RUNNING,
                started_at=datetime.now(),
            )
        
        state = self._current_state
        self._interrupt_requested = False
        self._pause_requested = False
        
        # Build initial context
        context = self.context_manager.build_context(
            agent_id=self.agent_id,
            project_name=project_name,
            query=task,
        )
        
        # Get current budget status for context injection
        budget_status = ""
        try:
            from ..core import get_token_account
            account = get_token_account()
            if account.budget and account.budget > 0:
                pct_used = (account.total_used / account.budget) * 100
                remaining = account.budget - account.total_used
                budget_status = f"\n\n## Budget Status\n📊 {pct_used:.0f}% of session budget used ({account.total_used:,}/{account.budget:,} tokens). ~{remaining:,} remaining."
                if pct_used >= 80:
                    budget_status += "\n⚠️ Budget is constrained. Prioritize completing core deliverables."
                elif pct_used >= 60:
                    budget_status += "\n📝 Be efficient. Focus on actionable outputs."
        except (ImportError, Exception):
            pass  # Budget tracking not available
        
        # Initialize messages if starting fresh
        if not state.messages:
            state.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"## Context\n{context.to_prompt()}{budget_status}\n\n## Task\n{task}"},
            ]
        
        # ReAct loop
        while state.iteration < self.max_iterations:
            # Check for interrupt/pause
            if self._interrupt_requested:
                state.status = AgentStatus.FAILED
                state.completed_at = datetime.now()
                return AgentResponse(
                    status=AgentStatus.FAILED,
                    error="Interrupted by user",
                    iterations=state.iteration,
                    tool_calls=len(state.tool_calls),
                    colleague_calls=len(state.colleague_calls),
                )
            
            if self._pause_requested:
                state.status = AgentStatus.PAUSED
                state.paused_at = datetime.now()
                self._checkpoint(state, project_name)
                return AgentResponse(
                    status=AgentStatus.PAUSED,
                    result="Paused by user",
                    iterations=state.iteration,
                    tool_calls=len(state.tool_calls),
                    colleague_calls=len(state.colleague_calls),
                )
            
            # Check budget before each LLM call
            try:
                from ..core import get_token_account, BudgetExceededError
                account = get_token_account()
                if account.is_budget_exceeded():
                    state.status = AgentStatus.COMPLETED
                    state.completed_at = datetime.now()
                    if self.transcript:
                        self.transcript.log_agent_message(
                            agent_name=self.agent_id,
                            message="Budget limit reached - wrapping up with current progress",
                        )
                    return AgentResponse(
                        status=AgentStatus.COMPLETED,
                        result="Budget limit reached - wrapping up with current progress",
                        iterations=state.iteration,
                        tool_calls=len(state.tool_calls),
                        colleague_calls=len(state.colleague_calls),
                    )
            except ImportError:
                pass  # Budget checking not available
            
            state.iteration += 1
            
            # Call LLM (with streaming if enabled)
            try:
                from ..core.token_context import token_context

                trigger = self._next_llm_trigger
                if not trigger:
                    trigger = "initial" if state.iteration == 1 else "react_loop"
                self._next_llm_trigger = None

                if self.streaming_enabled and hasattr(self.llm, 'acomplete_streaming'):
                    # Use streaming completion with chunk callback
                    def on_chunk(chunk: str) -> None:
                        if self.on_stream_chunk:
                            self.on_stream_chunk(self.agent_id, chunk)

                    with token_context(trigger=trigger):
                        response = await self.llm.acomplete_streaming(
                            state.messages,
                            on_chunk=on_chunk,
                        )
                else:
                    with token_context(trigger=trigger):
                        response = await self.llm.acomplete(state.messages)
            except Exception as e:
                state.status = AgentStatus.FAILED
                state.completed_at = datetime.now()
                return AgentResponse(
                    status=AgentStatus.FAILED,
                    error=f"LLM error: {e}",
                    iterations=state.iteration,
                )
            
            # Append assistant response
            state.messages.append({"role": "assistant", "content": response})

            # Log narrative content (not the raw action blocks) to transcript so consultations are visible.
            if self.transcript:
                narrative = self._strip_action_blocks_for_transcript(response)
                if narrative:
                    self.transcript.log_agent_message(
                        agent_name=self.agent_id,
                        message=narrative,
                    )
            
            # Notify callback
            if self.on_message:
                self.on_message(self.agent_id, response)
            
            # Parse response for actions
            action_result = await self._parse_and_execute(response, state, project_name)
            
            if action_result["type"] == "done":
                state.status = AgentStatus.COMPLETED
                state.completed_at = datetime.now()
                return AgentResponse(
                    status=AgentStatus.COMPLETED,
                    result=action_result.get("result"),
                    outputs=action_result.get("outputs", []),
                    iterations=state.iteration,
                    tool_calls=len(state.tool_calls),
                    colleague_calls=len(state.colleague_calls),
                )
            
            elif action_result["type"] in {"tool", "tool_batch", "colleague", "colleague_batch", "action_batch"}:
                for msg in action_result.get("messages", []):
                    state.messages.append(msg)
            
            elif action_result["type"] == "extend":
                # Agent requested more iterations
                pass  # Continue loop
            
            elif action_result["type"] == "continue":
                # No action parsed, continue with follow-up
                state.messages.append({
                    "role": "user",
                    "content": "Please continue. If you're done, use the ```done``` block to indicate completion.",
                })
            
            # Periodic checkpoint
            if state.iteration % 10 == 0:
                self._checkpoint(state, project_name)
        
        # Max iterations reached
        state.status = AgentStatus.FAILED
        state.completed_at = datetime.now()
        return AgentResponse(
            status=AgentStatus.FAILED,
            error=f"Max iterations ({self.max_iterations}) reached",
            iterations=state.iteration,
            tool_calls=len(state.tool_calls),
            colleague_calls=len(state.colleague_calls),
        )
    
    async def _parse_and_execute(
        self,
        response: str,
        state: AgentState,
        project_name: str,
    ) -> dict[str, Any]:
        """
        Parse agent response and execute any actions.
        
        Returns dict with:
        - type: 'done', 'tool', 'colleague', 'extend', 'continue'
        - Additional fields based on type
        """
        matches = list(self.ACTION_BLOCK_PATTERN.finditer(response))
        if not matches:
            return {"type": "continue"}

        messages_to_append: list[dict[str, str]] = []
        tool_messages: list[dict[str, str]] = []
        colleague_messages: list[dict[str, str]] = []
        extend_requested = False

        for m in matches:
            block_type = m.group(1)
            payload = (m.group(2) or "").strip()

            if block_type == "extend":
                extend_requested = True
                continue

            if block_type == "done":
                try:
                    done_data = json.loads(payload) if payload else {}
                    return {
                        "type": "done",
                        "result": done_data.get("result"),
                        "outputs": done_data.get("outputs", []),
                    }
                except json.JSONDecodeError:
                    return {"type": "done", "result": payload}

            if block_type == "tool":
                try:
                    tool_data = json.loads(payload)
                except json.JSONDecodeError as e:
                    tool_messages.append({
                        "role": "user",
                        "content": f"Tool result:\nError parsing tool call JSON: {e}",
                    })
                    continue

                tool_name = tool_data.get("tool")
                action = tool_data.get("action")
                params = tool_data.get("params", {})

                if not isinstance(tool_name, str) or not isinstance(action, str):
                    tool_messages.append({
                        "role": "user",
                        "content": f"Tool result:\nError: tool call missing required string fields 'tool' and 'action'. Payload={tool_data}",
                    })
                    continue
                if params is None:
                    params = {}
                if not isinstance(params, dict):
                    tool_messages.append({
                        "role": "user",
                        "content": f"Tool result:\nError: 'params' must be an object/dict. tool={tool_name} action={action}",
                    })
                    continue

                result = await self._execute_tool(tool_name, action, params, state)
                # Attribute the *next* LLM call to processing this tool result.
                self._next_llm_trigger = f"after_tool:{tool_name}.{action}"
                tool_messages.append({
                    "role": "user",
                    "content": f"Tool result ({tool_name}.{action}):\n{result}",
                })
                continue

            if block_type == "colleague":
                try:
                    colleague_data = json.loads(payload)
                except json.JSONDecodeError as e:
                    colleague_messages.append({
                        "role": "user",
                        "content": f"Colleague result:\nError parsing colleague call JSON: {e}",
                    })
                    continue

                colleague_id = colleague_data.get("colleague")
                question = colleague_data.get("question")
                # Optional mode field: "quick", "focused" (default), or "detailed"
                mode = colleague_data.get("mode", "focused")
                if mode not in ("quick", "focused", "detailed"):
                    mode = "focused"  # Default to focused if invalid mode
                    
                if not isinstance(colleague_id, str) or not isinstance(question, str):
                    colleague_messages.append({
                        "role": "user",
                        "content": f"Colleague result:\nError: colleague call missing required string fields 'colleague' and 'question'. Payload={colleague_data}",
                    })
                    continue

                result = await self._consult_colleague(colleague_id, question, state, project_name, mode=mode)
                # Attribute the *next* LLM call to processing this colleague response.
                self._next_llm_trigger = f"after_colleague:{colleague_id}"
                colleague_messages.append({
                    "role": "user",
                    "content": f"Response from {colleague_id}:\n{result}",
                })
                continue

        if tool_messages and colleague_messages:
            messages_to_append = tool_messages + colleague_messages
            return {"type": "action_batch", "messages": messages_to_append}

        if tool_messages:
            return {"type": "tool_batch", "messages": tool_messages}

        if colleague_messages:
            return {"type": "colleague_batch", "messages": colleague_messages}

        if extend_requested:
            return {"type": "extend"}

        return {"type": "continue"}
    
    async def _execute_tool(
        self,
        tool_name: str,
        action: str,
        params: dict,
        state: AgentState,
    ) -> str:
        """Execute a tool and return result string."""
        # Check for interrupt before executing tool
        if self._interrupt_requested:
            return "[INTERRUPTED] Tool execution cancelled - exit requested"
        
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"
        
        tool = self.tools[tool_name]
        
        # Report activity
        self._report_activity(f"[{self.agent_id.upper()}] using {tool_name}.{action}")
        
        # Notify callback
        if self.on_tool_call:
            self.on_tool_call(tool_name, action, params)
        
        try:
            result: ToolOutput = await tool.execute(action, params)
            
            # Record tool call
            state.tool_calls.append({
                "tool": tool_name,
                "action": action,
                "params": params,
                "success": result.success,
                "timestamp": datetime.now().isoformat(),
            })
                        # Log to transcript
            if self.transcript:
                result_summary = result.data[:100] if result.data else ""
                self.transcript.log_tool_use(
                    agent_name=self.agent_id,
                    tool_name=tool_name,
                    action=action,
                    params=params,
                    success=result.success,
                    result_summary=result_summary,
                    error=result.error if not result.success else None,
                )
                        # Format result
            if result.success:
                if result.data:
                    return f"Success: {result.data}\n{json.dumps(result.model_dump(exclude={'success', 'error', 'data'}), indent=2)}"
                return f"Success\n{json.dumps(result.model_dump(exclude={'success', 'error'}), indent=2)}"
            else:
                return f"Error: {result.error}"
                
        except Exception as e:
            state.tool_calls.append({
                "tool": tool_name,
                "action": action,
                "params": params,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            return f"Error executing tool: {e}"
    
    async def _consult_colleague(
        self,
        colleague_id: str,
        question: str,
        state: AgentState,
        project_name: str,
        mode: str = "focused",
    ) -> str:
        """
        Consult a colleague agent with visible output to user.
        
        Args:
            colleague_id: ID of colleague to consult
            question: The question to ask
            state: Current agent state
            project_name: Current project name
            mode: Consultation mode - affects how colleague interprets the request:
                - "quick": Brief, direct answer (2-4 sentences). Use for simple questions.
                - "focused" (default): Targeted expert input (1-2 paragraphs). Standard consultation.
                - "detailed": Comprehensive response with full reasoning. Use sparingly.
        
        Returns:
            Colleague's response string
        """
        if colleague_id not in self.colleagues:
            return f"Error: Unknown colleague '{colleague_id}'"
        
        colleague = self.colleagues[colleague_id]
        
        # Report activity and make it visible to user
        self._report_activity(f"[{self.agent_id.upper()}] consulting [{colleague_id.upper()}]")
        
        # Visible output to user
        from ..utils import console
        console.agent_handoff(self.agent_id, colleague_id, f"Consulting on: {question[:60]}..." if len(question) > 60 else f"Consulting on: {question}")
        
        # Log consultation start to transcript
        if self.transcript:
            self.transcript.log_consultation_start(
                from_agent=self.agent_id,
                to_agent=colleague_id,
                question=question,
            )
        
        # Record colleague call
        state.colleague_calls.append({
            "colleague": colleague_id,
            "question": question,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Build consultation context based on mode
        mode_guidance = {
            "quick": (
                "⚡ QUICK CONSULTATION: You are being consulted for a brief, direct answer.\n"
                "Respond in 2-4 sentences. Give the key recommendation and essential rationale only.\n"
                "Do NOT provide comprehensive frameworks or exhaustive explanations.\n\n"
            ),
            "focused": (
                "🎯 FOCUSED CONSULTATION: You are being consulted for targeted expert input.\n"
                "Respond in 1-2 paragraphs. Provide your recommendation, key considerations, and any critical caveats.\n"
                "Be concise but thorough on the SPECIFIC question asked. The consulting agent will implement.\n\n"
            ),
            "detailed": (
                "📋 DETAILED CONSULTATION: A comprehensive response is requested.\n"
                "Provide full reasoning, alternatives considered, and detailed recommendations.\n"
                "This is an exception to normal brevity norms—thoroughness is explicitly needed here.\n\n"
            ),
        }
        
        consultation_prefix = mode_guidance.get(mode, mode_guidance["focused"])
        
        try:
            # Execute colleague with consultation context prepended
            response = await colleague.execute(
                task=f"{consultation_prefix}[Question from {self.agent_id}]: {question}",
                project_name=project_name,
                task_id=f"{state.task_id}_consult_{colleague_id}",
            )
            
            if response.status == AgentStatus.COMPLETED:
                result = response.result or "Colleague completed without explicit result"
                # Show brief summary of consultation result
                console.agent_response(colleague_id, result[:80] + "..." if len(result) > 80 else result)
                
                # Log consultation response to transcript
                if self.transcript:
                    self.transcript.log_consultation_response(
                        from_agent=self.agent_id,
                        to_agent=colleague_id,
                        response=result,
                    )
                
                return result
            else:
                return f"Colleague response (status={response.status.value}): {response.error or response.result}"
                
        except Exception as e:
            return f"Error consulting colleague: {e}"
    
    async def execute_task(
        self,
        task: str,
        context: Optional[dict] = None,
        project_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute a task and return results as a dictionary.
        
        This is a convenience wrapper around execute() that returns
        a structured dict for easier integration.
        
        Args:
            task: Task description
            context: Optional context dict to include
            project_name: Project name (defaults to 'default')
            
        Returns:
            Dict with 'response', 'status', 'error' keys
        """
        if not project_name:
            raise ValueError("project_name is required for execute_task")
        
        # Build full task with context
        full_task = task
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            full_task = f"## Additional Context\n{context_str}\n\n## Task\n{task}"
        
        response = await self.execute(
            task=full_task,
            project_name=project_name,
        )
        
        return {
            "response": response.result,
            "status": response.status.value,
            "error": response.error,
            "iterations": response.iterations,
            "tool_calls": response.tool_calls,
            "colleague_calls": response.colleague_calls,
        }
    
    def _checkpoint(self, state: AgentState, project_name: str) -> None:
        """Save checkpoint of current state."""
        state.last_checkpoint = datetime.now()
        
        checkpoint_dir = Path(f"Sandbox/{project_name}/.checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{state.task_id}.json"
        state.save(checkpoint_path)
    
    def get_state(self) -> Optional[AgentState]:
        """Get current execution state."""
        return self._current_state
    
    async def simple_query(self, query: str, context: str = "") -> str:
        """
        Simple one-shot query without ReAct loop.
        
        Useful for quick consultations. Strips any tool call blocks from output.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        if context:
            messages.append({"role": "user", "content": f"{context}\n\n{query}"})
        else:
            messages.append({"role": "user", "content": query})
        
        response = await self.llm.acomplete(messages)
        
        # Strip any tool call blocks from the response
        return self._clean_tool_blocks(response)
    
    def _clean_tool_blocks(self, text: str) -> str:
        """Remove tool call blocks from text for clean display."""
        import re
        cleaned = re.sub(r"```tool\s*.*?```", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"```colleague\s*.*?```", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"```done\s*.*?```", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"```extend\s*.*?```", "", cleaned, flags=re.DOTALL)
        # Clean up extra whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()
