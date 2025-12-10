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
    - Persistent state per task
    - Interrupt/pause capabilities
    - Typed tool integration
    """
    
    # Regex patterns for parsing agent output
    TOOL_PATTERN = re.compile(r"```tool\s*(.*?)\s*```", re.DOTALL)
    COLLEAGUE_PATTERN = re.compile(r"```colleague\s*(.*?)\s*```", re.DOTALL)
    DONE_PATTERN = re.compile(r"```done\s*(.*?)\s*```", re.DOTALL)
    EXTEND_PATTERN = re.compile(r"```extend\s*(.*?)\s*```", re.DOTALL)
    
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
    ):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique agent identifier
            name: Display name
            guild: Guild membership
            system_prompt: SOTA-formatted system prompt
            llm_backend: LLM backend for completions
            tools: Dict of tool name to Tool instance
            context_manager: Context manager for RAG
            max_iterations: Max ReAct iterations
        """
        self.agent_id = agent_id
        self.name = name
        self.guild = guild
        self.system_prompt = system_prompt
        self.llm = llm_backend
        self.tools = tools
        self.context_manager = context_manager
        self.max_iterations = max_iterations
        
        # Colleagues (set by registry)
        self.colleagues: dict[str, Agent] = {}
        
        # State management
        self._current_state: Optional[AgentState] = None
        self._interrupt_requested = False
        self._pause_requested = False
        
        # Callbacks
        self.on_message: Optional[Callable[[str, str], None]] = None  # (agent_id, message)
        self.on_tool_call: Optional[Callable[[str, str, dict], None]] = None  # (tool, action, params)
    
    def set_colleagues(self, colleagues: dict[str, Agent]) -> None:
        """Set colleague references."""
        self.colleagues = colleagues
    
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
    ) -> AgentResponse:
        """
        Execute a task using the ReAct loop.
        
        Args:
            task: Task description/prompt
            project_name: Current project
            task_id: Unique task identifier (generated if not provided)
            resume_state: State to resume from (if paused previously)
            
        Returns:
            AgentResponse with results
        """
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
        
        # Initialize messages if starting fresh
        if not state.messages:
            state.messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"## Context\n{context.to_prompt()}\n\n## Task\n{task}"},
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
            
            state.iteration += 1
            
            # Call LLM
            try:
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
            
            elif action_result["type"] == "tool":
                # Append tool result to messages
                state.messages.append({
                    "role": "user",
                    "content": f"Tool result:\n{action_result['result']}",
                })
            
            elif action_result["type"] == "colleague":
                # Append colleague response to messages
                state.messages.append({
                    "role": "user",
                    "content": f"Response from {action_result['colleague']}:\n{action_result['result']}",
                })
            
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
        # Check for done
        done_match = self.DONE_PATTERN.search(response)
        if done_match:
            try:
                done_data = json.loads(done_match.group(1))
                return {
                    "type": "done",
                    "result": done_data.get("result"),
                    "outputs": done_data.get("outputs", []),
                }
            except json.JSONDecodeError:
                return {"type": "done", "result": done_match.group(1)}
        
        # Check for tool call
        tool_match = self.TOOL_PATTERN.search(response)
        if tool_match:
            try:
                tool_data = json.loads(tool_match.group(1))
                tool_name = tool_data.get("tool")
                action = tool_data.get("action")
                params = tool_data.get("params", {})
                
                result = await self._execute_tool(tool_name, action, params, state)
                return {"type": "tool", "result": result}
            except json.JSONDecodeError as e:
                return {"type": "tool", "result": f"Error parsing tool call: {e}"}
        
        # Check for colleague call
        colleague_match = self.COLLEAGUE_PATTERN.search(response)
        if colleague_match:
            try:
                colleague_data = json.loads(colleague_match.group(1))
                colleague_id = colleague_data.get("colleague")
                question = colleague_data.get("question")
                
                result = await self._consult_colleague(
                    colleague_id, question, state, project_name
                )
                return {"type": "colleague", "colleague": colleague_id, "result": result}
            except json.JSONDecodeError as e:
                return {"type": "colleague", "result": f"Error parsing colleague call: {e}"}
        
        # Check for extend request
        extend_match = self.EXTEND_PATTERN.search(response)
        if extend_match:
            return {"type": "extend"}
        
        # No action found
        return {"type": "continue"}
    
    async def _execute_tool(
        self,
        tool_name: str,
        action: str,
        params: dict,
        state: AgentState,
    ) -> str:
        """Execute a tool and return result string."""
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"
        
        tool = self.tools[tool_name]
        
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
    ) -> str:
        """Consult a colleague agent."""
        if colleague_id not in self.colleagues:
            return f"Error: Unknown colleague '{colleague_id}'"
        
        colleague = self.colleagues[colleague_id]
        
        # Record colleague call
        state.colleague_calls.append({
            "colleague": colleague_id,
            "question": question,
            "timestamp": datetime.now().isoformat(),
        })
        
        try:
            # Execute colleague with limited iterations
            response = await colleague.execute(
                task=f"[Question from {self.agent_id}]: {question}",
                project_name=project_name,
                task_id=f"{state.task_id}_consult_{colleague_id}",
            )
            
            if response.status == AgentStatus.COMPLETED:
                return response.result or "Colleague completed without explicit result"
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
        
        Useful for quick consultations.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        if context:
            messages.append({"role": "user", "content": f"{context}\n\n{query}"})
        else:
            messages.append({"role": "user", "content": query})
        
        return await self.llm.acomplete(messages)
