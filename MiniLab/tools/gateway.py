"""
ToolGateway: Unified tool dispatch and execution layer.

Provides:
- Centralized tool registration and discovery
- Permission checking before execution
- Output sanitization and truncation
- Execution logging for audit trail
- MCP adapter integration point
"""

from __future__ import annotations

__all__ = [
    "ToolScope",
    "ToolPermission",
    "ToolCall",
    "ToolCallLogger",
    "OutputSanitizer",
    "DefaultSanitizer",
    "ToolRegistry",
    "ToolGateway",
    "SkillPack",
]

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Protocol

from pydantic import BaseModel, Field

from MiniLab.tools.base import Tool, ToolOutput, ToolError


class ToolScope(str, Enum):
    """Security scopes for tools."""
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    RUN_COMMAND = "run_command"
    NETWORK = "network"
    USER_INPUT = "user_input"
    SYSTEM = "system"


class ToolPermission(BaseModel):
    """Permission configuration for a tool."""

    tool_name: str = Field(..., description="Name of the tool")
    scopes: list[ToolScope] = Field(default_factory=list, description="Required scopes")
    requires_approval: bool = Field(default=False, description="Requires user approval")
    max_calls_per_task: Optional[int] = Field(default=None, description="Rate limit per task")

    model_config = {"extra": "forbid"}


@dataclass
class ToolCall:
    """Record of a tool invocation."""

    call_id: str
    tool_name: str
    action: str
    agent_id: str
    task_id: Optional[str]

    # Input/output
    arguments: dict[str, Any]
    result: Optional[ToolOutput] = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Status
    success: bool = False
    error: Optional[str] = None

    # Metrics
    output_chars: int = 0
    execution_time_ms: int = 0


class ToolCallLogger(Protocol):
    """Protocol for logging tool calls."""

    def log_call(self, call: ToolCall) -> None:
        """Log a tool call."""
        ...


class OutputSanitizer(Protocol):
    """Protocol for sanitizing tool outputs."""

    def sanitize(self, output: str, max_chars: int) -> str:
        """Sanitize and truncate output."""
        ...


class DefaultSanitizer:
    """Default output sanitizer with truncation."""

    def sanitize(self, output: str, max_chars: int) -> str:
        """Sanitize and truncate output."""
        if len(output) <= max_chars:
            return output

        # Keep head and tail
        head_size = int(max_chars * 0.7)
        tail_size = max_chars - head_size - 50

        return (
            output[:head_size] +
            f"\n\n[...truncated {len(output) - max_chars:,} chars...]\n\n" +
            output[-tail_size:]
        )


class ToolRegistry:
    """
    Registry of available tools.
    
    Manages tool registration, discovery, and metadata.
    """

    def __init__(self):
        self._tools: dict[str, type[Tool]] = {}
        self._permissions: dict[str, ToolPermission] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        tool_class: type[Tool],
        permission: Optional[ToolPermission] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Register a tool class.
        
        Args:
            tool_class: The Tool class to register
            permission: Optional permission configuration
            metadata: Optional additional metadata
        """
        name = tool_class.name
        self._tools[name] = tool_class

        if permission:
            self._permissions[name] = permission

        if metadata:
            self._metadata[name] = metadata

    def get(self, name: str) -> Optional[type[Tool]]:
        """Get a registered tool class by name."""
        return self._tools.get(name)

    def get_permission(self, name: str) -> Optional[ToolPermission]:
        """Get permission config for a tool."""
        return self._permissions.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_schema(self, name: str) -> Optional[dict[str, Any]]:
        """Get the schema for a tool (for LLM function calling)."""
        tool_class = self._tools.get(name)
        if not tool_class:
            return None

        # Instantiate temporarily to get actions
        # This is a bit awkward but maintains compatibility
        return {
            "name": name,
            "description": tool_class.description,
        }


class ToolGateway:
    """
    Central gateway for all tool execution.
    
    Responsibilities:
    - Permission checking before execution
    - Rate limiting per task
    - Output sanitization
    - Execution logging
    - Error handling and wrapping
    
    Example:
        gateway = ToolGateway(registry, policy)
        result = await gateway.execute(
            tool_name="filesystem",
            action="read_file",
            arguments={"path": "/data/input.csv"},
            agent_id="analyst",
            task_id="task_1",
        )
    """

    def __init__(
        self,
        registry: ToolRegistry,
        allowed_scopes: Optional[set[ToolScope]] = None,
        max_output_chars: int = 50_000,
        sanitizer: Optional[OutputSanitizer] = None,
        logger: Optional[ToolCallLogger] = None,
        approval_callback: Optional[Callable[[str, str, dict], bool]] = None,
    ):
        """
        Initialize the gateway.
        
        Args:
            registry: Tool registry
            allowed_scopes: Set of allowed security scopes (deny-by-default)
            max_output_chars: Maximum output size before truncation
            sanitizer: Optional custom sanitizer
            logger: Optional call logger
            approval_callback: Callback for user approval prompts
        """
        self.registry = registry
        self.allowed_scopes = allowed_scopes or set()
        self.max_output_chars = max_output_chars
        self.sanitizer = sanitizer or DefaultSanitizer()
        self.logger = logger
        self.approval_callback = approval_callback

        # Tool instances by agent
        self._instances: dict[str, dict[str, Tool]] = {}

        # Rate limiting
        self._call_counts: dict[str, dict[str, int]] = {}  # task_id -> tool_name -> count

        # Call ID counter
        self._call_counter = 0

    def _get_instance(self, tool_name: str, agent_id: str) -> Tool:
        """Get or create a tool instance for an agent."""
        if agent_id not in self._instances:
            self._instances[agent_id] = {}

        if tool_name not in self._instances[agent_id]:
            tool_class = self.registry.get(tool_name)
            if not tool_class:
                raise ToolError(tool_name, "init", f"Unknown tool: {tool_name}")

            self._instances[agent_id][tool_name] = tool_class(
                agent_id=agent_id,
                permission_callback=self.approval_callback,
            )

        return self._instances[agent_id][tool_name]

    def _check_permission(
        self,
        tool_name: str,
        action: str,
        agent_id: str,
        task_id: Optional[str],
    ) -> tuple[bool, str]:
        """
        Check if execution is permitted.
        
        Returns:
            Tuple of (allowed, reason)
        """
        permission = self.registry.get_permission(tool_name)

        if permission:
            # Check scopes
            for scope in permission.scopes:
                if scope not in self.allowed_scopes:
                    return False, f"Scope '{scope.value}' not allowed"

            # Check rate limit
            if permission.max_calls_per_task and task_id:
                current = self._call_counts.get(task_id, {}).get(tool_name, 0)
                if current >= permission.max_calls_per_task:
                    return False, f"Rate limit exceeded ({current}/{permission.max_calls_per_task})"

            # Check approval
            if permission.requires_approval:
                if not self.approval_callback:
                    return False, "Approval required but no callback configured"
                # Approval is checked during execution

        return True, ""

    async def execute(
        self,
        tool_name: str,
        action: str,
        arguments: dict[str, Any],
        agent_id: str,
        task_id: Optional[str] = None,
    ) -> ToolOutput:
        """
        Execute a tool action.
        
        Args:
            tool_name: Name of the tool
            action: Action to execute
            arguments: Action arguments
            agent_id: ID of the calling agent
            task_id: Optional task context
            
        Returns:
            ToolOutput with result or error
        """
        # Generate call ID
        self._call_counter += 1
        call_id = f"{tool_name}_{action}_{self._call_counter}"

        # Create call record
        call = ToolCall(
            call_id=call_id,
            tool_name=tool_name,
            action=action,
            agent_id=agent_id,
            task_id=task_id,
            arguments=arguments,
        )

        try:
            # Check permission
            allowed, reason = self._check_permission(tool_name, action, agent_id, task_id)
            if not allowed:
                call.error = f"Permission denied: {reason}"
                return ToolOutput(success=False, error=call.error)

            # Get tool instance
            tool = self._get_instance(tool_name, agent_id)

            # Execute
            start = datetime.now()
            result = await tool.execute(action, arguments)
            end = datetime.now()

            # Update call record
            call.completed_at = end
            call.execution_time_ms = int((end - start).total_seconds() * 1000)
            call.success = result.success
            call.result = result

            # Sanitize output
            if result.data and isinstance(result.data, str):
                result.data = self.sanitizer.sanitize(result.data, self.max_output_chars)
                call.output_chars = len(result.data)

            # Update rate limit counter
            if task_id:
                if task_id not in self._call_counts:
                    self._call_counts[task_id] = {}
                self._call_counts[task_id][tool_name] = (
                    self._call_counts[task_id].get(tool_name, 0) + 1
                )

            return result

        except ToolError as e:
            call.error = str(e)
            call.success = False
            return ToolOutput(success=False, error=str(e))

        except Exception as e:
            call.error = f"Unexpected error: {e}"
            call.success = False
            return ToolOutput(success=False, error=call.error)

        finally:
            # Log the call
            if self.logger:
                self.logger.log_call(call)

    def get_available_tools(self, agent_id: str) -> list[dict[str, Any]]:
        """
        Get list of available tools with schemas for an agent.
        
        Args:
            agent_id: The agent requesting tools
            
        Returns:
            List of tool schemas for LLM function calling
        """
        schemas = []

        for tool_name in self.registry.list_tools():
            schema = self.registry.get_schema(tool_name)
            if schema:
                schemas.append(schema)

        return schemas

    def clear_task_limits(self, task_id: str) -> None:
        """Clear rate limit counters for a completed task."""
        if task_id in self._call_counts:
            del self._call_counts[task_id]


class SkillPack(BaseModel):
    """
    A collection of related tools bundled together.
    
    Skill packs group tools by domain (e.g., "data_analysis", "literature")
    and can be enabled/disabled as a unit.
    """

    name: str = Field(..., description="Pack name")
    description: str = Field(default="", description="What this pack provides")
    tools: list[str] = Field(default_factory=list, description="Tool names in pack")
    default_enabled: bool = Field(default=True, description="Enabled by default")

    model_config = {"extra": "forbid"}


# Pre-defined skill packs
SKILL_PACKS = {
    "filesystem": SkillPack(
        name="filesystem",
        description="File system operations (read, write, list)",
        tools=["filesystem"],
    ),
    "terminal": SkillPack(
        name="terminal",
        description="Shell command execution",
        tools=["terminal"],
    ),
    "literature": SkillPack(
        name="literature",
        description="Literature search and citation management",
        tools=["pubmed", "arxiv", "citation"],
    ),
    "web": SkillPack(
        name="web",
        description="Web search and retrieval",
        tools=["web_search"],
    ),
    "code": SkillPack(
        name="code",
        description="Code editing and environment management",
        tools=["code_editor", "environment"],
    ),
    "interaction": SkillPack(
        name="interaction",
        description="User interaction and input",
        tools=["user_input"],
    ),
}
