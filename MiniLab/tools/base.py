"""
Base classes for the typed tool system.

All tools inherit from Tool and use Pydantic models for input/output validation.

Now includes VS Code-style two-phase execution:
1. prepare() - Validate, estimate cost, request confirmation if needed
2. execute() - Actually perform the operation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypeVar, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .prepared_invocation import PreparedInvocation
    from .response_stream import ResponseStream


class ToolInput(BaseModel):
    """Base class for tool input parameters."""
    
    class Config:
        extra = "forbid"  # Reject unexpected fields


class ToolOutput(BaseModel):
    """Base class for tool output."""
    success: bool = True
    error: Optional[str] = None
    data: Any = None
    
    # Added: metadata for tracking
    estimated_tokens: int = 0
    actual_tokens: int = 0
    from_cache: bool = False
    
    class Config:
        extra = "allow"


class ToolError(Exception):
    """Exception raised when a tool operation fails."""
    
    def __init__(self, tool_name: str, action: str, message: str):
        self.tool_name = tool_name
        self.action = action
        self.message = message
        super().__init__(f"[{tool_name}.{action}] {message}")


T_Input = TypeVar("T_Input", bound=ToolInput)
T_Output = TypeVar("T_Output", bound=ToolOutput)


class Tool(ABC):
    """
    Abstract base class for all MiniLab tools.
    
    Tools are typed functions with explicit input/output schemas,
    integrated security via PathGuard, and agent-aware execution.
    
    Now supports VS Code-style two-phase execution:
    1. prepare() - Pre-flight validation, cost estimation, confirmation
    2. execute() - Actual execution
    """
    
    name: str
    description: str
    
    def __init__(
        self,
        agent_id: str,
        permission_callback: Optional[Callable[[str], bool]] = None,
        response_stream: Optional[ResponseStream] = None,
        **_unused: Any,
    ):
        """
        Initialize tool with agent context.
        
        Args:
            agent_id: The ID of the agent using this tool
            permission_callback: Optional callback for permission prompts
            response_stream: Optional stream for progress reporting
        """
        self.agent_id = agent_id
        self.permission_callback = permission_callback
        self.response_stream = response_stream
    
    def set_response_stream(self, stream: ResponseStream) -> None:
        """Set the response stream for progress reporting."""
        self.response_stream = stream
    
    @abstractmethod
    def get_actions(self) -> dict[str, str]:
        """
        Return available actions and their descriptions.
        
        Returns:
            Dict mapping action names to descriptions
        """
        pass
    
    @abstractmethod
    def get_input_schema(self, action: str) -> type[ToolInput]:
        """
        Get the Pydantic model for input validation.
        
        Args:
            action: The action name
            
        Returns:
            Pydantic model class for input validation
        """
        pass
    
    async def prepare(
        self,
        action: str,
        params: dict[str, Any],
    ) -> PreparedInvocation:
        """
        Prepare a tool invocation - VS Code style pre-flight check.
        
        Override this method to:
        - Validate parameters beyond schema
        - Estimate token/time cost
        - Request user confirmation for destructive actions
        - Check cache for existing results
        - Preview what will happen
        
        The default implementation does basic validation and returns
        a valid PreparedInvocation with no confirmation required.
        
        Args:
            action: The action to prepare
            params: Parameters dict
            
        Returns:
            PreparedInvocation with validation results and confirmation requirements
        """
        from .prepared_invocation import (
            PreparedInvocation,
            ConfirmationLevel,
            get_default_confirmation_level,
        )
        
        # Validate input schema
        try:
            self.validate_input(action, params)
        except ToolError as e:
            return PreparedInvocation.invalid([str(e)])
        
        # Check if action requires confirmation
        confirmation_level = get_default_confirmation_level(self.name, action)
        
        if confirmation_level == ConfirmationLevel.DESTRUCTIVE:
            return PreparedInvocation.needs_confirmation(
                title=f"Confirm {self.name}.{action}",
                message=f"This action may have irreversible effects.",
                level=confirmation_level,
                details=f"Parameters: {params}",
            )
        
        return PreparedInvocation.valid(
            preview=f"{self.name}.{action} with {len(params)} parameters",
        )
    
    @abstractmethod
    async def execute(self, action: str, params: dict[str, Any]) -> ToolOutput:
        """
        Execute a tool action with validated parameters.
        
        Args:
            action: The action to perform
            params: Parameters dict (will be validated against input schema)
            
        Returns:
            ToolOutput with results
        """
        pass
    
    async def prepare_and_execute(
        self,
        action: str,
        params: dict[str, Any],
        skip_confirmation: bool = False,
    ) -> tuple[PreparedInvocation, Optional[ToolOutput]]:
        """
        Two-phase execution: prepare then execute.
        
        Args:
            action: The action to perform
            params: Parameters dict
            skip_confirmation: Skip confirmation even if required
            
        Returns:
            Tuple of (PreparedInvocation, ToolOutput or None if not executed)
        """
        # Phase 1: Prepare
        prepared = await self.prepare(action, params)
        
        # Check if preparation failed
        if not prepared.is_valid:
            return prepared, ToolOutput(
                success=False,
                error=f"Validation failed: {', '.join(prepared.validation_errors)}",
            )
        
        # Check if we have a cached result
        if prepared.has_result:
            return prepared, prepared.cached_result
        
        # Check if confirmation is required
        if prepared.requires_confirmation and not skip_confirmation:
            if self.permission_callback:
                message = prepared.confirmation.message if prepared.confirmation else "Confirm action?"
                confirmed = self.permission_callback(message)
                if not confirmed:
                    return prepared, ToolOutput(
                        success=False,
                        error="User cancelled the operation",
                    )
            else:
                # No callback, but confirmation required - default deny
                return prepared, ToolOutput(
                    success=False,
                    error="Confirmation required but no callback available",
                )
        
        # Phase 2: Execute
        result = await self.execute(action, params)
        result.estimated_tokens = prepared.estimated_input_tokens + prepared.estimated_output_tokens
        
        return prepared, result
    
    def validate_input(self, action: str, params: dict[str, Any]) -> ToolInput:
        """
        Validate input parameters against the schema.
        
        Args:
            action: The action name
            params: Parameters to validate
            
        Returns:
            Validated ToolInput instance
            
        Raises:
            ToolError: If validation fails
        """
        schema = self.get_input_schema(action)
        try:
            return schema(**params)
        except Exception as e:
            raise ToolError(self.name, action, f"Invalid parameters: {e}")
    
    def format_for_prompt(self) -> str:
        """
        Format tool documentation for inclusion in agent prompts.
        
        Returns:
            Markdown-formatted tool documentation
        """
        lines = [f"### {self.name}", "", self.description, "", "**Actions:**", ""]
        
        for action, desc in self.get_actions().items():
            schema = self.get_input_schema(action)
            lines.append(f"- **{action}**: {desc}")
            
            # Add parameter documentation from schema
            if schema.__fields__:
                lines.append("  Parameters:")
                for field_name, field_info in schema.__fields__.items():
                    field_type = field_info.annotation.__name__ if hasattr(field_info.annotation, '__name__') else str(field_info.annotation)
                    required = field_info.is_required()
                    default = f" (default: {field_info.default})" if not required else ""
                    desc_text = field_info.description or ""
                    lines.append(f"    - `{field_name}` ({field_type}{'*' if required else ''}{default}): {desc_text}")
            lines.append("")
        
        return "\n".join(lines)


class ToolRegistry:
    """
    Registry for managing tool instances per agent.
    
    Provides centralized access to tools with agent-specific configurations.
    """
    
    def __init__(self):
        self._tools: dict[str, dict[str, Tool]] = {}  # agent_id -> tool_name -> Tool
    
    def register(self, agent_id: str, tool: Tool) -> None:
        """Register a tool for an agent."""
        if agent_id not in self._tools:
            self._tools[agent_id] = {}
        self._tools[agent_id][tool.name] = tool
    
    def get_tool(self, agent_id: str, tool_name: str) -> Optional[Tool]:
        """Get a specific tool for an agent."""
        return self._tools.get(agent_id, {}).get(tool_name)
    
    def get_agent_tools(self, agent_id: str) -> dict[str, Tool]:
        """Get all tools registered for an agent."""
        return self._tools.get(agent_id, {})
    
    def get_tool_names(self, agent_id: str) -> list[str]:
        """Get names of all tools available to an agent."""
        return list(self._tools.get(agent_id, {}).keys())
    
    def format_tools_for_prompt(self, agent_id: str) -> str:
        """Format all agent tools for inclusion in prompts."""
        tools = self.get_agent_tools(agent_id)
        if not tools:
            return "No tools available."
        
        sections = []
        for tool in tools.values():
            sections.append(tool.format_for_prompt())
        
        return "\n\n".join(sections)
