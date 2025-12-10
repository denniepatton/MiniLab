"""
User Input Tool for agent-user communication.

Allows agents to pause execution and request user input.
"""

from __future__ import annotations

from typing import Any, Optional, Callable
from pydantic import Field

from .base import Tool, ToolInput, ToolOutput, ToolError


class AskInput(ToolInput):
    """Input for asking the user a question."""
    question: str = Field(..., description="The question to ask the user")
    context: Optional[str] = Field(None, description="Additional context for the question")
    choices: Optional[list[str]] = Field(None, description="If provided, user must select from these choices")
    default: Optional[str] = Field(None, description="Default value if user provides no input")


class ConfirmInput(ToolInput):
    """Input for asking the user for confirmation."""
    message: str = Field(..., description="The confirmation message")
    default: bool = Field(True, description="Default value if user provides no input")


class UserInputOutput(ToolOutput):
    """Output for user input operations."""
    response: Optional[str] = None
    confirmed: Optional[bool] = None


class UserInputTool(Tool):
    """
    User interaction tool for agent-user communication.
    
    Allows agents to pause and request user input at any time.
    This is the primary mechanism for agents to ask clarifying questions.
    """
    
    name = "user_input"
    description = "Ask the user questions or request confirmation"
    
    def __init__(
        self,
        agent_id: str,
        input_callback: Optional[Callable[[str, Optional[list[str]]], str]] = None,
        **kwargs
    ):
        """
        Initialize user input tool.
        
        Args:
            agent_id: The ID of the agent using this tool
            input_callback: Callback function to get user input
                           Signature: (prompt, choices) -> response
        """
        super().__init__(agent_id, **kwargs)
        self.input_callback = input_callback
    
    def set_input_callback(self, callback: Callable[[str, Optional[list[str]]], str]) -> None:
        """Set the input callback after initialization."""
        self.input_callback = callback
    
    def get_actions(self) -> dict[str, str]:
        return {
            "ask": "Ask the user a question and wait for response",
            "confirm": "Ask the user for yes/no confirmation",
        }
    
    def get_input_schema(self, action: str) -> type[ToolInput]:
        schemas = {
            "ask": AskInput,
            "confirm": ConfirmInput,
        }
        if action not in schemas:
            raise ToolError(self.name, action, f"Unknown action: {action}")
        return schemas[action]
    
    async def execute(self, action: str, params: dict[str, Any]) -> UserInputOutput:
        """Execute a user input action."""
        validated = self.validate_input(action, params)
        
        try:
            if action == "ask":
                return await self._ask(validated)
            elif action == "confirm":
                return await self._confirm(validated)
            else:
                raise ToolError(self.name, action, f"Unknown action: {action}")
        except Exception as e:
            return UserInputOutput(success=False, error=f"Failed to get user input: {e}")
    
    async def _ask(self, params: AskInput) -> UserInputOutput:
        """Ask the user a question."""
        if not self.input_callback:
            return UserInputOutput(
                success=False,
                error="No input callback configured - cannot interact with user"
            )
        
        # Build prompt
        prompt_parts = [f"[{self.agent_id.upper()}]: {params.question}"]
        
        if params.context:
            prompt_parts.append(f"\nContext: {params.context}")
        
        if params.choices:
            prompt_parts.append(f"\nOptions: {', '.join(params.choices)}")
        
        if params.default:
            prompt_parts.append(f"\n(Default: {params.default})")
        
        prompt = "".join(prompt_parts)
        
        try:
            response = self.input_callback(prompt, params.choices)
            
            # Use default if no response
            if not response and params.default:
                response = params.default
            
            # Validate against choices if provided
            if params.choices and response:
                # Case-insensitive matching
                matches = [c for c in params.choices if c.lower() == response.lower()]
                if matches:
                    response = matches[0]
                else:
                    return UserInputOutput(
                        success=False,
                        error=f"Invalid choice. Please select from: {', '.join(params.choices)}"
                    )
            
            return UserInputOutput(success=True, response=response)
            
        except KeyboardInterrupt:
            return UserInputOutput(
                success=False,
                error="User interrupted input"
            )
    
    async def _confirm(self, params: ConfirmInput) -> UserInputOutput:
        """Ask the user for confirmation."""
        if not self.input_callback:
            return UserInputOutput(
                success=False,
                error="No input callback configured - cannot interact with user"
            )
        
        default_str = "Y/n" if params.default else "y/N"
        prompt = f"[{self.agent_id.upper()}]: {params.message} [{default_str}]"
        
        try:
            response = self.input_callback(prompt, None)
            
            if not response:
                confirmed = params.default
            else:
                confirmed = response.lower() in ("y", "yes", "true", "1")
            
            return UserInputOutput(success=True, confirmed=confirmed, response=response)
            
        except KeyboardInterrupt:
            return UserInputOutput(
                success=False,
                confirmed=False,
                error="User interrupted input"
            )
