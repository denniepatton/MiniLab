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


def _pause_spinner() -> bool:
    """Pause the global spinner if running. Returns True if was running."""
    try:
        from ..utils import Spinner
        return Spinner.pause_for_input()
    except Exception:
        return False


def _resume_spinner() -> None:
    """Resume the global spinner if it was paused."""
    try:
        from ..utils import Spinner
        Spinner.resume_after_input()
    except Exception:
        pass


class UserInputTool(Tool):
    """
    User interaction tool for agent-user communication.
    
    Allows agents to pause and request user input at any time.
    This is the primary mechanism for agents to ask clarifying questions.
    
    The tool respects user preferences passed via context:
    - If user indicated autonomous operation (e.g., "use your best judgment",
      "don't ask me", "complete without consulting"), the tool will auto-proceed
      with defaults or reasonable choices instead of blocking for input.
    """
    
    name = "user_input"
    description = "Ask the user questions or request confirmation"
    
    # Phrases that indicate user wants autonomous operation
    AUTONOMOUS_INDICATORS = [
        "best judgment", "best judgement", "don't ask", "do not ask",
        "without consulting", "autonomous", "independently", "proceed without",
        "don't need to ask", "your discretion", "make the call",
        "complete without", "finish without", "on your own",
        "use your judgment", "use your judgement", "figure it out",
    ]
    
    def __init__(
        self,
        agent_id: str,
        input_callback: Optional[Callable[[str, Optional[list[str]]], str]] = None,
        user_preferences: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize user input tool.
        
        Args:
            agent_id: The ID of the agent using this tool
            input_callback: Callback function to get user input
                           Signature: (prompt, choices) -> response
            user_preferences: Natural language user preferences from consultation
        """
        super().__init__(agent_id, **kwargs)
        self.input_callback = input_callback
        self._user_preferences = user_preferences or ""
    
    def set_user_preferences(self, preferences: str) -> None:
        """Set user preferences (from consultation output)."""
        self._user_preferences = preferences or ""
    
    def _should_auto_proceed(self) -> bool:
        """
        Check if user preferences indicate autonomous operation.
        
        This is NOT a binary flag - it's contextual interpretation of
        natural language preferences.
        """
        if not self._user_preferences:
            return False
        
        prefs_lower = self._user_preferences.lower()
        return any(indicator in prefs_lower for indicator in self.AUTONOMOUS_INDICATORS)
    
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
        # Check if user preferences indicate autonomous operation
        if self._should_auto_proceed():
            # Auto-proceed with default or first choice
            auto_response = params.default
            if not auto_response and params.choices:
                auto_response = params.choices[0]  # Pick first reasonable option
            if not auto_response:
                auto_response = "Proceeding with best judgment"
            
            return UserInputOutput(
                success=True,
                response=auto_response,
            )
        
        if not self.input_callback:
            return UserInputOutput(
                success=False,
                error="No input callback configured - cannot interact with user"
            )
        
        # Build prompt
        prompt_parts = [f"{params.question}"]
        
        if params.context:
            prompt_parts.append(f"\nContext: {params.context}")
        
        if params.choices:
            prompt_parts.append(f"\nOptions: {', '.join(params.choices)}")
        
        if params.default:
            prompt_parts.append(f"\n(Default: {params.default})")
        
        prompt = "".join(prompt_parts)
        
        # Pause spinner for user input
        was_spinning = _pause_spinner()
        
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
        finally:
            # Resume spinner
            if was_spinning:
                _resume_spinner()
    
    async def _confirm(self, params: ConfirmInput) -> UserInputOutput:
        """Ask the user for confirmation with plain English prompts."""
        # Check if user preferences indicate autonomous operation
        if self._should_auto_proceed():
            # Auto-confirm based on default
            return UserInputOutput(
                success=True,
                confirmed=params.default,
                response="yes" if params.default else "no",
            )
        
        if not self.input_callback:
            return UserInputOutput(
                success=False,
                error="No input callback configured - cannot interact with user"
            )
        
        # Build a clear, plain English prompt
        default_hint = "yes" if params.default else "no"
        prompt = f"{params.message}\n\nPlease respond with 'yes' or 'no' (default: {default_hint})"
        
        # Pause spinner for user input
        was_spinning = _pause_spinner()
        
        try:
            response = self.input_callback(prompt, None)
            
            if not response:
                confirmed = params.default
            else:
                response_lower = response.lower().strip()
                confirmed = response_lower in ("y", "yes", "true", "1", "ok", "sure", "proceed", "continue", "approve")
            
            return UserInputOutput(success=True, confirmed=confirmed, response=response)
            
        except KeyboardInterrupt:
            return UserInputOutput(
                success=False,
                confirmed=False,
                error="User interrupted input"
            )
        finally:
            # Resume spinner
            if was_spinning:
                _resume_spinner()
