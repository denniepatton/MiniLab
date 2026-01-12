"""
Permission Tool.

Provides user confirmation and permission prompts:
- permission.confirm: Request user confirmation
- permission.approve: Request approval for destructive action

Integrates with the two-phase execution pattern.
"""

from __future__ import annotations

from typing import Any, Optional, Callable
from pydantic import Field

from .base import Tool, ToolInput, ToolOutput, ToolError
from ..utils import console


class ConfirmInput(ToolInput):
    """Input for confirmation prompt."""
    message: str = Field(..., description="Message to display to user")
    default: bool = Field(True, description="Default response if user presses enter")
    timeout: Optional[int] = Field(None, description="Timeout in seconds (None for no timeout)")


class ApproveInput(ToolInput):
    """Input for approval prompt."""
    action: str = Field(..., description="Description of the action requiring approval")
    reason: str = Field("", description="Why this action is needed")
    impact: str = Field("", description="What will happen if approved")
    reversible: bool = Field(True, description="Whether the action can be undone")


class PermissionOutput(ToolOutput):
    """Output for permission operations."""
    approved: bool = False
    user_response: Optional[str] = None


class PermissionTool(Tool):
    """
    Tool for requesting user confirmation and approval.
    
    Used for:
    - Confirming destructive operations
    - Getting approval for external actions
    - User decision points in modules
    """
    
    name = "permission"
    description = "Request user confirmation and approval"
    
    def get_actions(self) -> dict[str, str]:
        return {
            "confirm": "Request yes/no confirmation from user",
            "approve": "Request approval for a specific action",
        }
    
    def get_input_schema(self, action: str) -> type[ToolInput]:
        schemas = {
            "confirm": ConfirmInput,
            "approve": ApproveInput,
        }
        return schemas.get(action, ToolInput)
    
    async def execute(self, action: str, params: dict[str, Any]) -> ToolOutput:
        """Execute a permission action."""
        schema = self.get_input_schema(action)
        validated = schema(**params)
        
        try:
            if action == "confirm":
                return await self._confirm(validated)
            elif action == "approve":
                return await self._approve(validated)
            else:
                return PermissionOutput(
                    success=False,
                    error=f"Unknown action: {action}",
                )
        except Exception as e:
            return PermissionOutput(success=False, error=str(e))
    
    async def _confirm(self, params: ConfirmInput) -> PermissionOutput:
        """Request yes/no confirmation."""
        default_str = "Y/n" if params.default else "y/N"
        prompt = f"{params.message} [{default_str}]: "
        
        # Use permission callback if available
        if self.permission_callback:
            approved = self.permission_callback(params.message)
            return PermissionOutput(
                success=True,
                approved=approved,
                user_response="yes" if approved else "no",
            )
        
        # Otherwise use console input
        try:
            console.print(f"\n[bold yellow]CONFIRMATION REQUIRED[/bold yellow]")
            response = console.input(prompt).strip().lower()
            
            if not response:
                approved = params.default
            elif response in ("y", "yes", "true", "1"):
                approved = True
            elif response in ("n", "no", "false", "0"):
                approved = False
            else:
                approved = params.default
            
            return PermissionOutput(
                success=True,
                approved=approved,
                user_response=response,
            )
            
        except (EOFError, KeyboardInterrupt):
            return PermissionOutput(
                success=True,
                approved=False,
                user_response="cancelled",
            )
    
    async def _approve(self, params: ApproveInput) -> PermissionOutput:
        """Request approval for an action."""
        # Build detailed approval message
        lines = [
            f"[bold yellow]APPROVAL REQUIRED[/bold yellow]",
            f"",
            f"[bold]Action:[/bold] {params.action}",
        ]
        
        if params.reason:
            lines.append(f"[bold]Reason:[/bold] {params.reason}")
        if params.impact:
            lines.append(f"[bold]Impact:[/bold] {params.impact}")
        
        reversible_text = "Yes" if params.reversible else "No (irreversible)"
        lines.append(f"[bold]Reversible:[/bold] {reversible_text}")
        lines.append("")
        
        message = "\n".join(lines)
        
        # Use permission callback if available
        if self.permission_callback:
            approved = self.permission_callback(message)
            return PermissionOutput(
                success=True,
                approved=approved,
                user_response="approved" if approved else "denied",
            )
        
        # Otherwise use console
        try:
            console.print(message)
            
            if not params.reversible:
                console.print("[bold red]WARNING: This action cannot be undone![/bold red]")
            
            response = console.input("Approve this action? [y/N]: ").strip().lower()
            
            approved = response in ("y", "yes")
            
            return PermissionOutput(
                success=True,
                approved=approved,
                user_response=response,
            )
            
        except (EOFError, KeyboardInterrupt):
            return PermissionOutput(
                success=True,
                approved=False,
                user_response="cancelled",
            )
