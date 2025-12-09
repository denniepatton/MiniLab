"""
User Input Tool for MiniLab Agents

Allows agents to directly ask the user for input and receive their response.
This enables fully agentic workflows where the AGENT decides when to ask
for user feedback, what to ask, and how to interpret the response.
"""

from __future__ import annotations

from typing import Any, Dict

from . import Tool


class UserInputTool(Tool):
    """
    Tool for agents to request input from the user.
    
    The agent decides:
    - WHEN to ask (at appropriate checkpoints)
    - WHAT to ask (the question/prompt)
    - HOW to interpret the response (agent's responsibility)
    
    The orchestrator does NOT interpret user input - it goes directly
    to the agent who asked for it.
    """

    def __init__(self):
        super().__init__(
            name="user_input",
            description="""Ask the user for input and wait for their response.
Action: ask
Params: {prompt: "Your question to the user"}

The user's response is returned exactly as typed. YOU (the agent) interpret it.
Use this at checkpoints to confirm plans, get feedback, or ask clarifying questions.

Example: {"tool": "user_input", "action": "ask", "params": {"prompt": "Does this project name work? (yes/no or suggest alternative)"}}"""
        )

    async def execute(self, action: str = "ask", **kwargs) -> Dict[str, Any]:
        """
        Ask the user for input.
        
        Args:
            action: Should be "ask"
            prompt: The question/prompt to show the user
            
        Returns:
            Dict with user's response
        """
        if action != "ask":
            return {
                "success": False,
                "error": f"Unknown action: {action}. Use 'ask'.",
            }
        
        prompt = kwargs.get("prompt", "Please provide input:")
        
        # Print the prompt and wait for input
        print(f"\n  🗣️ Agent asks: {prompt}")
        try:
            user_response = input("  > ").strip()
            
            return {
                "success": True,
                "response": user_response,
                "message": f"User responded: {user_response[:100]}{'...' if len(user_response) > 100 else ''}",
            }
        except EOFError:
            return {
                "success": False,
                "error": "No input available (EOF)",
            }
        except KeyboardInterrupt:
            return {
                "success": False,
                "error": "User cancelled input",
            }
