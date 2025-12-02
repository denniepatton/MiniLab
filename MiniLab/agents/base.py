from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from MiniLab.llm_backends.base import ChatMessage, LLMBackend


@dataclass
class Agent:
    id: str
    display_name: str
    guild: str
    role: str
    persona: str
    backend: LLMBackend
    tools: List[str] = field(default_factory=list)
    tool_instances: Dict[str, Any] = field(default_factory=dict)  # Actual tool objects

    # In future this could be a proper memory object / RAG store
    memory_notes: List[str] = field(default_factory=list)
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if agent has access to a specific tool."""
        return tool_name in self.tool_instances
    
    async def use_tool(self, tool_name: str, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool if available."""
        if not self.has_tool(tool_name):
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not available to {self.display_name}",
                "available_tools": list(self.tool_instances.keys()),
            }
        
        tool = self.tool_instances[tool_name]
        return await tool.execute(action=action, **kwargs)

    async def arespond(
        self,
        user_message: str,
        context: str | None = None,
        max_tokens: int = 1000,
    ) -> str:
        """
        Basic response method: persona + optional context + user message.
        
        Args:
            user_message: The question or prompt for the agent
            context: Optional shared context
            max_tokens: Maximum tokens in response (default 1000, ~700 words)
        """
        messages: List[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    f"You are {self.display_name}, {self.role} in the MiniLab team.\n"
                    f"Persona:\n{self.persona}\n\n"
                    "Guidelines:\n"
                    "- Be concise and focused - say what you need to say and no more\n"
                    "- Ground scientific claims in cited literature when relevant\n"
                    "- Aim for clarity over verbosity\n"
                    "- Typical responses should be 2-4 paragraphs unless more detail is specifically requested\n\n"
                    f"Available tools: {', '.join(self.tool_instances.keys()) if self.tool_instances else 'None'}\n"
                    "Note: You can create, read, and write files in the Sandbox directory using the filesystem tool."
                ),
            },
        ]
        if context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Shared context for this discussion:\n{context}",
                }
            )
        messages.append({"role": "user", "content": user_message})

        reply = await self.backend.acomplete(messages, max_tokens=max_tokens)
        # Later: update memory based on reply, etc.
        return reply

