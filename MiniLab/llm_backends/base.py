from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, TypedDict


MessageRole = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: MessageRole
    content: str


class LLMBackend(ABC):
    """
    Abstract interface for all LLM providers.
    """

    @abstractmethod
    async def acomplete(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        ...

    # Optionally add synchronous wrapper if you want
    async def acomplete_simple(self, prompt: str) -> str:
        return await self.acomplete([{"role": "user", "content": prompt}])


def parse_backend_name(name: str) -> tuple[str, str]:
    """
    Parse backend string like 'openai:gpt-4o' -> ('openai', 'gpt-4o').
    """
    provider, model = name.split(":", 1)
    return provider, model

