from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, TypedDict


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

    async def acomplete_streaming(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        """
        Streaming completion with optional chunk callback.
        Default implementation falls back to non-streaming.
        """
        return await self.acomplete(messages, temperature, max_tokens)

    # Optionally add synchronous wrapper if you want
    async def acomplete_simple(self, prompt: str) -> str:
        return await self.acomplete([{"role": "user", "content": prompt}])


def parse_backend_name(name: str) -> tuple[str, str]:
    """
    Parse backend string like 'openai:gpt-4o' -> ('openai', 'gpt-4o').
    """
    provider, model = name.split(":", 1)
    return provider, model

