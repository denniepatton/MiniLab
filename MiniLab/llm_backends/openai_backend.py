from __future__ import annotations

import os
from typing import List

import httpx

from .base import ChatMessage, LLMBackend


class OpenAIBackend(LLMBackend):
    """
    Minimal OpenAI Chat Completions backend using HTTPX.
    
    Features:
    - Token usage tracking via TokenAccount singleton
    - Automatic retries on rate limits with exponential backoff
    """

    def __init__(
        self, 
        model: str, 
        api_key: str | None = None,
        agent_id: str = "unknown",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.agent_id = agent_id  # For token tracking
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            timeout=120.0,  # Increased timeout for complex responses
        )
        
        # Local token tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        
        # Project context for resume functionality
        self._project_context: str = ""
        self._cached_persona: str = ""
    
    @property
    def token_usage(self) -> dict:
        """Return current token usage statistics."""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }
    
    def set_persona(self, persona: str) -> None:
        """Set the agent's persona/system prompt."""
        self._cached_persona = persona
    
    def append_project_context(self, context: str) -> None:
        """Append project context for resume functionality."""
        self._project_context = context
    
    def get_full_system_prompt(self, additional_context: str = "") -> str:
        """Build combined system prompt."""
        parts = []
        if self._cached_persona:
            parts.append(self._cached_persona)
        if self._project_context:
            parts.append(f"\n\n## PROJECT CONTEXT\n{self._project_context}")
        if additional_context:
            parts.append(f"\n\n{additional_context}")
        return "".join(parts)
    
    def _update_token_usage(self, usage: dict, operation: str = "llm.complete") -> None:
        """Update token usage from API response - both local and global."""
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        # Update local tracking
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        
        # Update global TokenAccount
        try:
            from ..core import get_token_account
            from ..core.token_context import get_workflow, get_trigger
            account = get_token_account()
            # Always record usage even before a budget is set.
            account.debit(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                agent_id=self.agent_id,
                operation=operation,
                workflow=get_workflow(),
                trigger=get_trigger(),
            )
        except Exception:
            pass  # Fail silently if TokenAccount not available

    async def acomplete(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        import asyncio
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Retry logic for rate limits with capped exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = await self._client.post(
                    "/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                
                # Track token usage
                if "usage" in data:
                    self._update_token_usage(data["usage"])
                
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    # Rate limited, wait with capped exponential backoff
                    wait_time = min((2 ** attempt) * 2, 30)  # Cap at 30 seconds
                    print(f"Rate limited, waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

