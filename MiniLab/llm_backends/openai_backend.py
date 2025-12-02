from __future__ import annotations

import os
from typing import List

import httpx

from .base import ChatMessage, LLMBackend


class OpenAIBackend(LLMBackend):
    """
    Minimal OpenAI Chat Completions backend using HTTPX.
    """

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            timeout=60.0,
        )

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

        # Retry logic for rate limits
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
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    # Rate limited, wait and retry
                    wait_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                    print(f"Rate limited, waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

