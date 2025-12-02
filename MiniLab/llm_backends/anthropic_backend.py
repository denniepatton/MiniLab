from __future__ import annotations

import os
from typing import List

import httpx

from .base import ChatMessage, LLMBackend


class AnthropicBackend(LLMBackend):
    """
    Minimal Anthropic Messages API backend using HTTPX.
    """

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            timeout=60.0,
        )

    async def acomplete(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        # Anthropic requires extracting system messages separately
        system_content = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                # Concatenate all system messages
                if system_content is None:
                    system_content = msg["content"]
                else:
                    system_content += "\n\n" + msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        if system_content:
            payload["system"] = system_content

        resp = await self._client.post(
            "/messages",
            json=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
