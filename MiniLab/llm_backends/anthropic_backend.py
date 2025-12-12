from __future__ import annotations

import asyncio
import base64
import os
import time
from pathlib import Path
from typing import List, Optional, Union

import httpx

from .base import ChatMessage, LLMBackend
from ..utils.timing import timing, async_timed_operation


def pdf_to_images(pdf_path: str, max_pages: int = 10, dpi: int = 150) -> List[dict]:
    """
    Convert PDF pages to base64-encoded images for vision API.
    
    Returns list of dicts with format: {"type": "image", "source": {"type": "base64", ...}}
    
    Requires: pip install pdf2image (and poppler installed on system)
    """
    try:
        from pdf2image import convert_from_path
        import io
        
        images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=max_pages)
        
        result = []
        for i, img in enumerate(images):
            # Convert to PNG bytes
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
            
            # Encode to base64
            img_b64 = base64.standard_b64encode(img_bytes).decode('utf-8')
            
            result.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                }
            })
        
        return result
    except ImportError:
        print("  ⚠ pdf2image not installed. Install with: pip install pdf2image")
        print("  ⚠ Also requires poppler: brew install poppler (macOS) or apt install poppler-utils (Linux)")
        return []
    except Exception as e:
        print(f"  ⚠ Error converting PDF to images: {e}")
        return []


def image_to_base64(image_path: str) -> Optional[dict]:
    """
    Convert an image file to base64 for vision API.
    
    Supports: PNG, JPEG, GIF, WebP
    """
    try:
        path = Path(image_path)
        if not path.exists():
            print(f"  ⚠ Image not found: {image_path}")
            return None
        
        # Determine media type
        suffix = path.suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        media_type = media_types.get(suffix, 'image/png')
        
        with open(path, 'rb') as f:
            img_bytes = f.read()
        
        img_b64 = base64.standard_b64encode(img_bytes).decode('utf-8')
        
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": img_b64,
            }
        }
    except Exception as e:
        print(f"  ⚠ Error encoding image: {e}")
        return None


class AnthropicBackend(LLMBackend):
    """
    Minimal Anthropic Messages API backend using HTTPX.
    
    Features:
    - Prompt caching for system prompts (reduces costs by up to 90% on cache hits)
    - Token usage tracking
    - Automatic retries on transient errors
    """

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            timeout=180.0,  # Increased to 3 minutes for complex responses
        )
        
        # Token usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_creation_tokens = 0
        self._total_cache_read_tokens = 0
    
    @property
    def token_usage(self) -> dict:
        """Return current token usage statistics."""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "cache_creation_tokens": self._total_cache_creation_tokens,
            "cache_read_tokens": self._total_cache_read_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }
    
    def _update_token_usage(self, usage: dict) -> None:
        """Update token usage from API response."""
        self._total_input_tokens += usage.get("input_tokens", 0)
        self._total_output_tokens += usage.get("output_tokens", 0)
        self._total_cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
        self._total_cache_read_tokens += usage.get("cache_read_input_tokens", 0)

    async def acomplete(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        max_retries: int = 3,
        use_cache: bool = True,
    ) -> str:
        # Track timing
        start_time = time.perf_counter()
        
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
        
        # Build headers - enable caching beta if requested
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        
        if system_content:
            if use_cache:
                # Enable prompt caching beta
                headers["anthropic-beta"] = "prompt-caching-2024-07-31"
                # Format system as array with cache_control on last block
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system_content,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                payload["system"] = system_content

        # Retry logic for transient errors
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = await self._client.post(
                    "/messages",
                    json=payload,
                    headers=headers,
                )
                
                # Handle HTTP errors
                if resp.status_code != 200:
                    try:
                        error_data = resp.json()
                        error_msg = error_data.get("error", {}).get("message", str(error_data))
                        print(f"\n  API Error ({resp.status_code}): {error_msg}\n")
                    except Exception:
                        print(f"\n  API Error ({resp.status_code}): {resp.text[:500]}\n")
                    
                    # Retry on 5xx errors or rate limits
                    if resp.status_code >= 500 or resp.status_code == 429:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5
                            print(f"  Retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                            await asyncio.sleep(wait_time)
                            continue
                    resp.raise_for_status()
                
                data = resp.json()
                result = data["content"][0]["text"]
                
                # Track token usage from response
                if "usage" in data:
                    self._update_token_usage(data["usage"])
                    # Log cache effectiveness if timing enabled
                    usage = data["usage"]
                    cache_read = usage.get("cache_read_input_tokens", 0)
                    cache_create = usage.get("cache_creation_input_tokens", 0)
                    if cache_read > 0 or cache_create > 0:
                        import os
                        if os.environ.get("MINILAB_TIMING") == "1":
                            if cache_read > 0:
                                print(f"  💾 Cache hit: {cache_read:,} tokens read from cache")
                            if cache_create > 0:
                                print(f"  💾 Cache miss: {cache_create:,} tokens cached for future use")
                
                # Record timing
                duration_ms = (time.perf_counter() - start_time) * 1000
                timing().record("acomplete", "llm", duration_ms, model=self.model)
                
                return result
                
            except httpx.ReadTimeout as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"\n  Request timed out. Retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"\n  Request timed out after {max_retries} attempts.\n")
                    raise
            except httpx.ConnectError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"\n  Connection error. Retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        raise last_error

    async def acomplete_with_vision(
        self,
        messages: List[ChatMessage],
        images: List[dict],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        max_retries: int = 3,
        use_cache: bool = True,
    ) -> str:
        """
        Complete with vision - send images along with text for visual analysis.
        
        Args:
            messages: Standard chat messages
            images: List of image dicts from pdf_to_images() or image_to_base64()
            temperature: Sampling temperature
            max_tokens: Max response tokens
            max_retries: Retry count
            use_cache: Enable prompt caching for system prompts
        """
        # Anthropic requires extracting system messages separately
        system_content = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                if system_content is None:
                    system_content = msg["content"]
                else:
                    system_content += "\n\n" + msg["content"]
            else:
                # For user messages, we can include images
                if msg["role"] == "user" and images:
                    # Build content array with images + text
                    content_parts = []
                    for img in images:
                        content_parts.append(img)
                    content_parts.append({
                        "type": "text",
                        "text": msg["content"]
                    })
                    anthropic_messages.append({
                        "role": "user",
                        "content": content_parts
                    })
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
        
        # Build headers - enable caching beta if requested
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        
        if system_content:
            if use_cache:
                headers["anthropic-beta"] = "prompt-caching-2024-07-31"
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system_content,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                payload["system"] = system_content

        last_error = None
        for attempt in range(max_retries):
            try:
                resp = await self._client.post(
                    "/messages",
                    json=payload,
                    headers=headers,
                )
                
                if resp.status_code != 200:
                    try:
                        error_data = resp.json()
                        error_msg = error_data.get("error", {}).get("message", str(error_data))
                        print(f"\n  Vision API Error ({resp.status_code}): {error_msg}\n")
                    except Exception:
                        print(f"\n  Vision API Error ({resp.status_code}): {resp.text[:500]}\n")
                    
                    if resp.status_code >= 500 or resp.status_code == 429:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5
                            print(f"  Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                    resp.raise_for_status()
                
                data = resp.json()
                
                # Track token usage
                if "usage" in data:
                    self._update_token_usage(data["usage"])
                
                return data["content"][0]["text"]
                
            except (httpx.ReadTimeout, httpx.ConnectError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"\n  Vision request error. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        raise last_error
