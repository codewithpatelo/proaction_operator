"""Unified LLM client with retry/backoff and per-provider quirks.

Handles OpenAI (seed), Anthropic (temp=0), DeepSeek (temp=0).
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable

import httpx


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════════

class RateLimitError(Exception):
    """429 or quota exceeded."""
    pass


class ParseError(Exception):
    """JSON parsing failed after retries."""
    pass


class TransientError(Exception):
    """5xx or timeout — may succeed on retry."""
    pass


class FatalError(Exception):
    """Auth failure or malformed request — do not retry."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MAX_RETRIES: int = 4
TIMEOUT_SECONDS: int = 60
CONCURRENCY_LIMIT: int = 8  # Bumped per latency analysis


# ═══════════════════════════════════════════════════════════════════════════════
# Provider detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_provider(model: str) -> str:
    """Detect provider from model name."""
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("deepseek-"):
        return "deepseek"
    raise ValueError(f"Unknown model: {model}")


# ═══════════════════════════════════════════════════════════════════════════════
# JSON extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_json(text: str) -> dict | None:
    """Extract first balanced JSON object from text.
    
    Same pattern as JAIIO agent.py::_extract_json.
    """
    # Find first {
    start = text.find("{")
    if start == -1:
        return None
    
    # Track brace depth
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    import json
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Clients
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    system_fingerprint: str | None = None  # OpenAI only
    response_id: str | None = None


class BaseClient:
    """Base async LLM client with retry logic."""
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or self._default_key()
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    def _default_key(self) -> str:
        raise NotImplementedError
    
    async def call(
        self,
        model: str,
        system_prompt: str | None,
        user_prompt: str,
        seed: int | None = None,
        max_retries: int = MAX_RETRIES,
    ) -> LLMResponse:
        """Make call with exponential backoff."""
        import random
        
        base_delay = 2.0
        
        for attempt in range(max_retries + 1):
            try:
                async with self.semaphore:
                    return await self._call_once(model, system_prompt, user_prompt, seed)
            except RateLimitError:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(min(delay, 60))
                else:
                    raise
            except TransientError:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(min(delay, 60))
                else:
                    raise
            except FatalError:
                raise
        
        raise TransientError(f"Max retries ({max_retries}) exceeded")
    
    async def _call_once(
        self,
        model: str,
        system_prompt: str | None,
        user_prompt: str,
        seed: int | None,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAIClient(BaseClient):
    """OpenAI client with native seed support."""
    
    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            self.client = None
    
    def _default_key(self) -> str:
        # Support both OPENAI_API_KEY and OPEN_AI_API_KEY for compatibility
        key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
        if not key:
            raise FatalError("OPENAI_API_KEY not set")
        return key
    
    async def _call_once(
        self,
        model: str,
        system_prompt: str | None,
        user_prompt: str,
        seed: int | None,
    ) -> LLMResponse:
        if not self.client:
            raise FatalError("openai package not installed")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # OpenAI reasoning models (gpt-5-nano) use temperature=1 fixed
        # Seed is the primary determinism mechanism
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            kwargs["seed"] = seed
        
        try:
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            
            # Empty content from API is treated as transient (will retry)
            if not content.strip():
                raise TransientError("OpenAI returned empty content")
            
            return LLMResponse(
                content=content,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                model=response.model,
                system_fingerprint=getattr(response, "system_fingerprint", None),
                response_id=getattr(response, "id", None),
            )
        except (RateLimitError, FatalError, TransientError):
            raise
        except Exception as e:
            error_str = str(e).lower()
            # Funds exhausted: mark provider as fatally unusable
            if any(kw in error_str for kw in ["insufficient_quota", "402", "payment_required", "credit balance", "insufficient credit", "billing_hard_limit"]):
                raise FatalError(f"INSUFFICIENT_FUNDS: {e}")
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(str(e))
            if "401" in error_str or "authentication" in error_str:
                raise FatalError(str(e))
            if "timeout" in error_str:
                raise TransientError(str(e))
            raise TransientError(str(e))


class AnthropicClient(BaseClient):
    """Anthropic client — no seed, uses temperature=0."""
    
    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            self.client = None
    
    def _default_key(self) -> str:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise FatalError("ANTHROPIC_API_KEY not set")
        return key
    
    async def _call_once(
        self,
        model: str,
        system_prompt: str | None,
        user_prompt: str,
        seed: int | None,  # Ignored — Anthropic has no seed param
    ) -> LLMResponse:
        if not self.client:
            raise FatalError("anthropic package not installed")
        
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=512,
                temperature=0,  # Best-effort determinism
                system=system_prompt or "",
                messages=[{"role": "user", "content": user_prompt}],
            )
            
            content = response.content[0].text if response.content else ""
            
            if not content.strip():
                raise TransientError("Anthropic returned empty content")
            
            return LLMResponse(
                content=content,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                model=response.model,
                response_id=getattr(response, "id", None),
            )
        except (RateLimitError, FatalError, TransientError):
            raise
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["insufficient_quota", "402", "payment_required", "credit balance", "insufficient credit", "billing_hard_limit"]):
                raise FatalError(f"INSUFFICIENT_FUNDS: {e}")
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(str(e))
            if "401" in error_str:
                raise FatalError(str(e))
            raise TransientError(str(e))


class DeepSeekClient(BaseClient):
    """DeepSeek client — OpenAI-compatible API, seed deprecated, temp=0."""
    
    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        self.base_url = "https://api.deepseek.com"
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            self.client = None
    
    def _default_key(self) -> str:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise FatalError("DEEPSEEK_API_KEY not set")
        return key
    
    async def _call_once(
        self,
        model: str,
        system_prompt: str | None,
        user_prompt: str,
        seed: int | None,  # Ignored — DeepSeek deprecated seed
    ) -> LLMResponse:
        if not self.client:
            raise FatalError("openai package not installed (used for DeepSeek)")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,  # Best-effort determinism
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content or ""
            
            if not content.strip():
                raise TransientError("DeepSeek returned empty content")
            
            return LLMResponse(
                content=content,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                model=response.model,
                response_id=getattr(response, "id", None),
            )
        except (RateLimitError, FatalError, TransientError):
            raise
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["insufficient_quota", "402", "payment_required", "credit balance", "insufficient credit", "billing_hard_limit"]):
                raise FatalError(f"INSUFFICIENT_FUNDS: {e}")
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(str(e))
            if "401" in error_str:
                raise FatalError(str(e))
            raise TransientError(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

CLIENTS: dict[str, type[BaseClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "deepseek": DeepSeekClient,
}

# Singleton instances per provider so the asyncio.Semaphore inside each client
# is SHARED across all parallel calls (otherwise concurrency limit is meaningless
# when multiple cells run in batches and each builds its own client).
_CLIENT_INSTANCES: dict[str, BaseClient] = {}


def get_client(provider: str) -> BaseClient:
    """Get appropriate client for provider (singleton — shared semaphore)."""
    if provider in _CLIENT_INSTANCES:
        return _CLIENT_INSTANCES[provider]
    client_cls = CLIENTS.get(provider)
    if not client_cls:
        raise ValueError(f"Unknown provider: {provider}")
    instance = client_cls()
    _CLIENT_INSTANCES[provider] = instance
    return instance


async def call_llm(
    provider: str,
    model: str,
    system_prompt: str | None,
    user_prompt: str,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """High-level LLM call with full metadata return.
    
    Returns:
        (content, metadata) where metadata includes tokens, latency, fingerprint.
    """
    import time
    
    client = get_client(provider)
    start = time.time()
    
    response = await client.call(model, system_prompt, user_prompt, seed)
    
    latency_ms = (time.time() - start) * 1000
    
    metadata = {
        "provider": provider,
        "model": model,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "latency_ms": latency_ms,
        "system_fingerprint": response.system_fingerprint,
        "response_id": response.response_id,
    }
    
    return response.content, metadata
