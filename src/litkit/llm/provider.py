"""LLM provider abstraction and implementations.

Encapsulates OpenAI SDK interactions behind a protocol, enabling:
- Provider-agnostic calling code
- Easy testing with mock providers
- Future support for additional backends

All providers lazy-import the OpenAI SDK to preserve import purity.
"""
from __future__ import annotations

import sys
from typing import Any, Protocol

from .types import LLMConfig, LLMResponse
from .errors import (
    LLMError,
    ContextOverflowError,
    EndpointNotSupportedError,
    is_overflow_error,
    is_endpoint_unsupported,
)


class LLMProvider(Protocol):
    """Protocol for LLM providers.
    
    Implementations must provide generate() for making LLM calls.
    The preflight() method is optional for endpoint probing.
    """
    
    def generate(
        self,
        prompt: str,
        config: LLMConfig,
        *,
        max_out_tokens: int | None = None,
        sys_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            prompt: The user prompt (question + context)
            config: LLM configuration
            max_out_tokens: Override max output tokens
            sys_prompt: Override system prompt
        
        Returns:
            LLMResponse with generated text
        
        Raises:
            ContextOverflowError: If input exceeds context window
            EndpointNotSupportedError: If endpoint lacks required API
            LLMError: For other LLM-related errors
        """
        ...
    
    def preflight(self, config: LLMConfig) -> dict[str, Any]:
        """Optional: probe endpoint capabilities.
        
        Best-effort; should not raise on servers that lack models.list().
        
        Returns:
            Dict with endpoint info (e.g., available models)
        """
        ...


class OpenAIChatProvider:
    """Chat Completions API provider (local/OSS models).
    
    Uses the standard OpenAI Chat Completions API, compatible with:
    - OpenAI cloud
    - LM Studio, Ollama, vLLM, text-generation-inference
    - Any OpenAI-compatible endpoint
    """
    
    def generate(
        self,
        prompt: str,
        config: LLMConfig,
        *,
        max_out_tokens: int | None = None,
        sys_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate using Chat Completions API."""
        # Lazy import - critical for offline/build-only runs
        from openai import OpenAI
        
        client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_sec,
        )
        
        max_out = max_out_tokens or config.max_out_tokens_default
        prompt_text = sys_prompt or config.sys_prompt
        
        try:
            resp = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_out,
            )
            text = getattr(resp.choices[0].message, "content", "") or ""
            usage = None
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
            return LLMResponse(
                text=text.strip(),
                provider="openai-chat",
                usage=usage,
            )
        except Exception as e:
            if is_overflow_error(e):
                raise ContextOverflowError(str(e)) from e
            raise LLMError(str(e)) from e
    
    def preflight(self, config: LLMConfig) -> dict[str, Any]:
        """Probe endpoint for available models."""
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout_sec,
            )
            models_resp = client.models.list()
            available = [
                getattr(x, "id", str(x))
                for x in getattr(models_resp, "data", []) or []
            ]
            return {"models": available}
        except Exception:
            return {"models": []}


class OpenAIResponsesProvider:
    """Responses API provider (o-series models).
    
    Uses the OpenAI Responses API for reasoning models (o3, etc.).
    This API supports extended reasoning and is only available on
    OpenAI cloud or compatible enterprise endpoints.
    """
    
    def generate(
        self,
        prompt: str,
        config: LLMConfig,
        *,
        max_out_tokens: int | None = None,
        sys_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate using Responses API."""
        from openai import OpenAI
        
        client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_sec,
        )
        
        max_out = max_out_tokens or config.max_out_tokens_default
        prompt_text = sys_prompt or config.sys_prompt
        
        try:
            resp = client.responses.create(
                model=config.model,
                input=prompt,
                instructions=prompt_text,
                max_output_tokens=max_out,
                reasoning={"effort": "medium"},
            )
            # Extract text from response
            text = getattr(resp, "output_text", None)
            if text is None:
                # Older SDK versions: synthesize from content
                try:
                    parts = []
                    for item in getattr(resp, "output", []) or []:
                        for c in getattr(item, "content", []) or []:
                            if getattr(c, "type", "") == "output_text":
                                parts.append(getattr(c, "text", ""))
                    text = "".join(parts).strip() if parts else ""
                except Exception:
                    text = ""
            return LLMResponse(
                text=(text or "").strip(),
                provider="openai-responses",
                usage=None,  # Responses API usage format differs
            )
        except Exception as e:
            # Detect "endpoint doesn't support Responses API"
            if is_endpoint_unsupported(e):
                raise EndpointNotSupportedError(
                    f"Endpoint {config.base_url} doesn't support "
                    f"Responses API for {config.model}"
                ) from e
            if is_overflow_error(e):
                raise ContextOverflowError(str(e)) from e
            raise LLMError(str(e)) from e
    
    def preflight(self, config: LLMConfig) -> dict[str, Any]:
        """Probe endpoint (same as Chat provider)."""
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout_sec,
            )
            models_resp = client.models.list()
            available = [
                getattr(x, "id", str(x))
                for x in getattr(models_resp, "data", []) or []
            ]
            return {"models": available}
        except Exception:
            return {"models": []}


class AutoProvider:
    """Auto-selecting provider based on model and endpoint capabilities.
    
    Selection logic:
    1. If provider_preference is explicit ("responses" or "chat"), use it
    2. If model is o-series (o3*), try Responses first, fall back to Chat
    3. Otherwise, use Chat
    """
    
    def __init__(self) -> None:
        self._chat = OpenAIChatProvider()
        self._responses = OpenAIResponsesProvider()
    
    def generate(
        self,
        prompt: str,
        config: LLMConfig,
        *,
        max_out_tokens: int | None = None,
        sys_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate using auto-selected provider."""
        pref = config.provider_preference
        is_o_series = config.is_o_series()
        
        kwargs = {
            "max_out_tokens": max_out_tokens,
            "sys_prompt": sys_prompt,
        }
        
        # Explicit preference
        if pref == "chat":
            return self._chat.generate(prompt, config, **kwargs)
        elif pref == "responses":
            return self._responses.generate(prompt, config, **kwargs)
        
        # Auto-detect for o-series
        if is_o_series:
            try:
                return self._responses.generate(prompt, config, **kwargs)
            except EndpointNotSupportedError:
                # Fall back to Chat (with warning)
                sys.stderr.write(
                    f"[llm] Responses API not available; "
                    f"falling back to Chat for {config.model}\n"
                )
                return self._chat.generate(prompt, config, **kwargs)
        
        # Default: Chat API
        return self._chat.generate(prompt, config, **kwargs)
    
    def preflight(self, config: LLMConfig) -> dict[str, Any]:
        """Probe endpoint using Chat provider."""
        return self._chat.preflight(config)
