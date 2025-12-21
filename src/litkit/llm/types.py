"""LLM type definitions and data contracts.

Provides UI- and provider-neutral data structures for LLM operations.
These types form stable contracts between the LLM module and callers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class LLMConfig:
    """Configuration for LLM operations (reusable across calls).
    
    Encapsulates all settings needed to make LLM calls, including:
    - Endpoint configuration (model, URL, auth)
    - Token budgets (per model family)
    - Provider preferences
    
    Use `from_env_and_args()` to construct from environment/CLI args.
    """
    # Endpoint configuration
    model: str = "gpt-oss:20b"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "no-auth"
    timeout_sec: int = 15
    
    # Output configuration
    max_out_tokens_default: int = 3000
    sys_prompt: str = ""  # Empty = use prompts.SYS_PROMPT_QA
    
    # Provider selection
    # - "detect" (default): probe endpoint, prefer Responses for o-series
    # - "auto": alias for detect
    # - "responses": force Responses API (fail if unavailable)
    # - "chat": force Chat Completions API
    provider_preference: Literal[
        "auto", "detect", "responses", "chat"
    ] = "detect"
    
    # Token budgets (approximate; ~4 chars/token heuristic)
    # Override via env LITKIT_BUDGET_O3 / LITKIT_BUDGET_OSS20B
    budget_o3: int = 32000
    budget_oss: int = 8000
    prompt_headroom: int = 200  # Safety buffer for SDK scaffolding
    
    @classmethod
    def from_env_and_args(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_sec: int | None = None,
        max_out_tokens: int | None = None,
        provider_preference: str | None = None,
        **kwargs: Any,
    ) -> LLMConfig:
        """Build config from environment variables with CLI arg overrides.
        
        Priority: explicit args > environment variables > defaults
        
        Environment variables:
            LLM_MODEL: Default model name
            OPENAI_BASE_URL: Default endpoint URL
            OPENAI_API_KEY: Default API key
            LITKIT_OPENAI_TIMEOUT_SEC: Request timeout
            LITKIT_BUDGET_O3: Token budget for o-series models
            LITKIT_BUDGET_OSS20B: Token budget for OSS models
            LITKIT_PROMPT_HEADROOM_TOKENS: Safety buffer
        
        Args:
            model: Override for LLM_MODEL
            base_url: Override for OPENAI_BASE_URL
            api_key: Override for OPENAI_API_KEY
            timeout_sec: Override for LITKIT_OPENAI_TIMEOUT_SEC
            max_out_tokens: Override for default max output tokens
            provider_preference: Override for provider selection
            **kwargs: Additional fields to set
        
        Returns:
            Configured LLMConfig instance
        """
        # Resolve API key with model-aware defaults
        resolved_model = model or os.environ.get("LLM_MODEL", "gpt-oss:20b")
        resolved_api_key = api_key
        if resolved_api_key is None:
            resolved_api_key = os.environ.get("OPENAI_API_KEY")
            if resolved_api_key is None:
                # Model-aware default: local models tolerate "no-auth"
                if resolved_model.lower().startswith("gpt-oss"):
                    resolved_api_key = "no-auth"
                else:
                    resolved_api_key = ""
        
        return cls(
            model=resolved_model,
            base_url=(
                base_url
                or os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
            ),
            api_key=resolved_api_key,
            timeout_sec=(
                timeout_sec
                or int(os.environ.get("LITKIT_OPENAI_TIMEOUT_SEC", "15"))
            ),
            max_out_tokens_default=(max_out_tokens or 3000),
            provider_preference=(  # type: ignore[arg-type]
                provider_preference or "detect"
            ),
            budget_o3=int(os.environ.get("LITKIT_BUDGET_O3", "32000")),
            budget_oss=int(os.environ.get("LITKIT_BUDGET_OSS20B", "8000")),
            prompt_headroom=int(
                os.environ.get("LITKIT_PROMPT_HEADROOM_TOKENS", "200")
            ),
            **kwargs,
        )
    
    def is_o_series(self) -> bool:
        """Check if model is an o-series (reasoning) model."""
        return self.model.lower().startswith("o3")
    
    def is_openai_cloud(self) -> bool:
        """Check if endpoint is the official OpenAI cloud."""
        if not self.base_url:
            return True  # Conservative if unknown
        try:
            from urllib.parse import urlparse
            host = (urlparse(self.base_url).hostname or "").lower()
            return host.endswith("api.openai.com")
        except Exception:
            return "openai.com" in self.base_url.lower()
    
    def get_budget(self) -> int:
        """Get token budget for the configured model."""
        return self.budget_o3 if self.is_o_series() else self.budget_oss


@dataclass
class LLMResponse:
    """Response from an LLM call.
    
    Provides a provider-neutral view of the LLM output.
    """
    text: str
    provider: str  # "openai-responses" | "openai-chat" | "local-chat"
    usage: dict[str, Any] | None = None  # Token usage if available
    raw: Any | None = None  # Original response object (for debugging)


# ═══════════════════════════════════════════════════════════════════════════════
# FUTURE MULTI-TURN SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════
# These types establish contracts for future REPL/TUI interactive modes. They
# are defined now but not used by the current single-turn answer_question().
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    """Single turn in a conversation.
    
    Captures the full lifecycle of one user interaction:
    1. User input (raw question)
    2. Optional query rewriting (for retrieval optimization)
    3. Retrieval (selected chunk IDs)
    4. LLM answer generation
    """
    turn_id: str
    user_text: str
    rewritten_text: str | None = None  # Query rewriting result
    retrieval_query: str = ""  # Final query for retrieval
    answer_text: str | None = None  # LLM response
    selected_chunk_ids: list[int] | None = None  # Context chunks
    timestamp: float = 0.0  # Unix timestamp


@dataclass
class SessionState:
    """Multi-turn session state.
    
    Maintains conversation history for context-aware interactions.
    Future use: REPL mode, TUI, web interface.
    """
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_turn(self, turn: Turn) -> None:
        """Append a turn to the session history."""
        self.turns.append(turn)
    
    def last_turn(self) -> Turn | None:
        """Get the most recent turn, or None if no turns."""
        return self.turns[-1] if self.turns else None
    
    def turn_count(self) -> int:
        """Get number of turns in this session."""
        return len(self.turns)
