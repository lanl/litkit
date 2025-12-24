"""LLM module for litkit.

Provides provider-agnostic LLM integration with:
- Typed exceptions for error handling
- Configuration management via LLMConfig
- Context packing with token budgeting
- Automatic overflow retry with progressive trimming
- Session/Turn types for future multi-turn support
- Query rewriting stubs for future interactive modes
"""
from .errors import (
    LLMError,
    AuthError,
    ModelNotFoundError,
    EndpointNotSupportedError,
    ContextOverflowError,
    TransportError,
    is_overflow_error,
    clarify_error,
)
from .types import LLMConfig, LLMResponse, SessionState, Turn
from .prompts import SYS_PROMPT_QA, SYS_PROMPT_REWRITE
from .qa import answer_question, pack_context, approx_tokens
from .rewrite import rewrite_query, RewriteResult

__all__ = [
    # Errors
    "LLMError",
    "AuthError",
    "ModelNotFoundError",
    "EndpointNotSupportedError",
    "ContextOverflowError",
    "TransportError",
    "is_overflow_error",
    "clarify_error",
    # Types
    "LLMConfig",
    "LLMResponse",
    "SessionState",
    "Turn",
    "RewriteResult",
    # Core functions
    "answer_question",
    "pack_context",
    "approx_tokens",
    "rewrite_query",
    # Prompts
    "SYS_PROMPT_QA",
    "SYS_PROMPT_REWRITE",
]
