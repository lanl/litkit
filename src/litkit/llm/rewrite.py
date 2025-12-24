"""Query rewriting for improved retrieval.

Provides query rewriting capabilities for the RAG pipeline:
- Expand acronyms
- Add relevant synonyms
- Remove conversational filler
- Ensure key technical terms are present

Currently a stub with mode="none" passthrough.
Future modes ("light", "interactive") will use LLM-based rewriting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .types import LLMConfig, SessionState


@dataclass
class RewriteResult:
    """Result of query rewriting.
    
    Attributes:
        final_query: The rewritten query to use for retrieval
        alternates: Alternative rewrites (for interactive mode)
        notes: Optional diagnostic notes
    """
    final_query: str
    alternates: list[str]
    notes: str | None = None


def rewrite_query(
    session: SessionState | None,
    user_text: str,
    config: LLMConfig,
    *,
    mode: Literal["none", "light", "interactive"] = "none",
    max_alternates: int = 3,
) -> RewriteResult:
    """Rewrite user query for improved retrieval.
    
    Modes:
        none: Passthrough (final_query == user_text). Default.
        light: Single best rewrite optimized for retrieval.
               Expands acronyms, removes chatter, adds synonyms.
        interactive: Multiple alternates for operator selection.
                     Used in future TUI/REPL modes.
    
    Args:
        session: Optional session state (for context-aware rewriting)
        user_text: Raw user input
        config: LLM configuration (for light/interactive modes)
        mode: Rewrite mode
        max_alternates: Max alternates for interactive mode
    
    Returns:
        RewriteResult with final query and any alternates
    
    Raises:
        NotImplementedError: For unimplemented modes
    
    Example:
        >>> result = rewrite_query(None, "what is hiv", config, mode="none")
        >>> result.final_query
        'what is hiv'
        >>> result.alternates
        []
    """
    if mode == "none":
        return RewriteResult(
            final_query=user_text,
            alternates=[],
            notes=None,
        )
    
    if mode == "light":
        # TODO: Implement single-shot LLM rewriting
        # - Use SYS_PROMPT_REWRITE
        # - Call provider.generate() with user_text
        # - Return rewritten query
        raise NotImplementedError(
            "rewrite mode='light' not yet implemented. "
            "Use mode='none' for passthrough."
        )
    
    if mode == "interactive":
        # TODO: Implement multi-alternate generation
        # - Generate max_alternates different rewrites
        # - Return for user selection in TUI
        raise NotImplementedError(
            "rewrite mode='interactive' not yet implemented. "
            "Use mode='none' for passthrough."
        )
    
    # Should not reach here due to Literal type
    raise ValueError(f"Unknown rewrite mode: {mode!r}")
