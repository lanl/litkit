"""Question answering with context packing and overflow handling.

Provides the main entry point for RAG-style Q&A:
- pack_context(): Token-aware context assembly
- answer_question(): Full Q&A pipeline with retry
"""
from __future__ import annotations

import sys
from typing import Any

from .types import LLMConfig
from .provider import AutoProvider
from .prompts import SYS_PROMPT_QA
from .errors import ContextOverflowError, LLMError, clarify_error


def approx_tokens(s: str) -> int:
    """Rough char→token approximation (~4 chars/token).
    
    Returns at least 1 to avoid division-by-zero issues.
    
    Note: This is a conservative heuristic. For more accurate
    counting with OpenAI models, consider using tiktoken.
    """
    return max(1, len(s) // 4)


def pack_context(
    chunks: list[dict[str, str]],
    question: str,
    config: LLMConfig,
    *,
    sys_prompt: str | None = None,
    max_out_tokens: int | None = None,
) -> tuple[str, list[int], dict[str, Any]]:
    """Pack ranked chunks into a context string respecting token budget.
    
    Assembles chunks into a formatted context block, stopping when
    the token budget is exhausted. Returns partial results if not
    all chunks fit (caller decides how to handle truncation).
    
    Args:
        chunks: Ranked list of chunk dicts with 'text', 'paper_title',
                'pmcid', 'pmid' keys
        question: User question text
        config: LLM configuration (for budget selection)
        sys_prompt: Override system prompt (for budget calculation)
        max_out_tokens: Override max output tokens
    
    Returns:
        (context_text, used_indices, token_meta)
        - context_text: Formatted context blocks with [i] prefixes
        - used_indices: 1-based indices of chunks that fit
        - token_meta: Budget metadata for logging
    """
    prompt = sys_prompt or config.sys_prompt or SYS_PROMPT_QA
    max_out = max_out_tokens or config.max_out_tokens_default
    budget = config.get_budget()
    
    # Safety clamp: ensure room for output
    # If max_out >= budget, we'd have no room for input context
    if max_out >= budget:
        old_max = max_out
        max_out = max(128, budget // 4)
        sys.stderr.write(
            f"[context] WARNING: max_out_tokens ({old_max}) >= budget "
            f"({budget}); clamping to {max_out}\n"
        )
    
    input_budget = max(0, budget - max_out)
    
    # Reserve space for fixed prompt components
    base_cost = (
        approx_tokens(prompt)
        + approx_tokens("QUESTION:\n")
        + approx_tokens(question)
        + approx_tokens("\n\nCONTEXT:\n")
        + config.prompt_headroom
    )
    remain = max(0, input_budget - base_cost)
    
    blocks: list[str] = []
    used: list[int] = []
    
    for i, ch in enumerate(chunks, 1):
        # Build metadata suffix (PMCID or PMID)
        meta_parts = []
        if ch.get("pmcid"):
            meta_parts.append(f"PMCID:{ch['pmcid']}")
        if ch.get("pmid") and not ch.get("pmcid"):
            meta_parts.append(f"PMID:{ch['pmid']}")
        meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
        
        # Format block: [i] Title (PMCID:xxx)\nText
        title = ch.get("paper_title", "").strip()
        block = f"[{i}] {title}{meta_str}\n{ch['text']}"
        cost = approx_tokens(block) + 20  # formatting overhead
        
        if cost <= remain:
            blocks.append(block)
            used.append(i)
            remain -= cost
        else:
            break  # No more room
    
    ctx_text = "\n\n".join(blocks) if blocks else "(no context)"
    token_meta = {
        "approx_tokens": input_budget - remain,
        "budget": budget,
        "input_budget": input_budget,
        "max_out_tokens": max_out,
        "chunks_used": len(used),
        "chunks_total": len(chunks),
    }
    return ctx_text, used, token_meta


def answer_question(
    question: str,
    ranked_chunks: list[dict[str, str]],
    config: LLMConfig,
    *,
    max_out_tokens: int | None = None,
    sys_prompt: str | None = None,
) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    """Answer a question using ranked context chunks.
    
    Handles overflow by progressively trimming chunks and reducing
    max_out_tokens. Up to 4 retry attempts are made before giving up.
    
    Args:
        question: User question text
        ranked_chunks: List of chunk dicts with 'text', 'paper_title', etc.
        config: LLM configuration
        max_out_tokens: Override for config.max_out_tokens_default
        sys_prompt: Override for config.sys_prompt or SYS_PROMPT_QA
    
    Returns:
        (answer_text, final_chunks_sent, token_meta)
        - answer_text: LLM response text
        - final_chunks_sent: The exact chunks sent (after any trimming)
        - token_meta: Budget/usage metadata
    
    Raises:
        LLMError: On non-recoverable LLM errors (auth, model not found)
        ContextOverflowError: If overflow persists after all retries
    """
    provider = AutoProvider()
    prompt_template = sys_prompt or config.sys_prompt or SYS_PROMPT_QA
    max_out = max_out_tokens or config.max_out_tokens_default
    
    working_chunks = list(ranked_chunks)
    
    # Up to 4 attempts with progressive trimming
    for attempt in range(4):
        ctx_text, used_idx, token_meta = pack_context(
            working_chunks, question, config,
            sys_prompt=prompt_template,
            max_out_tokens=max_out,
        )
        
        # Build final_chunks from used indices (1-based)
        final_chunks = [working_chunks[i - 1] for i in used_idx]
        prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_text}"
        
        try:
            response = provider.generate(
                prompt, config,
                max_out_tokens=max_out,
                sys_prompt=prompt_template,
            )
            return response.text, final_chunks, token_meta
            
        except ContextOverflowError:
            if attempt >= 3:
                raise  # Give up after 4 attempts
            
            # Trim chunks more aggressively each time
            if len(working_chunks) > 1:
                if attempt == 0:
                    new_len = max(1, int(len(working_chunks) * 0.7))
                else:
                    new_len = max(1, len(working_chunks) // 2)
                working_chunks = working_chunks[:new_len]
            
            # Also reduce output allowance
            max_out = max(128, int(max_out * 0.75))
            continue
        
        except LLMError as e:
            # Non-overflow errors: generate user-friendly message
            raise LLMError(
                clarify_error(config.model, config.base_url, e)
            ) from e
    
    # Should not reach here, but satisfy type checker
    raise LLMError("Unexpected: exhausted retry attempts")
