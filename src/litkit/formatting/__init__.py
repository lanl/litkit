"""Answer formatting and citation processing for litkit.

Provides:
- normalize_answer_and_build_refs: Collapse chunk citations to doc-level
- render_references: Format bibliography block
- normalize_citations: Canonicalize raw LLM citation output
"""
from .answers import normalize_answer_and_build_refs, render_references
from .citations import normalize_citations

__all__ = [
    "normalize_answer_and_build_refs",
    "render_references",
    "normalize_citations",
]
