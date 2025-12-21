"""Citation normalization utilities.

Normalizes LLM-generated citations (e.g., [1], [2†L1-L8]) into a canonical
format for downstream bibliography matching.
"""
import re
import unicodedata

# Regex for line-location tails like "†L1–L8" in citations
_CITATION_LINELOC = re.compile(
    r"([\[【]\s*\d+)\s*†L\d+(?:[–—-]\d+)?(\s*[】\]])"
)

# General bracket grabber for rebuilding citations
_CITATION_ANYBR = re.compile(
    r"(?P<open>[\[【])(?P<inside>[^\[\]【】]{0,200}?)(?P<close>[】\]])"
)

# Leading numeric list pattern (supports ASCII/CJK commas)
_LEADING_NUM_LIST = re.compile(r"^\s*(\d+(?:\s*[，、,]\s*\d+)*)")


def normalize_citations(text: str) -> str:
    """Canonically normalize bracketed numeric citations.
    
    1. Convert fullwidth brackets to ASCII [ ]
    2. Remove '†Lx–Ly' / location tails by only keeping leading numeric list
    3. Normalize commas/spacing, de-duplicate while preserving order
    
    Args:
        text: Raw text containing citations like [1], [2†L1-L8], etc.
    
    Returns:
        Text with normalized citations like [1], [2]
    
    Examples:
        "[1†L1–L8]" -> "[1]"
        "【1, 2, 1】" -> "[1, 2]"
        "[1,2,3]" -> "[1, 2, 3]"
    """
    try:
        s = unicodedata.normalize("NFKC", text)
    except Exception:
        s = text

    # Strip common zero-width junk
    for z in ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"):
        s = s.replace(z, "")

    # Unify bracket style (fullwidth -> ASCII)
    s = s.replace("【", "[").replace("】", "]")

    # Collapse explicit †L tails like "[2†L1–L8]" -> "[2]"
    s = _CITATION_LINELOC.sub(r"\1\2", s)

    # Rebuild each bracketed group from just the leading numeric list
    def _rebuild(m: re.Match) -> str:
        inside = m.group("inside")
        lead = _LEADING_NUM_LIST.match(inside)
        if not lead:
            # Not a numeric citation; leave untouched (e.g., [Note])
            return m.group(0)

        nums_str = lead.group(1)
        parts = [p.strip() for p in re.split(r"[，、,]", nums_str) if p.strip()]

        out, seen = [], set()
        for p in parts:
            if p.isdigit():
                n = int(p)
                if n not in seen:
                    seen.add(n)
                    out.append(n)

        if not out:
            # Nothing numeric survived; drop the brackets
            return ""

        return "[" + ", ".join(str(n) for n in out) + "]"

    return _CITATION_ANYBR.sub(_rebuild, s)
