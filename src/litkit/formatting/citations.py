"""Citation normalization utilities.

Normalizes LLM-generated citations (e.g., [1], [2†L1-L8]) into a canonical
format for downstream bibliography matching.
"""

import re
import unicodedata

# Regex for line-location tails like "†L1–L8" in citations
_CITATION_LINELOC = re.compile(r"([\[【]\s*\d+)\s*†L\d+(?:[–—-]\d+)?(\s*[】\]])")

# General bracket grabber for rebuilding citations
_CITATION_ANYBR = re.compile(r"(?P<open>[\[【])(?P<inside>[^\[\]【】]{0,200}?)(?P<close>[】\]])")

# Require the entire bracket to be a citation so prose such as "[3 patients]"
# stays intact. Keep range endpoints for the answer parser to expand.
_CITATION_ITEM = r"\d+(?:\s*[-–—]\s*\d+)?"
_LIST_SEPARATOR = r"[，、,;；]"
_CITATION_LIST = re.compile(
    rf"\s*(?P<items>{_CITATION_ITEM}(?:\s*{_LIST_SEPARATOR}\s*{_CITATION_ITEM})*)"
    r"(?:\s*(?:†L\d+(?:[–—-]L?\d+)?|:\d+†\S+))?\s*"
)


def normalize_citations(text: str) -> str:
    """Canonically normalize bracketed numeric citations.

    1. Convert fullwidth brackets to ASCII [ ]
    2. Remove '†Lx–Ly' / location tails by only keeping the numeric list
    3. Normalize separators/spacing, de-duplicate while preserving order;
       brackets that are not a pure numeric list are left untouched

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

    # Rebuild each bracketed group, but only when the whole bracket is a
    # numeric citation list. Ordinary prose is left exactly as written.
    def _rebuild(m: re.Match) -> str:
        citation = _CITATION_LIST.fullmatch(m.group("inside"))
        if citation is None:
            return m.group(0)

        items = re.split(_LIST_SEPARATOR, citation.group("items"))
        # Canonical decimal IDs preserve the original handling of leading zeros.
        normalized = (re.sub(r"\d+", lambda n: str(int(n[0])), item.strip()) for item in items)
        return "[" + ", ".join(dict.fromkeys(normalized)) + "]"

    return _CITATION_ANYBR.sub(_rebuild, s)
