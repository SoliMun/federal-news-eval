import re
from urllib.parse import urlparse

# Allowed domains for source URLs
ALLOWED_DOMAINS = ('.gov', '.mil')
# Optional non-.gov/.mil exceptions (leaving empty to keep it strict)
ALLOWED_ORGS = set()

# ---------- Bullet helpers ----------


def split_lines(summary: str):
    """Return non-empty, stripped lines."""
    return [ln.strip() for ln in (summary or "").splitlines() if ln.strip()]


def is_bullet_line(line: str) -> bool:
    """A valid bullet starts with '- ' followed by a non-whitespace."""
    return bool(re.match(r'^\s*-\s+\S', line or ""))


def bullet_format_ok(summary: str) -> bool:
    """All lines present must be valid bullets."""
    lines = split_lines(summary)
    return len(lines) > 0 and all(is_bullet_line(ln) for ln in lines)


def bullet_count(summary: str) -> int:
    """Count only valid bullet lines."""
    return sum(1 for ln in split_lines(summary) if is_bullet_line(ln))


def bullet_count_in_range(summary: str, lo: int = 3, hi: int = 5) -> bool:
    """Check bullet count falls in range."""
    n = bullet_count(summary)
    return lo <= n <= hi


# ---------- Word counting ----------


def word_count(text: str) -> int:
    """Token-agnostic word count."""
    return len(re.findall(r'\b\w+\b', text or ""))


def bullet_word_counts(summary: str):
    """Word counts per bullet (strips the '- ' prefix)."""
    counts = []
    for ln in split_lines(summary):
        if is_bullet_line(ln):
            body = ln[2:].strip() if ln.startswith("- ") else ln.strip()
            counts.append(word_count(body))
    return counts


def total_words_in_range(summary: str, lo: int = 70, hi: int = 250) -> bool:
    """Total words across all bullets within [lo, hi]."""
    return lo <= word_count(summary) <= hi


def per_bullet_len_ok(summary: str, lo: int = 12, hi: int = 40) -> bool:
    """
    Every bullet must be within [lo, hi] words (Stricter than averaging).
    """
    counts = bullet_word_counts(summary)
    if not counts:
        return False
    return all(lo <= c <= hi for c in counts)


# ---------- URL allowlist ----------


def domain_allowlist_ok(url: str) -> bool:
    """
    URL host must end with .gov/.mil OR be explicitly allowlisted.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if any(host.endswith(suf) for suf in ALLOWED_DOMAINS):
        return True
    return host in ALLOWED_ORGS
