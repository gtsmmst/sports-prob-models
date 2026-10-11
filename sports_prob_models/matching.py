"""Team-name normalisation and fuzzy matching.

Different venues spell the same team differently ("FaZe Clan", "faze",
"FAZE CLAN CS2").  These helpers reduce names to a canonical slug so markets
from two sources can be paired by a deterministic *match signature*.

No team table is shipped; pass your own ``aliases`` mapping
(``{"natus vincere": "navi", "na'vi": "navi"}``) where needed.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable, Mapping

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")

GENERIC_WORDS = frozenset(
    {"esports", "esport", "gaming", "team", "clan", "club", "fc", "cf", "sc", "afc", "cs2", "cs"}
)
"""Tokens dropped from a name when other tokens remain ("Team Spirit" -> "spirit")."""


def _clean(name: str) -> str:
    cleaned = _PUNCT_RE.sub(" ", name.lower().strip())
    return _SPACE_RE.sub(" ", cleaned).strip()


def normalize_name(
    name: str,
    aliases: Mapping[str, str] | None = None,
    strip_generic: bool = True,
) -> str:
    """Canonical slug for a team name.

    Steps: lower-case and strip -> exact alias lookup (raw, then with
    punctuation removed) -> whole-word alias match inside the name -> drop
    generic words such as "team"/"esports" -> join remaining tokens.

    >>> normalize_name("FaZe Clan")
    'faze'
    >>> normalize_name("Natus Vincere", {"natus vincere": "navi"})
    'navi'
    """
    if not name or not name.strip():
        return ""
    raw = name.lower().strip()
    cleaned = _clean(name)
    if aliases:
        alias_map = {k.lower().strip(): v for k, v in aliases.items()}
        if raw in alias_map:
            return alias_map[raw]
        if cleaned in alias_map:
            return alias_map[cleaned]
        # Whole-word containment, longest alias first so "team liquid" beats "liquid".
        for alias in sorted(alias_map, key=len, reverse=True):
            alias_clean = _clean(alias)
            if alias_clean and re.search(rf"(?<!\w){re.escape(alias_clean)}(?!\w)", cleaned):
                return alias_map[alias]
    tokens = cleaned.split()
    if strip_generic and len(tokens) > 1:
        kept = [t for t in tokens if t not in GENERIC_WORDS]
        if kept:
            tokens = kept
    return "".join(tokens)


def match_signature(team_a: str, team_b: str, aliases: Mapping[str, str] | None = None) -> str:
    """Order-independent key for a fixture: ``"slug1__slug2"`` (sorted)."""
    a, b = normalize_name(team_a, aliases), normalize_name(team_b, aliases)
    if not a or not b:
        raise ValueError("both team names must be non-empty")
    if a > b:
        a, b = b, a
    return f"{a}__{b}"


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fuzzy_match(
    query: str,
    candidates: Iterable[str],
    cutoff: float = 0.8,
    aliases: Mapping[str, str] | None = None,
    max_edits: int = 2,
) -> str | None:
    """Best candidate slug for ``query`` or ``None``.

    Order of attempts:

    1. exact match after :func:`normalize_name`;
    2. :func:`difflib.get_close_matches` with similarity ``cutoff``;
    3. smallest :func:`edit_distance` if it is at most ``max_edits`` and at most
       half the query length (so very short names cannot match loosely).
    """
    cands = list(candidates)
    if not query or not cands:
        return None
    norm = normalize_name(query, aliases)
    if not norm:
        return None
    if norm in cands:
        return norm
    close = difflib.get_close_matches(norm, cands, n=1, cutoff=cutoff)
    if close:
        return close[0]
    best: tuple[int, float, str] | None = None
    for cand in cands:
        d = edit_distance(norm, cand)
        if d > max_edits or d > len(norm) // 2:
            continue
        ratio = difflib.SequenceMatcher(None, norm, cand).ratio()
        key = (d, -ratio, cand)
        if best is None or key < best:
            best = key
    return best[2] if best else None


# ---------------------------------------------------------------------------
# Market-question parsing
# ---------------------------------------------------------------------------

_NAME = r"[A-Za-z0-9][A-Za-z0-9 .'`\-]{0,40}?"
_PATTERNS = [
    # "Will FaZe beat NaVi in Map 1?"
    re.compile(
        rf"\bwill\s+(?P<a>{_NAME})\s+(?:beat|defeat)\s+(?P<b>{_NAME})(?=\s+(?:in|on|at)\b|\s*[-–—:?,(]|\s*$)",
        re.IGNORECASE,
    ),
    # "FaZe vs NaVi - Map 1 Winner", "Map 1: FaZe vs. NaVi"
    re.compile(
        rf"(?:^|:\s*|\bwins?\s+)(?P<a>{_NAME})\s+vs?\.?\s+(?P<b>{_NAME})"
        r"(?=\s*[-–—:?,(]|\s+(?:map|game|match|winner|series|bo\d)\b|\s*$)",
        re.IGNORECASE,
    ),
    # "FaZe against NaVi"
    re.compile(
        rf"(?P<a>{_NAME})\s+against\s+(?P<b>{_NAME})(?=\s*[-–—:?,(]|\s+(?:in|on|at|map|game|match)\b|\s*$)",
        re.IGNORECASE,
    ),
]

_STOP_WORDS = frozenset(
    {"will", "who", "which", "the", "winner", "win", "wins", "map", "game", "match", "round", "series", "of", "in"}
)


def _clean_team(raw: str) -> str | None:
    cleaned = raw.strip(" -–—.?,!:")
    if len(cleaned) < 2 or cleaned.lower() in _STOP_WORDS:
        return None
    return cleaned


def extract_teams_from_question(question: str) -> tuple[str, str] | None:
    """Pull ``(team_a, team_b)`` out of a market title such as
    ``"Will FaZe beat NaVi in Map 1?"`` or ``"Map 1: FaZe vs NaVi"``.

    Returns ``None`` when no supported pattern matches.
    """
    if not question:
        return None
    for pat in _PATTERNS:
        m = pat.search(question)
        if m:
            a, b = _clean_team(m.group("a")), _clean_team(m.group("b"))
            if a and b:
                return a, b
    return None


__all__ = [
    "GENERIC_WORDS",
    "edit_distance",
    "extract_teams_from_question",
    "fuzzy_match",
    "match_signature",
    "normalize_name",
]
