"""In-play win-probability model for Counter-Strike 2 (CS2).

The model is hierarchical:

    round win probability  ->  map win probability  ->  series (best-of-N) win probability

* **Round**: a logistic-style ratio of the two teams' strength ratings, with a
  multiplicative bonus for the team on the CT side (CT wins roughly 53-56 % of
  rounds in professional play; ``ct_advantage=1.15`` gives ~53.5 % for equal
  teams).
* **Map**: either a Monte-Carlo simulation of the remaining rounds
  (:func:`map_win_prob`) or an exact dynamic-programming solution over the
  score state (:func:`map_win_prob_exact`).  Both honour MR12 / MR15 halves,
  side swaps at half time and MR3 overtime periods.
* **Series**: a tiny dynamic program over maps won (:func:`series_win_prob`).

All functions are pure.  The Monte-Carlo path takes an explicit ``seed`` so
results are reproducible.

Conventions
-----------
``strength_a`` / ``strength_b`` are positive ratings on an arbitrary scale
where 1.0 is "median team".  A 1.2-vs-1.0 pairing gives the stronger team a
54.5 % chance of winning a side-neutral round before the CT bonus.

Overtime sides follow the common MR3 convention: teams keep the side they
played in the second half for the first overtime half, then swap every
``ot_half_rounds`` rounds.  Tournament rule-sets differ; edit
:func:`side_for_round` if yours does.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Mapping

Side = Literal["CT", "T"]

DEFAULT_CT_ADVANTAGE = 1.15
"""Multiplier applied to the CT-side team's strength in a regular round."""

PISTOL_STRENGTH_WEIGHT = 0.7
"""How much of the strength gap survives in a pistol round (they are noisier)."""

PISTOL_CT_ADVANTAGE = 1.07
"""Reduced CT bonus for pistol rounds."""

ROUND_PROB_CLAMP = (0.05, 0.95)
"""Round probabilities are clamped into this interval to avoid degenerate maps."""


# ---------------------------------------------------------------------------
# Team strength
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamStrength:
    """A team and its strength rating (1.0 = median; higher is stronger)."""

    name: str
    rating: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TeamStrength.name must be non-empty")
        if not (self.rating > 0):
            raise ValueError(f"TeamStrength.rating must be positive, got {self.rating!r}")


def lookup_strength(
    name: str,
    table: Mapping[str, float],
    default: float = 1.0,
) -> TeamStrength:
    """Case-insensitive lookup of ``name`` in a ``{team: rating}`` table.

    Unknown teams get ``default`` (the median rating).  The table is supplied
    by the caller; this package ships no hard-coded ratings.
    """
    key = name.strip().lower()
    for team, rating in table.items():
        if team.strip().lower() == key:
            return TeamStrength(name=team, rating=float(rating))
    return TeamStrength(name=name.strip() or name, rating=float(default))


# ---------------------------------------------------------------------------
# Round level
# ---------------------------------------------------------------------------


def _clamp(p: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, p))


def round_win_prob(
    strength_a: float,
    strength_b: float,
    side_a: Side,
    ct_advantage: float = DEFAULT_CT_ADVANTAGE,
    clamp: tuple[float, float] = ROUND_PROB_CLAMP,
) -> float:
    """Probability that team A wins the current round.

    ``p = s_a' / (s_a' + s_b')`` where the CT-side team's strength is
    multiplied by ``ct_advantage``.  The result is clamped into ``clamp``.
    """
    if strength_a <= 0 or strength_b <= 0:
        raise ValueError("strengths must be positive")
    if side_a not in ("CT", "T"):
        raise ValueError(f"side_a must be 'CT' or 'T', got {side_a!r}")
    if side_a == "CT":
        eff_a, eff_b = strength_a * ct_advantage, strength_b
    else:
        eff_a, eff_b = strength_a, strength_b * ct_advantage
    return _clamp(eff_a / (eff_a + eff_b), *clamp)


def pistol_round_win_prob(
    strength_a: float,
    strength_b: float,
    side_a: Side,
    strength_weight: float = PISTOL_STRENGTH_WEIGHT,
    ct_advantage: float = PISTOL_CT_ADVANTAGE,
) -> float:
    """Round-win probability for a pistol round.

    Pistol rounds are higher-variance, so the strength gap is shrunk towards
    the median (``1 + (s - 1) * strength_weight``) and a smaller CT bonus is
    used.
    """
    sa = 1.0 + (strength_a - 1.0) * strength_weight
    sb = 1.0 + (strength_b - 1.0) * strength_weight
    return round_win_prob(sa, sb, side_a, ct_advantage=ct_advantage)


# ---------------------------------------------------------------------------
# Map level
# ---------------------------------------------------------------------------


def side_for_round(
    round_index: int,
    a_starts_ct: bool,
    half_rounds: int = 12,
    ot_half_rounds: int = 3,
) -> Side:
    """Which side team A plays in round number ``round_index`` (0-based).

    * rounds ``0 .. half_rounds-1``: starting side
    * rounds ``half_rounds .. 2*half_rounds-1``: swapped
    * overtime: keep the second-half side for the first ``ot_half_rounds``
      rounds, then swap every ``ot_half_rounds`` rounds.
    """
    if round_index < 0:
        raise ValueError("round_index must be >= 0")
    regulation = 2 * half_rounds
    if round_index < half_rounds:
        a_ct = a_starts_ct
    elif round_index < regulation:
        a_ct = not a_starts_ct
    else:
        swaps = (round_index - regulation) // ot_half_rounds
        a_ct = (not a_starts_ct) if swaps % 2 == 0 else a_starts_ct
    return "CT" if a_ct else "T"


def map_winner(
    score_a: int,
    score_b: int,
    half_rounds: int = 12,
    ot_half_rounds: int = 3,
) -> int:
    """Return ``1`` if A has won the map, ``-1`` if B has, ``0`` if still live.

    Regulation is won by reaching ``half_rounds + 1`` while the opponent has at
    most ``half_rounds - 1``.  A ``half_rounds : half_rounds`` tie goes to
    overtime, played in periods of ``2 * ot_half_rounds`` rounds; a period is
    won by taking ``ot_half_rounds + 1`` of its rounds, and a tied period
    triggers another.
    """
    if score_a < 0 or score_b < 0:
        raise ValueError("scores must be non-negative")
    target = half_rounds + 1
    hi, lo = max(score_a, score_b), min(score_a, score_b)
    if hi < target:
        return 0
    leader = 1 if score_a > score_b else -1
    if lo < half_rounds:
        return leader  # regulation win
    # Overtime: locate the current period's base score.
    base = half_rounds + ot_half_rounds * ((lo - half_rounds) // ot_half_rounds)
    if hi >= base + ot_half_rounds + 1 and lo <= base + ot_half_rounds - 1:
        return leader
    return 0


def map_win_prob(
    score_a: int,
    score_b: int,
    a_starts_ct: bool,
    strength_a: float,
    strength_b: float,
    *,
    half_rounds: int = 12,
    ot_half_rounds: int = 3,
    n_sims: int = 3000,
    seed: int = 0,
    ct_advantage: float = DEFAULT_CT_ADVANTAGE,
) -> float:
    """Monte-Carlo estimate of P(A wins the map) from the current score.

    Each simulation plays out the remaining rounds with
    :func:`round_win_prob`, using :func:`side_for_round` to decide who is CT.
    ``seed`` makes the estimate reproducible.  For an exact answer use
    :func:`map_win_prob_exact`.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    decided = map_winner(score_a, score_b, half_rounds, ot_half_rounds)
    if decided:
        return 1.0 if decided > 0 else 0.0

    rng = random.Random(seed)
    wins = 0
    for _ in range(n_sims):
        sa, sb = score_a, score_b
        while True:
            side = side_for_round(sa + sb, a_starts_ct, half_rounds, ot_half_rounds)
            p = round_win_prob(strength_a, strength_b, side, ct_advantage)
            if rng.random() < p:
                sa += 1
            else:
                sb += 1
            outcome = map_winner(sa, sb, half_rounds, ot_half_rounds)
            if outcome:
                wins += outcome > 0
                break
    return wins / n_sims


def map_win_prob_exact(
    score_a: int,
    score_b: int,
    a_starts_ct: bool,
    strength_a: float,
    strength_b: float,
    *,
    half_rounds: int = 12,
    ot_half_rounds: int = 3,
    ct_advantage: float = DEFAULT_CT_ADVANTAGE,
    max_ot_periods: int = 8,
) -> float:
    """Exact P(A wins the map) by dynamic programming over the score state.

    Because the side in play depends only on the number of rounds played, the
    state ``(score_a, score_b)`` fully determines the future.  Overtime is
    truncated after ``max_ot_periods`` periods (the residual probability mass
    is negligible and is split 50/50).
    """
    decided = map_winner(score_a, score_b, half_rounds, ot_half_rounds)
    if decided:
        return 1.0 if decided > 0 else 0.0

    max_rounds = 2 * half_rounds + max_ot_periods * 2 * ot_half_rounds
    memo: dict[tuple[int, int], float] = {}

    def solve(sa: int, sb: int) -> float:
        key = (sa, sb)
        if key in memo:
            return memo[key]
        outcome = map_winner(sa, sb, half_rounds, ot_half_rounds)
        if outcome:
            val = 1.0 if outcome > 0 else 0.0
        elif sa + sb >= max_rounds:
            val = 0.5
        else:
            side = side_for_round(sa + sb, a_starts_ct, half_rounds, ot_half_rounds)
            p = round_win_prob(strength_a, strength_b, side, ct_advantage)
            val = p * solve(sa + 1, sb) + (1.0 - p) * solve(sa, sb + 1)
        memo[key] = val
        return val

    return solve(score_a, score_b)


def neutral_map_win_prob(
    strength_a: float,
    strength_b: float,
    *,
    half_rounds: int = 12,
    ot_half_rounds: int = 3,
    ct_advantage: float = DEFAULT_CT_ADVANTAGE,
) -> float:
    """P(A wins a map that has not started), averaged over who starts CT."""
    kwargs = dict(half_rounds=half_rounds, ot_half_rounds=ot_half_rounds, ct_advantage=ct_advantage)
    p_ct = map_win_prob_exact(0, 0, True, strength_a, strength_b, **kwargs)
    p_t = map_win_prob_exact(0, 0, False, strength_a, strength_b, **kwargs)
    return 0.5 * (p_ct + p_t)


# ---------------------------------------------------------------------------
# Series level
# ---------------------------------------------------------------------------


def series_win_prob(
    maps_won_a: int,
    maps_won_b: int,
    p_future_map: float,
    p_current_map: float | None = None,
    best_of: int = 3,
) -> float:
    """P(A wins a best-of-``best_of`` series) from the current map score.

    ``p_current_map`` is P(A wins the map in progress); pass ``None`` when no
    map is in progress (it then equals ``p_future_map``).  ``p_future_map`` is
    P(A wins each not-yet-started map), assumed i.i.d.

    Closed form for 0-0 with ``p = p_future_map = p_current_map``:
    BO3 -> ``p**2 * (3 - 2p)``, BO5 -> ``p**3 * (1 + 3q + 6q**2)`` with ``q = 1-p``.
    """
    if best_of <= 0 or best_of % 2 == 0:
        raise ValueError("best_of must be a positive odd number")
    if maps_won_a < 0 or maps_won_b < 0:
        raise ValueError("maps won must be non-negative")
    for p in (p_future_map, p_current_map):
        if p is not None and not (0.0 <= p <= 1.0):
            raise ValueError(f"probabilities must lie in [0, 1], got {p!r}")

    need = best_of // 2 + 1
    if maps_won_a >= need:
        return 1.0
    if maps_won_b >= need:
        return 0.0
    if p_current_map is None:
        p_current_map = p_future_map

    @lru_cache(maxsize=None)
    def dp(a: int, b: int) -> float:
        if a >= need:
            return 1.0
        if b >= need:
            return 0.0
        return p_future_map * dp(a + 1, b) + (1.0 - p_future_map) * dp(a, b + 1)

    return p_current_map * dp(maps_won_a + 1, maps_won_b) + (1.0 - p_current_map) * dp(
        maps_won_a, maps_won_b + 1
    )


def match_win_prob(
    strength_a: float,
    strength_b: float,
    *,
    best_of: int = 3,
    maps_won_a: int = 0,
    maps_won_b: int = 0,
    map_in_progress: bool = False,
    score_a: int = 0,
    score_b: int = 0,
    a_starts_ct: bool = True,
    half_rounds: int = 12,
    ot_half_rounds: int = 3,
    ct_advantage: float = DEFAULT_CT_ADVANTAGE,
    method: Literal["exact", "mc"] = "exact",
    n_sims: int = 3000,
    seed: int = 0,
) -> float:
    """Convenience wrapper: strengths + live state -> P(A wins the series).

    ``method="exact"`` uses :func:`map_win_prob_exact` for the live map,
    ``method="mc"`` uses the seeded Monte-Carlo :func:`map_win_prob`.
    Future maps always use :func:`neutral_map_win_prob`.
    """
    p_future = neutral_map_win_prob(
        strength_a, strength_b, half_rounds=half_rounds, ot_half_rounds=ot_half_rounds, ct_advantage=ct_advantage
    )
    p_current: float | None = None
    if map_in_progress:
        if method == "exact":
            p_current = map_win_prob_exact(
                score_a, score_b, a_starts_ct, strength_a, strength_b,
                half_rounds=half_rounds, ot_half_rounds=ot_half_rounds, ct_advantage=ct_advantage,
            )
        elif method == "mc":
            p_current = map_win_prob(
                score_a, score_b, a_starts_ct, strength_a, strength_b,
                half_rounds=half_rounds, ot_half_rounds=ot_half_rounds,
                n_sims=n_sims, seed=seed, ct_advantage=ct_advantage,
            )
        else:
            raise ValueError(f"unknown method {method!r}")
    return series_win_prob(maps_won_a, maps_won_b, p_future, p_current, best_of=best_of)


__all__ = [
    "DEFAULT_CT_ADVANTAGE",
    "PISTOL_CT_ADVANTAGE",
    "PISTOL_STRENGTH_WEIGHT",
    "ROUND_PROB_CLAMP",
    "Side",
    "TeamStrength",
    "lookup_strength",
    "round_win_prob",
    "pistol_round_win_prob",
    "side_for_round",
    "map_winner",
    "map_win_prob",
    "map_win_prob_exact",
    "neutral_map_win_prob",
    "series_win_prob",
    "match_win_prob",
]
