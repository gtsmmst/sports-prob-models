"""Poisson / xG goal model for association football.

Goals for each side are modelled as independent Poisson variables.  Everything
derives from a **scoreline matrix** ``P[i][j] = P(home scores i, away scores j)``
over the *remaining* part of the match (for pre-match, that is the whole match):

* 1X2 (home / draw / away)          :func:`match_result_probs`
* both teams to score (BTTS)        :func:`btts_prob`
* totals (over / under, any line)   :func:`over_under_probs`
* Asian handicap (any line)         :func:`asian_handicap_probs`

For in-play use, :class:`LiveMatchState` turns minute / score / xG into the
expected goals still to come (:func:`live_goal_rates`), and every market
function accepts the goals already scored so the matrix is shifted correctly.

The independence assumption is known to under-price draws and low-scoring
correlations (see Dixon & Coles, 1997); no correlation correction is applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Matrix = Sequence[Sequence[float]]

DEFAULT_MAX_GOALS = 12
"""Matrix is truncated at this many goals per side (mass lost < 1e-3 for lambda <= 4)."""


# ---------------------------------------------------------------------------
# Poisson primitives
# ---------------------------------------------------------------------------


def poisson_pmf(k: int, rate: float) -> float:
    """P(X = k) for X ~ Poisson(rate)."""
    if k < 0:
        return 0.0
    if rate < 0:
        raise ValueError("rate must be non-negative")
    if rate == 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-rate + k * math.log(rate) - math.lgamma(k + 1))


def poisson_cdf(k: int, rate: float) -> float:
    """P(X <= k) for X ~ Poisson(rate)."""
    if k < 0:
        return 0.0
    return min(1.0, sum(poisson_pmf(i, rate) for i in range(k + 1)))


def poisson_at_least(k: int, rate: float) -> float:
    """P(X >= k) for X ~ Poisson(rate)."""
    return 1.0 - poisson_cdf(k - 1, rate)


# ---------------------------------------------------------------------------
# Scoreline matrix
# ---------------------------------------------------------------------------


def scoreline_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> list[list[float]]:
    """Independent-Poisson scoreline matrix ``P[i][j]``, ``i, j in 0..max_goals``.

    The matrix is not renormalised; with ``max_goals=12`` the truncated mass is
    negligible for realistic rates.
    """
    if lambda_home < 0 or lambda_away < 0:
        raise ValueError("goal rates must be non-negative")
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")
    ph = [poisson_pmf(i, lambda_home) for i in range(max_goals + 1)]
    pa = [poisson_pmf(j, lambda_away) for j in range(max_goals + 1)]
    return [[ph[i] * pa[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]


def matrix_total(matrix: Matrix) -> float:
    """Sum of all entries (should be ~1)."""
    return sum(sum(row) for row in matrix)


def _clamp01(x: float) -> float:
    """Guard against floating-point drift such as 1.0000000000000002."""
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResultProbs:
    """1X2 probabilities."""

    home: float
    draw: float
    away: float

    def __post_init__(self) -> None:
        for f in ("home", "draw", "away"):
            object.__setattr__(self, f, _clamp01(getattr(self, f)))

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.home, self.draw, self.away)


def match_result_probs(matrix: Matrix, home_goals: int = 0, away_goals: int = 0) -> MatchResultProbs:
    """1X2 probabilities given goals already scored (``home_goals``/``away_goals``).

    The matrix describes *additional* goals; the final score is current + extra.
    """
    home = draw = away = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            margin = (home_goals + i) - (away_goals + j)
            if margin > 0:
                home += p
            elif margin == 0:
                draw += p
            else:
                away += p
    total = home + draw + away
    if total <= 0:
        raise ValueError("matrix has no probability mass")
    return MatchResultProbs(home / total, draw / total, away / total)


def btts_prob(matrix: Matrix, home_goals: int = 0, away_goals: int = 0) -> float:
    """P(both teams score by full time)."""
    total = matrix_total(matrix)
    if home_goals > 0 and away_goals > 0:
        return 1.0
    if home_goals > 0:  # only the away side still needs to score
        return _clamp01(1.0 - sum(row[0] for row in matrix) / total)
    if away_goals > 0:
        return _clamp01(1.0 - sum(matrix[0]) / total)
    p_home_blank = sum(matrix[0])
    p_away_blank = sum(row[0] for row in matrix)
    return _clamp01(1.0 - (p_home_blank + p_away_blank - matrix[0][0]) / total)


def total_goals_distribution(matrix: Matrix, home_goals: int = 0, away_goals: int = 0) -> list[float]:
    """Distribution of the final total, ``dist[t] = P(total goals == t)``."""
    n = len(matrix)
    base = home_goals + away_goals
    dist = [0.0] * (base + 2 * (n - 1) + 1)
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            dist[base + i + j] += p
    total = sum(dist)
    return [d / total for d in dist] if total > 0 else dist


@dataclass(frozen=True)
class ThreeWay:
    """Win / push / lose split used for totals and handicaps."""

    win: float
    push: float
    lose: float

    def __post_init__(self) -> None:
        for f in ("win", "push", "lose"):
            object.__setattr__(self, f, _clamp01(getattr(self, f)))

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.win, self.push, self.lose)


def _quarter_split(line: float) -> tuple[float, float] | None:
    """For quarter lines (x.25 / x.75) return the two half-stake lines, else None."""
    q = round(line * 4)
    if abs(q - line * 4) > 1e-9:
        raise ValueError(f"line must be a multiple of 0.25, got {line!r}")
    if q % 2 != 0:  # odd quarter -> quarter line
        return (line - 0.25, line + 0.25)
    return None


def over_under_probs(matrix: Matrix, line: float, home_goals: int = 0, away_goals: int = 0) -> ThreeWay:
    """(over, push, under) for a totals ``line`` (e.g. 2.5, 3, 2.75).

    Half lines never push; integer lines push on an exact hit; quarter lines
    are the average of the two neighbouring lines.
    """
    split = _quarter_split(line)
    if split is not None:
        lo, hi = (over_under_probs(matrix, L, home_goals, away_goals) for L in split)
        return ThreeWay(0.5 * (lo.win + hi.win), 0.5 * (lo.push + hi.push), 0.5 * (lo.lose + hi.lose))
    dist = total_goals_distribution(matrix, home_goals, away_goals)
    line4 = round(line * 4)
    over = push = under = 0.0
    for t, p in enumerate(dist):
        d = 4 * t - line4
        if d > 0:
            over += p
        elif d == 0:
            push += p
        else:
            under += p
    return ThreeWay(over, push, under)


def over_prob(matrix: Matrix, line: float = 2.5, home_goals: int = 0, away_goals: int = 0) -> float:
    """P(total goals > line) for a half line such as 2.5."""
    return over_under_probs(matrix, line, home_goals, away_goals).win


def asian_handicap_probs(
    matrix: Matrix,
    handicap: float,
    home_goals: int = 0,
    away_goals: int = 0,
) -> ThreeWay:
    """(win, push, lose) for the **home** side with ``handicap`` goals added.

    ``handicap=-0.5`` means home must win; ``+1`` means home + 1 goal, which
    pushes on a one-goal away win.  Quarter lines split the stake across the
    two neighbouring lines.  For the away side use ``-handicap`` and swap
    win/lose.
    """
    split = _quarter_split(handicap)
    if split is not None:
        lo, hi = (asian_handicap_probs(matrix, h, home_goals, away_goals) for h in split)
        return ThreeWay(0.5 * (lo.win + hi.win), 0.5 * (lo.push + hi.push), 0.5 * (lo.lose + hi.lose))
    h4 = round(handicap * 4)
    win = push = lose = 0.0
    total = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            margin4 = 4 * ((home_goals + i) - (away_goals + j)) + h4
            total += p
            if margin4 > 0:
                win += p
            elif margin4 == 0:
                push += p
            else:
                lose += p
    if total <= 0:
        raise ValueError("matrix has no probability mass")
    return ThreeWay(win / total, push / total, lose / total)


# ---------------------------------------------------------------------------
# Pre-match helpers
# ---------------------------------------------------------------------------


def prematch_btts_prob(
    lambda_home: float,
    lambda_away: float,
    league_rate: float | None = None,
    blend: float = 0.5,
) -> float:
    """Pre-match BTTS: ``(1 - e^-lh) * (1 - e^-la)``, optionally blended with a
    league base rate (``blend`` = weight on the Poisson estimate)."""
    p = _clamp01(poisson_at_least(1, lambda_home) * poisson_at_least(1, lambda_away))
    if league_rate is None:
        return p
    if not (0.0 <= league_rate <= 1.0) or not (0.0 <= blend <= 1.0):
        raise ValueError("league_rate and blend must lie in [0, 1]")
    return blend * p + (1.0 - blend) * league_rate


# ---------------------------------------------------------------------------
# Live state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveMatchState:
    """Snapshot of a match in progress.

    ``avg_goals_home`` / ``avg_goals_away`` are the sides' full-match expected
    goals (pre-match rates).  ``xg_home`` / ``xg_away`` are the expected goals
    accumulated so far.  ``shots_on_target_*`` add a small pressure bonus.
    """

    minute: float = 0.0
    home_goals: int = 0
    away_goals: int = 0
    xg_home: float = 0.0
    xg_away: float = 0.0
    shots_on_target_home: int = 0
    shots_on_target_away: int = 0
    avg_goals_home: float = 1.4
    avg_goals_away: float = 1.2
    match_minutes: float = 90.0
    time_remaining: float | None = None

    def __post_init__(self) -> None:
        if self.minute < 0 or self.home_goals < 0 or self.away_goals < 0:
            raise ValueError("minute and goals must be non-negative")
        if self.time_remaining is None:
            object.__setattr__(self, "time_remaining", max(0.0, self.match_minutes - self.minute))


XG_BLEND_CURRENT = 0.7
"""Weight on the observed xG pace (vs the pre-match rate) when extrapolating."""

SHOT_BONUS_PER_SHOT = 0.02
"""Extra expected goals per shot on target, scaled by the fraction of time left."""


def _remaining_rate(
    minute: float,
    time_remaining: float,
    match_minutes: float,
    xg: float,
    avg_goals: float,
    shots: int,
) -> float:
    if time_remaining <= 0:
        return 0.0
    if minute <= 0:
        return avg_goals
    historical_rate = avg_goals / match_minutes
    current_rate = xg / minute
    blended = XG_BLEND_CURRENT * current_rate + (1.0 - XG_BLEND_CURRENT) * historical_rate
    fraction_remaining = max(0.0, min(1.0, time_remaining / match_minutes))
    return max(0.0, blended * time_remaining + shots * SHOT_BONUS_PER_SHOT * fraction_remaining)


def live_goal_rates(state: LiveMatchState) -> tuple[float, float]:
    """Expected goals still to come for (home, away).

    Blends the observed xG pace (70 %) with the pre-match rate (30 %), scales by
    time remaining and adds a small shots-on-target bonus.  Before kick-off the
    pre-match rates are returned unchanged.
    """
    tr = state.time_remaining or 0.0
    home = _remaining_rate(
        state.minute, tr, state.match_minutes, state.xg_home, state.avg_goals_home, state.shots_on_target_home
    )
    away = _remaining_rate(
        state.minute, tr, state.match_minutes, state.xg_away, state.avg_goals_away, state.shots_on_target_away
    )
    return home, away


def live_scoreline_matrix(state: LiveMatchState, max_goals: int = DEFAULT_MAX_GOALS) -> list[list[float]]:
    """Scoreline matrix of the *additional* goals from now to full time."""
    lh, la = live_goal_rates(state)
    return scoreline_matrix(lh, la, max_goals)


@dataclass(frozen=True)
class LiveProbs:
    """Bundle of in-play market probabilities."""

    home: float
    draw: float
    away: float
    btts: float
    over_2_5: float
    lambda_home: float
    lambda_away: float


def live_probs(state: LiveMatchState, max_goals: int = DEFAULT_MAX_GOALS) -> LiveProbs:
    """1X2, BTTS and over 2.5 for a live match in one call."""
    lh, la = live_goal_rates(state)
    m = scoreline_matrix(lh, la, max_goals)
    r = match_result_probs(m, state.home_goals, state.away_goals)
    return LiveProbs(
        home=r.home,
        draw=r.draw,
        away=r.away,
        btts=btts_prob(m, state.home_goals, state.away_goals),
        over_2_5=over_prob(m, 2.5, state.home_goals, state.away_goals),
        lambda_home=lh,
        lambda_away=la,
    )


__all__ = [
    "DEFAULT_MAX_GOALS",
    "LiveMatchState",
    "LiveProbs",
    "MatchResultProbs",
    "ThreeWay",
    "asian_handicap_probs",
    "btts_prob",
    "live_goal_rates",
    "live_probs",
    "live_scoreline_matrix",
    "match_result_probs",
    "matrix_total",
    "over_prob",
    "over_under_probs",
    "poisson_at_least",
    "poisson_cdf",
    "poisson_pmf",
    "prematch_btts_prob",
    "scoreline_matrix",
    "total_goals_distribution",
]
