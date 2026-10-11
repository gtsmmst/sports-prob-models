"""Model probability vs market price: edge, fair price, expected value.

Prices are binary-contract prices in ``(0, 1)`` (a YES share paying 1).
Bookmaker decimal odds convert with :func:`implied_prob`.

Fees on exchanges / prediction markets are usually charged on *profit*.  A
share bought at ``price`` with fee ``fee`` has net payout ``1 - (1 - price) * fee``,
so the price you effectively pay per unit of net payout is

    effective_price = price / (1 - (1 - price) * fee)

(:func:`effective_price`).  Edges and EV below use this fee-adjusted cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .kelly import kelly_fraction

Side = Literal["YES", "NO", "PASS"]


def implied_prob(decimal_odds: float) -> float:
    """Implied probability of decimal odds (includes the bookmaker margin)."""
    if not (decimal_odds > 1.0):
        raise ValueError(f"decimal_odds must exceed 1, got {decimal_odds!r}")
    return 1.0 / decimal_odds


def fair_price(p: float) -> float:
    """Break-even contract price for a true probability ``p`` (fee-free)."""
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must lie in [0, 1]")
    return p


def fair_decimal_odds(p: float) -> float:
    """Break-even decimal odds for a true probability ``p``."""
    if not (0.0 < p <= 1.0):
        raise ValueError("p must lie in (0, 1]")
    return 1.0 / p


def remove_overround(implied: Sequence[float]) -> list[float]:
    """Normalise a set of implied probabilities to sum to 1 (proportional method)."""
    total = sum(implied)
    if total <= 0:
        raise ValueError("implied probabilities must sum to a positive number")
    return [x / total for x in implied]


def effective_price(price: float, fee: float = 0.0) -> float:
    """Price per unit of *net* payout after a fee on profit."""
    if not (0.0 < price < 1.0):
        raise ValueError(f"price must lie in (0, 1), got {price!r}")
    if not (0.0 <= fee < 1.0):
        raise ValueError(f"fee must lie in [0, 1), got {fee!r}")
    return price / (1.0 - (1.0 - price) * fee)


def edge_abs(p: float, price: float, fee: float = 0.0) -> float:
    """Absolute edge in probability units: ``p - effective_price``."""
    return p - effective_price(price, fee)


def edge_rel(p: float, price: float, fee: float = 0.0) -> float:
    """Relative edge: ``p / effective_price - 1`` (equals EV per unit staked)."""
    return p / effective_price(price, fee) - 1.0


def ev_per_dollar(p: float, price: float, fee: float = 0.0) -> float:
    """Expected profit per 1 unit staked on a contract at ``price``.

    Staking 1 buys ``1/price`` shares; a win returns ``n - (n - 1) * fee``.
    ``EV = p * (n - (n - 1) * fee) - 1``.  Equals :func:`edge_rel`.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must lie in [0, 1]")
    if not (0.0 < price < 1.0):
        raise ValueError(f"price must lie in (0, 1), got {price!r}")
    n = 1.0 / price
    return p * (n - (n - 1.0) * fee) - 1.0


@dataclass(frozen=True)
class EdgeSignal:
    """Result of comparing a model probability with a two-sided market."""

    side: Side
    """Which contract to buy, or ``"PASS"``."""
    model_prob: float
    """Model probability that the chosen side pays out."""
    price: float
    """Quoted price of the chosen side."""
    edge: float
    """``model_prob - effective_price`` (probability units)."""
    ev_per_dollar: float
    """Expected profit per unit staked."""
    kelly_full: float
    """Full-Kelly fraction for the chosen side (0 for PASS)."""


def evaluate_binary_market(
    model_p: float,
    yes_price: float,
    *,
    no_price: float | None = None,
    fee: float = 0.0,
    min_edge: float = 0.0,
) -> EdgeSignal:
    """Compare ``model_p`` (P(YES)) with the YES and NO prices and pick a side.

    ``no_price`` defaults to ``1 - yes_price``.  The side with the larger
    fee-adjusted edge is chosen; if that edge is below ``min_edge`` the signal
    is ``PASS`` (with the better side's numbers still reported).
    """
    if not (0.0 <= model_p <= 1.0):
        raise ValueError("model_p must lie in [0, 1]")
    if no_price is None:
        no_price = 1.0 - yes_price
    candidates = (
        ("YES", model_p, yes_price),
        ("NO", 1.0 - model_p, no_price),
    )
    best: tuple[str, float, float, float] | None = None
    for side, p, price in candidates:
        if not (0.0 < price < 1.0):
            continue
        e = edge_abs(p, price, fee)
        if best is None or e > best[3]:
            best = (side, p, price, e)
    if best is None:
        raise ValueError("at least one side must have a price strictly between 0 and 1")
    side, p, price, e = best
    if e < max(min_edge, 0.0) or e <= 0.0:
        return EdgeSignal("PASS", p, price, e, ev_per_dollar(p, price, fee), 0.0)
    return EdgeSignal(
        side,  # type: ignore[arg-type]
        p,
        price,
        e,
        ev_per_dollar(p, price, fee),
        kelly_fraction(p, price, fee=fee),
    )


__all__ = [
    "EdgeSignal",
    "Side",
    "edge_abs",
    "edge_rel",
    "effective_price",
    "ev_per_dollar",
    "evaluate_binary_market",
    "fair_decimal_odds",
    "fair_price",
    "implied_prob",
    "remove_overround",
]
