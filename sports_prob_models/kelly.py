"""Kelly criterion sizing for binary markets.

A binary contract costs ``price`` (in ``(0, 1)``) and pays 1 if the event
happens.  Staking 1 unit buys ``1/price`` contracts, so the net odds are

    b = (1 - price) / price            (decimal odds d  ->  price = 1/d, b = d - 1)

With a fee of ``fee`` charged on *profit* (exchange / prediction-market style)
the net odds shrink to ``b * (1 - fee)``.  The full-Kelly fraction of bankroll is

    f* = (p * (b + 1) - 1) / b  =  p - (1 - p) / b

which is 0 when there is no edge (``p <= price`` for fee = 0).  Fractional
Kelly (e.g. ``fraction=0.25``) and a hard cap (``max_fraction``) are the usual
protection against model error.
"""

from __future__ import annotations

import math


def _validate(p: float, price: float, fee: float, fraction: float, max_fraction: float | None) -> None:
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must lie in [0, 1], got {p!r}")
    if not (0.0 < price < 1.0):
        raise ValueError(f"price must lie in (0, 1), got {price!r}")
    if not (0.0 <= fee < 1.0):
        raise ValueError(f"fee must lie in [0, 1), got {fee!r}")
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must lie in [0, 1], got {fraction!r}")
    if max_fraction is not None and not (0.0 <= max_fraction <= 1.0):
        raise ValueError(f"max_fraction must lie in [0, 1], got {max_fraction!r}")


def net_odds(price: float, fee: float = 0.0) -> float:
    """Net odds ``b`` per unit staked at ``price`` after a fee on profit."""
    if not (0.0 < price < 1.0):
        raise ValueError(f"price must lie in (0, 1), got {price!r}")
    return (1.0 - price) / price * (1.0 - fee)


def kelly_fraction(
    p: float,
    price: float,
    *,
    fee: float = 0.0,
    fraction: float = 1.0,
    max_fraction: float | None = None,
) -> float:
    """Fraction of bankroll to stake on a binary contract.

    Args:
        p: model probability the contract pays out.
        price: contract price in (0, 1).
        fee: fee rate charged on profit (0.02 = 2 %).
        fraction: Kelly multiplier (1.0 = full Kelly, 0.25 = quarter Kelly).
        max_fraction: hard cap on the returned fraction.

    Returns:
        ``max(0, f*) * fraction``, capped at ``max_fraction``.
    """
    _validate(p, price, fee, fraction, max_fraction)
    b = net_odds(price, fee)
    f = (p * (b + 1.0) - 1.0) / b
    f = max(0.0, f) * fraction
    if max_fraction is not None:
        f = min(f, max_fraction)
    return f


def kelly_fraction_from_odds(
    p: float,
    decimal_odds: float,
    *,
    fee: float = 0.0,
    fraction: float = 1.0,
    max_fraction: float | None = None,
) -> float:
    """Same as :func:`kelly_fraction` but for bookmaker decimal odds (``> 1``)."""
    if not (decimal_odds > 1.0):
        raise ValueError(f"decimal_odds must exceed 1, got {decimal_odds!r}")
    return kelly_fraction(p, 1.0 / decimal_odds, fee=fee, fraction=fraction, max_fraction=max_fraction)


def kelly_stake(
    bankroll: float,
    p: float,
    price: float,
    *,
    fee: float = 0.0,
    fraction: float = 1.0,
    max_fraction: float | None = None,
) -> float:
    """Stake in currency units: ``bankroll * kelly_fraction(...)``."""
    if bankroll < 0:
        raise ValueError("bankroll must be non-negative")
    return bankroll * kelly_fraction(p, price, fee=fee, fraction=fraction, max_fraction=max_fraction)


def log_growth_rate(p: float, price: float, f: float, fee: float = 0.0) -> float:
    """Expected log growth per bet when staking fraction ``f``:

    ``p * ln(1 + f b) + (1 - p) * ln(1 - f)``.  Maximised at ``f = f*``.
    """
    if not (0.0 <= f < 1.0):
        raise ValueError("f must lie in [0, 1)")
    b = net_odds(price, fee)
    return p * math.log1p(f * b) + (1.0 - p) * math.log1p(-f)


__all__ = [
    "kelly_fraction",
    "kelly_fraction_from_odds",
    "kelly_stake",
    "log_growth_rate",
    "net_odds",
]
