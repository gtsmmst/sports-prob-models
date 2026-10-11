"""Arbitrage detection and delta-neutral stake splitting.

Two flavours:

1. **Same-venue, n-way** (:func:`arbitrage_from_odds`): decimal odds on every
   outcome of an event.  An arbitrage exists when ``sum(1/odds) < 1``; staking
   proportionally to ``1/odds`` equalises the payout across outcomes.

2. **Two-venue, fee-aware** (:func:`two_venue_arb`): outcome A backed at a
   bookmaker with decimal odds ``o`` (no explicit fee), and the complementary
   outcome bought on an exchange / prediction market as a binary contract at
   ``price q`` with fee ``fee`` on profit.  ``N`` contracts cost ``N q`` and net
   ``N g`` on a win where ``g = 1 - (1 - q) * fee``.  Equal payouts require

       X * o = N * g,     X + N q = bankroll
       =>  X = bankroll / (1 + o q / g),   N = X o / g

   and the position is an arbitrage iff ``1/o + q/g < 1``.  ``q / g`` is the
   fee-adjusted implied probability of the exchange leg.

:func:`hedge_stake` solves the related problem of locking in the profit of an
*existing* exchange position with a bookmaker bet on the opposite outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .edge import effective_price


def implied_sum(decimal_odds: Sequence[float]) -> float:
    """``sum(1 / odds)`` across outcomes (``< 1`` means arbitrage)."""
    if len(decimal_odds) < 2:
        raise ValueError("need odds for at least two outcomes")
    for o in decimal_odds:
        if not (o > 1.0):
            raise ValueError(f"decimal odds must exceed 1, got {o!r}")
    return sum(1.0 / o for o in decimal_odds)


@dataclass(frozen=True)
class ArbResult:
    """Outcome of :func:`arbitrage_from_odds`."""

    is_arb: bool
    implied_total: float
    """Sum of implied probabilities; edge is ``1 - implied_total``."""
    edge: float
    stakes: tuple[float, ...]
    """Stake on each outcome (sums to ``total_stake``)."""
    payout: float
    """Payout received whichever outcome wins."""
    profit: float
    roi: float


def arbitrage_from_odds(decimal_odds: Sequence[float], total_stake: float = 1.0, min_edge: float = 0.0) -> ArbResult:
    """Check an n-way set of decimal odds and split ``total_stake`` so every
    outcome returns the same payout."""
    if total_stake < 0:
        raise ValueError("total_stake must be non-negative")
    total = implied_sum(decimal_odds)
    edge = 1.0 - total
    stakes = tuple(total_stake * (1.0 / o) / total for o in decimal_odds)
    payout = total_stake / total
    profit = payout - total_stake
    roi = profit / total_stake if total_stake > 0 else edge / total
    return ArbResult(edge > max(min_edge, 0.0), total, edge, stakes, payout, profit, roi)


def exchange_net_multiplier(price: float, fee: float = 0.0) -> float:
    """``g = 1 - (1 - price) * fee``: net payout per contract after fee on profit."""
    if not (0.0 < price < 1.0):
        raise ValueError(f"price must lie in (0, 1), got {price!r}")
    if not (0.0 <= fee < 1.0):
        raise ValueError(f"fee must lie in [0, 1), got {fee!r}")
    return 1.0 - (1.0 - price) * fee


def exchange_implied(price: float, fee: float = 0.0) -> float:
    """Fee-adjusted implied probability of an exchange contract: ``price / g``.

    Identical to :func:`sports_prob_models.edge.effective_price`.
    """
    return effective_price(price, fee)


@dataclass(frozen=True)
class TwoVenueArb:
    """Outcome of :func:`two_venue_arb`."""

    is_arb: bool
    book_implied: float
    exchange_implied: float
    """Fee-adjusted implied probability of the exchange leg."""
    edge: float
    """``1 - (book_implied + exchange_implied)``."""
    stake_book: float
    """Currency staked at the bookmaker on outcome A."""
    cost_exchange: float
    """Currency spent on exchange contracts for the complementary outcome."""
    contracts_exchange: float
    """Number of contracts bought."""
    payout_if_book_wins: float
    payout_if_exchange_wins: float
    """Net of the exchange fee."""
    profit: float
    """Guaranteed profit (identical on both branches up to rounding)."""
    roi: float


def two_venue_arb(
    book_odds: float,
    exchange_price: float,
    *,
    exchange_fee: float = 0.0,
    bankroll: float = 1.0,
    min_edge: float = 0.0,
) -> TwoVenueArb:
    """Fee-aware two-venue arbitrage check with a delta-neutral stake split.

    Args:
        book_odds: decimal odds on outcome A at the bookmaker.
        exchange_price: price of the contract paying 1 if A does *not* happen.
        exchange_fee: fee rate on exchange profit.
        bankroll: total currency to deploy across both legs.
        min_edge: minimum edge (probability units) to flag as an arbitrage.
    """
    if not (book_odds > 1.0):
        raise ValueError(f"book_odds must exceed 1, got {book_odds!r}")
    if bankroll < 0:
        raise ValueError("bankroll must be non-negative")
    g = exchange_net_multiplier(exchange_price, exchange_fee)
    book_imp = 1.0 / book_odds
    exch_imp = exchange_price / g
    edge = 1.0 - (book_imp + exch_imp)

    stake_book = bankroll / (1.0 + book_odds * exchange_price / g)
    contracts = stake_book * book_odds / g
    cost_exchange = contracts * exchange_price
    payout_book = stake_book * book_odds
    payout_exchange = contracts * g
    profit = payout_book - bankroll
    roi = profit / bankroll if bankroll > 0 else edge / (1.0 - edge)
    return TwoVenueArb(
        is_arb=edge > max(min_edge, 0.0),
        book_implied=book_imp,
        exchange_implied=exch_imp,
        edge=edge,
        stake_book=stake_book,
        cost_exchange=cost_exchange,
        contracts_exchange=contracts,
        payout_if_book_wins=payout_book,
        payout_if_exchange_wins=payout_exchange,
        profit=profit,
        roi=roi,
    )


def delta_neutral_split(
    book_odds: float,
    exchange_price: float,
    *,
    exchange_fee: float = 0.0,
    bankroll: float = 1.0,
) -> tuple[float, float, float]:
    """``(stake_book, cost_exchange, contracts_exchange)`` that equalise payouts.

    Thin wrapper over :func:`two_venue_arb` for callers who only need sizes.
    """
    r = two_venue_arb(book_odds, exchange_price, exchange_fee=exchange_fee, bankroll=bankroll)
    return r.stake_book, r.cost_exchange, r.contracts_exchange


@dataclass(frozen=True)
class HedgeResult:
    """Outcome of :func:`hedge_stake`."""

    stake: float
    """Bookmaker stake on the opposite outcome."""
    locked_profit: float
    """Profit regardless of outcome (negative = guaranteed loss)."""
    payout_if_position_wins: float
    payout_if_hedge_wins: float


def hedge_stake(
    contracts: float,
    cost: float,
    hedge_odds: float,
    *,
    fee: float = 0.0,
) -> HedgeResult:
    """Lock in an existing exchange position with a bookmaker bet.

    You hold ``contracts`` contracts (each paying 1) bought for ``cost`` in
    total, with ``fee`` charged on profit if they win.  Staking ``X`` at decimal
    ``hedge_odds`` on the opposite outcome equalises the two branches when

        X * hedge_odds = contracts - (contracts - cost) * fee
    """
    if contracts <= 0 or cost < 0:
        raise ValueError("contracts must be positive and cost non-negative")
    if not (hedge_odds > 1.0):
        raise ValueError(f"hedge_odds must exceed 1, got {hedge_odds!r}")
    if not (0.0 <= fee < 1.0):
        raise ValueError(f"fee must lie in [0, 1), got {fee!r}")
    net_if_win = contracts - max(0.0, contracts - cost) * fee
    stake = net_if_win / hedge_odds
    locked = net_if_win - cost - stake
    return HedgeResult(stake, locked, net_if_win, stake * hedge_odds)


__all__ = [
    "ArbResult",
    "HedgeResult",
    "TwoVenueArb",
    "arbitrage_from_odds",
    "delta_neutral_split",
    "exchange_implied",
    "exchange_net_multiplier",
    "hedge_stake",
    "implied_sum",
    "two_venue_arb",
]
