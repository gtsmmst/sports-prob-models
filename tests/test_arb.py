import random

import pytest

from sports_prob_models import arb, edge


# ---------------------------------------------------------------------------
# Same-venue n-way
# ---------------------------------------------------------------------------


def test_two_way_arb_is_flagged_with_correct_split():
    r = arb.arbitrage_from_odds([2.1, 2.1], total_stake=100)
    assert r.is_arb
    assert r.implied_total == pytest.approx(2 / 2.1)
    assert r.edge == pytest.approx(1 - 2 / 2.1)
    assert r.stakes == pytest.approx((50.0, 50.0))
    assert sum(r.stakes) == pytest.approx(100.0)
    assert r.payout == pytest.approx(105.0)
    assert r.profit == pytest.approx(5.0)
    assert r.roi == pytest.approx(0.05)
    # every outcome returns the same payout
    for stake, odds in zip(r.stakes, [2.1, 2.1]):
        assert stake * odds == pytest.approx(r.payout)


def test_non_arb_is_not_flagged():
    r = arb.arbitrage_from_odds([1.9, 1.9], total_stake=100)
    assert not r.is_arb
    assert r.implied_total > 1.0
    assert r.profit < 0.0


def test_three_way_arb_equalises_payouts():
    odds = [3.5, 3.6, 3.4]
    r = arb.arbitrage_from_odds(odds, total_stake=1000)
    assert r.is_arb
    assert sum(r.stakes) == pytest.approx(1000.0)
    payouts = [s * o for s, o in zip(r.stakes, odds)]
    assert max(payouts) - min(payouts) < 1e-9
    assert payouts[0] == pytest.approx(r.payout)
    assert r.profit > 0


def test_min_edge_threshold():
    assert arb.arbitrage_from_odds([2.05, 2.05], min_edge=0.05).is_arb is False
    assert arb.arbitrage_from_odds([2.05, 2.05], min_edge=0.01).is_arb is True


def test_rejects_bad_odds():
    with pytest.raises(ValueError):
        arb.arbitrage_from_odds([2.0])
    with pytest.raises(ValueError):
        arb.arbitrage_from_odds([2.0, 1.0])


# ---------------------------------------------------------------------------
# Two-venue, fee-aware
# ---------------------------------------------------------------------------


def test_exchange_implied_matches_edge_module():
    assert arb.exchange_implied(0.52, 0.02) == pytest.approx(edge.effective_price(0.52, 0.02))
    assert arb.exchange_implied(0.52, 0.0) == 0.52
    assert arb.exchange_net_multiplier(0.52, 0.02) == pytest.approx(1 - 0.48 * 0.02)


def test_two_venue_arb_flagged_and_delta_neutral():
    r = arb.two_venue_arb(2.3, 0.52, exchange_fee=0.02, bankroll=1000)
    assert r.is_arb
    assert r.book_implied == pytest.approx(1 / 2.3)
    assert r.exchange_implied == pytest.approx(0.52 / (1 - 0.48 * 0.02))
    assert r.edge == pytest.approx(1 - r.book_implied - r.exchange_implied)
    assert r.stake_book + r.cost_exchange == pytest.approx(1000.0)
    assert r.payout_if_book_wins == pytest.approx(r.payout_if_exchange_wins)
    assert r.profit == pytest.approx(r.payout_if_book_wins - 1000.0)
    assert r.profit > 0
    assert r.roi == pytest.approx(r.profit / 1000.0)
    # the exchange leg really nets what we claim after the fee
    n, q, fee = r.contracts_exchange, 0.52, 0.02
    assert n - (n - n * q) * fee == pytest.approx(r.payout_if_exchange_wins)
    assert n * q == pytest.approx(r.cost_exchange)


def test_two_venue_not_arb_when_implied_exceeds_one():
    r = arb.two_venue_arb(1.8, 0.5, exchange_fee=0.02, bankroll=1000)
    assert not r.is_arb
    assert r.edge < 0
    assert r.profit < 0
    # still delta neutral (a guaranteed loss)
    assert r.payout_if_book_wins == pytest.approx(r.payout_if_exchange_wins)


def test_fee_shrinks_edge_and_can_kill_a_thin_arb():
    no_fee = arb.two_venue_arb(2.1, 0.5, exchange_fee=0.0)
    with_fee = arb.two_venue_arb(2.1, 0.5, exchange_fee=0.02)
    assert with_fee.edge < no_fee.edge
    assert with_fee.profit < no_fee.profit
    # 1/2.05 + 0.51 = 0.9978 -> thin arb without fee, gone with a 2 % fee
    assert arb.two_venue_arb(2.05, 0.51).is_arb
    assert not arb.two_venue_arb(2.05, 0.51, exchange_fee=0.02).is_arb


def test_delta_neutral_split_property():
    rng = random.Random(123)
    for _ in range(200):
        odds = rng.uniform(1.2, 6.0)
        price = rng.uniform(0.05, 0.95)
        fee = rng.choice([0.0, 0.01, 0.02, 0.05])
        bankroll = rng.uniform(10, 10_000)
        stake, cost, n = arb.delta_neutral_split(odds, price, exchange_fee=fee, bankroll=bankroll)
        assert stake >= 0 and cost >= 0 and n >= 0
        assert stake + cost == pytest.approx(bankroll)
        payout_book = stake * odds
        payout_exch = n - (n - n * price) * fee
        assert payout_book == pytest.approx(payout_exch, rel=1e-9)
        r = arb.two_venue_arb(odds, price, exchange_fee=fee, bankroll=bankroll)
        assert r.is_arb == (1 / odds + arb.exchange_implied(price, fee) < 1.0)


def test_two_venue_rejects_bad_inputs():
    with pytest.raises(ValueError):
        arb.two_venue_arb(1.0, 0.5)
    with pytest.raises(ValueError):
        arb.two_venue_arb(2.0, 1.0)
    with pytest.raises(ValueError):
        arb.two_venue_arb(2.0, 0.5, exchange_fee=1.0)


# ---------------------------------------------------------------------------
# Hedging an existing position
# ---------------------------------------------------------------------------


def test_hedge_stake_locks_equal_profit():
    h = arb.hedge_stake(contracts=100, cost=40, hedge_odds=2.5)
    assert h.stake == pytest.approx(40.0)
    assert h.locked_profit == pytest.approx(20.0)
    assert h.payout_if_position_wins == pytest.approx(h.payout_if_hedge_wins)
    # branch check: position wins -> 100 - 40 - 40; hedge wins -> 40*2.5 - 40 - 40
    assert 100 - 40 - h.stake == pytest.approx(h.locked_profit)
    assert h.stake * 2.5 - 40 - h.stake == pytest.approx(h.locked_profit)


def test_hedge_stake_with_fee_and_losing_hedge():
    h = arb.hedge_stake(contracts=100, cost=40, hedge_odds=2.5, fee=0.02)
    assert h.payout_if_position_wins == pytest.approx(100 - 60 * 0.02)
    assert h.payout_if_position_wins == pytest.approx(h.payout_if_hedge_wins)
    assert h.locked_profit < 20.0
    bad = arb.hedge_stake(contracts=100, cost=55, hedge_odds=2.0)
    assert bad.locked_profit < 0
