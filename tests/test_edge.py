import pytest

from sports_prob_models import edge, kelly


def test_implied_and_fair_conversions():
    assert edge.implied_prob(2.0) == pytest.approx(0.5)
    assert edge.implied_prob(4.0) == pytest.approx(0.25)
    assert edge.fair_price(0.37) == 0.37
    assert edge.fair_decimal_odds(0.25) == pytest.approx(4.0)
    assert edge.implied_prob(edge.fair_decimal_odds(0.61)) == pytest.approx(0.61)
    with pytest.raises(ValueError):
        edge.implied_prob(1.0)
    with pytest.raises(ValueError):
        edge.fair_decimal_odds(0.0)


def test_remove_overround():
    fair = edge.remove_overround([0.55, 0.50])
    assert sum(fair) == pytest.approx(1.0)
    assert fair[0] / fair[1] == pytest.approx(0.55 / 0.50)
    three = edge.remove_overround([1 / 2.1, 1 / 3.4, 1 / 3.9])
    assert sum(three) == pytest.approx(1.0)


def test_effective_price_with_fee():
    assert edge.effective_price(0.52, 0.0) == 0.52
    assert edge.effective_price(0.52, 0.02) == pytest.approx(0.52 / (1 - 0.48 * 0.02))
    assert edge.effective_price(0.52, 0.02) > 0.52
    # buying 1 / g contracts at price q nets exactly 1 after the fee
    q, fee = 0.3, 0.05
    g = 1 - (1 - q) * fee
    n = 1 / g
    assert n - (n - n * q) * fee == pytest.approx(1.0)
    assert n * q == pytest.approx(edge.effective_price(q, fee))


def test_ev_per_dollar_matches_relative_edge():
    assert edge.ev_per_dollar(0.6, 0.5) == pytest.approx(0.2)
    assert edge.ev_per_dollar(0.5, 0.5) == pytest.approx(0.0)
    assert edge.ev_per_dollar(0.4, 0.5) == pytest.approx(-0.2)
    for p, q, fee in [(0.6, 0.5, 0.02), (0.3, 0.25, 0.05), (0.8, 0.7, 0.0)]:
        assert edge.ev_per_dollar(p, q, fee) == pytest.approx(edge.edge_rel(p, q, fee))
    assert edge.ev_per_dollar(0.6, 0.5, fee=0.02) < edge.ev_per_dollar(0.6, 0.5)


def test_edge_abs_sign():
    assert edge.edge_abs(0.6, 0.5) == pytest.approx(0.1)
    assert edge.edge_abs(0.5, 0.5) == pytest.approx(0.0)
    assert edge.edge_abs(0.5, 0.5, fee=0.02) < 0.0


def test_evaluate_picks_yes_side():
    sig = edge.evaluate_binary_market(0.6, 0.5)
    assert sig.side == "YES"
    assert sig.model_prob == pytest.approx(0.6)
    assert sig.price == pytest.approx(0.5)
    assert sig.edge == pytest.approx(0.1)
    assert sig.ev_per_dollar == pytest.approx(0.2)
    assert sig.kelly_full == pytest.approx(kelly.kelly_fraction(0.6, 0.5))


def test_evaluate_picks_no_side():
    sig = edge.evaluate_binary_market(0.4, 0.5)
    assert sig.side == "NO"
    assert sig.model_prob == pytest.approx(0.6)
    assert sig.price == pytest.approx(0.5)
    assert sig.edge == pytest.approx(0.1)


def test_evaluate_passes_without_edge_or_below_threshold():
    assert edge.evaluate_binary_market(0.5, 0.5).side == "PASS"
    assert edge.evaluate_binary_market(0.5, 0.5).kelly_full == 0.0
    assert edge.evaluate_binary_market(0.52, 0.5, min_edge=0.05).side == "PASS"
    assert edge.evaluate_binary_market(0.56, 0.5, min_edge=0.05).side == "YES"


def test_evaluate_uses_explicit_no_price_and_fee():
    # YES at 0.55, NO at 0.50 (book has a spread): model 0.5 -> NO is the only +EV side
    sig = edge.evaluate_binary_market(0.5, 0.55, no_price=0.50, fee=0.0)
    assert sig.side == "PASS"
    sig = edge.evaluate_binary_market(0.5, 0.55, no_price=0.45)
    assert sig.side == "NO"
    assert sig.edge == pytest.approx(0.05)
    fee_sig = edge.evaluate_binary_market(0.5, 0.55, no_price=0.45, fee=0.02)
    assert fee_sig.edge < sig.edge


def test_evaluate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        edge.evaluate_binary_market(1.2, 0.5)
    with pytest.raises(ValueError):
        edge.evaluate_binary_market(0.5, 1.0, no_price=0.0)
