import pytest

from sports_prob_models import kelly


def test_closed_form_even_money():
    # p = 0.6 at price 0.5 (b = 1): f* = 2p - 1 = 0.2
    assert kelly.kelly_fraction(0.6, 0.5) == pytest.approx(0.2)
    assert kelly.kelly_fraction_from_odds(0.6, 2.0) == pytest.approx(0.2)


@pytest.mark.parametrize("p,price", [(0.7, 0.4), (0.55, 0.5), (0.9, 0.8), (0.3, 0.2)])
def test_matches_textbook_formula(p, price):
    b = (1 - price) / price
    expected = (p * (b + 1) - 1) / b
    assert kelly.kelly_fraction(p, price) == pytest.approx(expected)
    # equivalent form: (p - price) / (1 - price)
    assert kelly.kelly_fraction(p, price) == pytest.approx((p - price) / (1 - price))


@pytest.mark.parametrize("p,price", [(0.5, 0.5), (0.4, 0.5), (0.1, 0.9), (0.0, 0.3)])
def test_zero_when_no_edge(p, price):
    assert kelly.kelly_fraction(p, price) == 0.0
    assert kelly.kelly_stake(1000, p, price) == 0.0


def test_fractional_and_cap():
    assert kelly.kelly_fraction(0.6, 0.5, fraction=0.25) == pytest.approx(0.05)
    assert kelly.kelly_fraction(0.6, 0.5, max_fraction=0.1) == pytest.approx(0.1)
    assert kelly.kelly_fraction(0.6, 0.5, fraction=0.25, max_fraction=0.1) == pytest.approx(0.05)
    assert kelly.kelly_fraction(0.95, 0.2, max_fraction=0.3) == pytest.approx(0.3)


def test_fee_reduces_stake_monotonically():
    fees = [0.0, 0.01, 0.02, 0.05, 0.1]
    fs = [kelly.kelly_fraction(0.6, 0.5, fee=f) for f in fees]
    assert fs == sorted(fs, reverse=True)
    assert fs[0] > fs[-1]
    # a large enough fee wipes out a thin edge entirely
    assert kelly.kelly_fraction(0.51, 0.5, fee=0.1) == 0.0


def test_fee_closed_form():
    # b' = b (1 - fee); f = p - (1 - p) / b'
    p, price, fee = 0.6, 0.5, 0.02
    b = (1 - price) / price * (1 - fee)
    assert kelly.kelly_fraction(p, price, fee=fee) == pytest.approx(p - (1 - p) / b)
    assert kelly.net_odds(price, fee) == pytest.approx(b)


def test_stake_scales_with_bankroll():
    assert kelly.kelly_stake(10_000, 0.6, 0.5, fraction=0.25) == pytest.approx(500.0)
    assert kelly.kelly_stake(0, 0.6, 0.5) == 0.0


def test_growth_rate_is_maximised_at_kelly_fraction():
    p, price = 0.6, 0.5
    f_star = kelly.kelly_fraction(p, price)
    g_star = kelly.log_growth_rate(p, price, f_star)
    for df in (-0.1, -0.05, -0.01, 0.01, 0.05, 0.1):
        assert kelly.log_growth_rate(p, price, f_star + df) < g_star
    assert kelly.log_growth_rate(p, price, 0.0) == 0.0


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        kelly.kelly_fraction(1.2, 0.5)
    with pytest.raises(ValueError):
        kelly.kelly_fraction(0.6, 1.0)
    with pytest.raises(ValueError):
        kelly.kelly_fraction(0.6, 0.5, fee=1.0)
    with pytest.raises(ValueError):
        kelly.kelly_fraction(0.6, 0.5, fraction=2.0)
    with pytest.raises(ValueError):
        kelly.kelly_fraction_from_odds(0.6, 1.0)
    with pytest.raises(ValueError):
        kelly.kelly_stake(-1, 0.6, 0.5)
