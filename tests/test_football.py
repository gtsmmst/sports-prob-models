import math

import pytest

from sports_prob_models import football as fb


# ---------------------------------------------------------------------------
# Poisson primitives
# ---------------------------------------------------------------------------


def test_poisson_pmf_known_values():
    assert fb.poisson_pmf(0, 1.0) == pytest.approx(math.exp(-1))
    assert fb.poisson_pmf(2, 2.0) == pytest.approx(2 * math.exp(-2))
    assert fb.poisson_pmf(0, 0.0) == 1.0
    assert fb.poisson_pmf(3, 0.0) == 0.0
    assert fb.poisson_pmf(-1, 1.0) == 0.0


def test_poisson_pmf_sums_to_one():
    assert sum(fb.poisson_pmf(k, 2.5) for k in range(60)) == pytest.approx(1.0)


def test_poisson_cdf_and_tail():
    assert fb.poisson_cdf(1, 1.0) == pytest.approx(2 * math.exp(-1))
    assert fb.poisson_at_least(1, 1.0) == pytest.approx(1 - math.exp(-1))
    assert fb.poisson_at_least(0, 3.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scoreline matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lh,la", [(1.0, 1.0), (1.6, 1.1), (2.8, 0.7), (0.3, 0.4), (3.5, 3.0)])
def test_matrix_sums_to_one_and_is_a_distribution(lh, la):
    m = fb.scoreline_matrix(lh, la)
    assert fb.matrix_total(m) == pytest.approx(1.0, abs=2e-3)
    assert all(0.0 <= p <= 1.0 for row in m for p in row)


def test_matrix_known_cells_lambda_one():
    m = fb.scoreline_matrix(1.0, 1.0)
    assert m[0][0] == pytest.approx(math.exp(-2))
    assert m[1][0] == pytest.approx(math.exp(-2))
    assert m[2][1] == pytest.approx(math.exp(-2) / 2)


# ---------------------------------------------------------------------------
# Markets: hand-computed values for lambda = 1, 1
# ---------------------------------------------------------------------------


@pytest.fixture
def m11():
    return fb.scoreline_matrix(1.0, 1.0)


def test_1x2_symmetric_and_hand_computed(m11):
    r = fb.match_result_probs(m11)
    assert r.home + r.draw + r.away == pytest.approx(1.0)
    assert r.home == pytest.approx(r.away)
    # draw = sum_k pmf(k,1)^2 = e^-2 * sum 1/(k!)^2 = e^-2 * I_0(2)
    draw = math.exp(-2) * sum(1 / math.factorial(k) ** 2 for k in range(15))
    assert r.draw == pytest.approx(draw, abs=1e-6)
    assert r.home == pytest.approx((1 - draw) / 2, abs=1e-6)


def test_btts_hand_computed_and_consistent_with_matrix(m11):
    assert fb.btts_prob(m11) == pytest.approx((1 - math.exp(-1)) ** 2, abs=1e-6)
    m = fb.scoreline_matrix(1.7, 1.2)
    direct = sum(m[i][j] for i in range(1, len(m)) for j in range(1, len(m)))
    assert fb.btts_prob(m) == pytest.approx(direct, abs=1e-6)


def test_over_2_5_hand_computed_and_consistent(m11):
    # total ~ Poisson(2): P(>2) = 1 - e^-2 (1 + 2 + 2)
    assert fb.over_prob(m11, 2.5) == pytest.approx(1 - 5 * math.exp(-2), abs=1e-6)
    m = fb.scoreline_matrix(1.7, 1.2)
    direct = sum(m[i][j] for i in range(len(m)) for j in range(len(m)) if i + j > 2)
    assert fb.over_prob(m, 2.5) == pytest.approx(direct, abs=1e-6)


def test_over_under_three_way_consistency(m11):
    for line in (0.5, 1.5, 2.5, 2.0, 3.0, 2.25, 2.75):
        t = fb.over_under_probs(m11, line)
        assert t.win + t.push + t.lose == pytest.approx(1.0, abs=1e-6)
        assert all(0.0 <= x <= 1.0 for x in t.as_tuple())
    assert fb.over_under_probs(m11, 2.5).push == 0.0
    dist = fb.total_goals_distribution(m11)
    assert fb.over_under_probs(m11, 3).push == pytest.approx(dist[3])
    # quarter line = average of the neighbours
    lo, hi, q = (fb.over_under_probs(m11, L) for L in (2.5, 3.0, 2.75))
    assert q.win == pytest.approx((lo.win + hi.win) / 2)
    assert q.push == pytest.approx((lo.push + hi.push) / 2)


def test_over_under_rejects_non_quarter_lines(m11):
    with pytest.raises(ValueError):
        fb.over_under_probs(m11, 2.3)


def test_asian_handicap_consistency(m11):
    r = fb.match_result_probs(m11)
    ah0 = fb.asian_handicap_probs(m11, 0.0)
    assert ah0.win == pytest.approx(r.home)
    assert ah0.push == pytest.approx(r.draw)
    assert ah0.lose == pytest.approx(r.away)
    ah_half = fb.asian_handicap_probs(m11, -0.5)
    assert ah_half.win == pytest.approx(r.home)
    assert ah_half.push == 0.0
    plus_half = fb.asian_handicap_probs(m11, +0.5)
    assert plus_half.win == pytest.approx(r.home + r.draw)
    quarter = fb.asian_handicap_probs(m11, -0.25)
    assert quarter.win == pytest.approx((ah0.win + ah_half.win) / 2)
    assert quarter.push == pytest.approx(ah0.push / 2)
    for h in (-2.0, -1.75, -1.0, -0.75, 0.25, 1.5):
        t = fb.asian_handicap_probs(m11, h)
        assert t.win + t.push + t.lose == pytest.approx(1.0, abs=1e-6)


def test_asian_handicap_monotone_in_line():
    m = fb.scoreline_matrix(1.5, 1.2)
    wins = [fb.asian_handicap_probs(m, h).win for h in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0)]
    assert wins == sorted(wins)


# ---------------------------------------------------------------------------
# Current-score offsets
# ---------------------------------------------------------------------------


def test_markets_respect_goals_already_scored():
    m = fb.scoreline_matrix(0.8, 0.6)
    assert fb.btts_prob(m, 1, 1) == 1.0
    assert fb.btts_prob(m, 1, 0) == pytest.approx(1 - math.exp(-0.6), abs=1e-6)
    assert fb.btts_prob(m, 0, 2) == pytest.approx(1 - math.exp(-0.8), abs=1e-6)
    assert fb.over_prob(m, 2.5, 2, 1) == 1.0
    assert fb.over_prob(m, 2.5, 1, 1) == pytest.approx(1 - math.exp(-1.4), abs=1e-6)
    r = fb.match_result_probs(m, 3, 0)
    # away needs 3+ more goals than home from lambda 0.6 vs 0.8: P(home) = 1 - P(draw) - P(away)
    assert r.home > 0.98
    assert r.home == pytest.approx(1 - r.draw - r.away)
    # handicaps apply to the final score: -0.5 from 3-0 is "home wins outright";
    # -2.5 from 3-0 is "home does not lose the rest of the match"
    assert fb.asian_handicap_probs(m, -0.5, 3, 0).win == pytest.approx(r.home)
    rest = fb.match_result_probs(m)
    assert fb.asian_handicap_probs(m, -2.5, 3, 0).win == pytest.approx(rest.home + rest.draw)


# ---------------------------------------------------------------------------
# Pre-match BTTS
# ---------------------------------------------------------------------------


def test_prematch_btts_and_blend():
    p = fb.prematch_btts_prob(1.0, 1.0)
    assert p == pytest.approx((1 - math.exp(-1)) ** 2)
    assert fb.prematch_btts_prob(1.0, 1.0, league_rate=0.6, blend=0.5) == pytest.approx(0.5 * p + 0.3)
    assert fb.prematch_btts_prob(1.0, 1.0, league_rate=0.6, blend=1.0) == pytest.approx(p)
    with pytest.raises(ValueError):
        fb.prematch_btts_prob(1.0, 1.0, league_rate=1.5)


# ---------------------------------------------------------------------------
# Live state
# ---------------------------------------------------------------------------


def test_live_rates_before_kickoff_equal_prematch():
    st = fb.LiveMatchState(minute=0, avg_goals_home=1.8, avg_goals_away=1.1)
    assert fb.live_goal_rates(st) == (1.8, 1.1)


def test_live_rates_zero_at_full_time():
    st = fb.LiveMatchState(minute=90, xg_home=2.0, xg_away=1.0, shots_on_target_home=9)
    assert fb.live_goal_rates(st) == (0.0, 0.0)


def test_live_rates_track_xg_pace_and_time():
    slow = fb.LiveMatchState(minute=45, xg_home=0.2, xg_away=0.2, avg_goals_home=1.4, avg_goals_away=1.4)
    fast = fb.LiveMatchState(minute=45, xg_home=1.5, xg_away=0.2, avg_goals_home=1.4, avg_goals_away=1.4)
    assert fb.live_goal_rates(fast)[0] > fb.live_goal_rates(slow)[0]
    assert fb.live_goal_rates(fast)[1] == pytest.approx(fb.live_goal_rates(slow)[1])
    late = fb.LiveMatchState(minute=80, xg_home=1.5, xg_away=0.2, avg_goals_home=1.4, avg_goals_away=1.4)
    assert fb.live_goal_rates(late)[0] < fb.live_goal_rates(fast)[0]


def test_live_probs_are_probabilities_and_consistent():
    st = fb.LiveMatchState(minute=50, home_goals=0, away_goals=1, xg_home=1.2, xg_away=0.6, shots_on_target_home=5)
    lp = fb.live_probs(st)
    assert lp.home + lp.draw + lp.away == pytest.approx(1.0)
    for x in (lp.home, lp.draw, lp.away, lp.btts, lp.over_2_5):
        assert 0.0 <= x <= 1.0
    # away leads 0-1: BTTS requires home to score at least once
    assert lp.btts == pytest.approx(1 - math.exp(-lp.lambda_home), abs=1e-6)
    done = fb.live_probs(fb.LiveMatchState(minute=90, home_goals=2, away_goals=1))
    assert (done.home, done.draw, done.away, done.btts, done.over_2_5) == (1.0, 0.0, 0.0, 1.0, 1.0)
