import math

import pytest

from sports_prob_models import cs2


# ---------------------------------------------------------------------------
# TeamStrength
# ---------------------------------------------------------------------------


def test_team_strength_validation():
    assert cs2.TeamStrength("A", 1.2).rating == 1.2
    with pytest.raises(ValueError):
        cs2.TeamStrength("A", 0.0)
    with pytest.raises(ValueError):
        cs2.TeamStrength("", 1.0)


def test_lookup_strength_case_insensitive_and_default():
    table = {"Alpha": 1.3, "Beta": 0.9}
    assert cs2.lookup_strength("alpha", table) == cs2.TeamStrength("Alpha", 1.3)
    assert cs2.lookup_strength("  BETA ", table).rating == 0.9
    assert cs2.lookup_strength("Gamma", table).rating == 1.0
    assert cs2.lookup_strength("Gamma", table, default=0.8).rating == 0.8


# ---------------------------------------------------------------------------
# Round level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sa,sb", [(1.0, 1.0), (1.4, 0.7), (0.6, 1.3), (5.0, 0.5)])
@pytest.mark.parametrize("side", ["CT", "T"])
def test_round_prob_in_unit_interval(sa, sb, side):
    p = cs2.round_win_prob(sa, sb, side)
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("sa,sb", [(1.0, 1.0), (1.4, 0.7), (0.6, 1.3), (5.0, 0.5)])
def test_round_prob_complementary(sa, sb):
    # A on CT  <=>  B on T
    assert cs2.round_win_prob(sa, sb, "CT") + cs2.round_win_prob(sb, sa, "T") == pytest.approx(1.0)
    assert cs2.round_win_prob(sa, sb, "T") + cs2.round_win_prob(sb, sa, "CT") == pytest.approx(1.0)


def test_round_prob_ct_bonus_and_equal_teams():
    p_ct = cs2.round_win_prob(1.0, 1.0, "CT")
    p_t = cs2.round_win_prob(1.0, 1.0, "T")
    assert p_ct == pytest.approx(1.15 / 2.15)
    assert p_ct > 0.5 > p_t
    assert cs2.round_win_prob(1.0, 1.0, "CT", ct_advantage=1.0) == pytest.approx(0.5)


def test_round_prob_monotone_in_strength():
    probs = [cs2.round_win_prob(s, 1.0, "T") for s in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0)]
    assert probs == sorted(probs)


def test_pistol_round_compresses_strength_gap():
    strong, weak = 1.5, 0.8
    assert abs(cs2.pistol_round_win_prob(strong, weak, "T") - 0.5) < abs(cs2.round_win_prob(strong, weak, "T") - 0.5)


def test_round_prob_rejects_bad_inputs():
    with pytest.raises(ValueError):
        cs2.round_win_prob(0.0, 1.0, "CT")
    with pytest.raises(ValueError):
        cs2.round_win_prob(1.0, 1.0, "X")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sides and map winner
# ---------------------------------------------------------------------------


def test_side_for_round_mr12_sequence():
    sides = [cs2.side_for_round(r, True) for r in range(30)]
    assert sides[:12] == ["CT"] * 12
    assert sides[12:24] == ["T"] * 12
    assert sides[24:27] == ["T"] * 3  # first OT half keeps second-half side
    assert sides[27:30] == ["CT"] * 3
    assert [cs2.side_for_round(r, False) for r in range(3)] == ["T"] * 3


def test_side_for_round_mr15():
    assert cs2.side_for_round(14, True, half_rounds=15) == "CT"
    assert cs2.side_for_round(15, True, half_rounds=15) == "T"


@pytest.mark.parametrize(
    "sa,sb,expected",
    [
        (0, 0, 0),
        (12, 12, 0),
        (13, 11, 1),
        (11, 13, -1),
        (13, 12, 0),  # overtime, not a win
        (16, 14, 1),
        (16, 13, 1),
        (14, 16, -1),
        (15, 15, 0),
        (16, 15, 0),
        (19, 17, 1),
        (18, 17, 0),
        (17, 19, -1),
    ],
)
def test_map_winner_mr12(sa, sb, expected):
    assert cs2.map_winner(sa, sb) == expected


def test_map_winner_mr15():
    assert cs2.map_winner(16, 14, half_rounds=15) == 1
    assert cs2.map_winner(16, 15, half_rounds=15) == 0
    assert cs2.map_winner(19, 17, half_rounds=15) == 1


# ---------------------------------------------------------------------------
# Map level: exact DP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score", [(0, 0), (5, 3), (12, 12), (14, 13), (3, 11)])
@pytest.mark.parametrize("start_ct", [True, False])
@pytest.mark.parametrize("sa,sb", [(1.0, 1.0), (1.3, 0.9), (0.8, 1.2)])
def test_map_exact_complementary(score, start_ct, sa, sb):
    a, b = score
    p_a = cs2.map_win_prob_exact(a, b, start_ct, sa, sb)
    p_b = cs2.map_win_prob_exact(b, a, not start_ct, sb, sa)
    assert 0.0 <= p_a <= 1.0
    assert p_a + p_b == pytest.approx(1.0, abs=1e-9)


def test_map_exact_neutral_equal_teams_is_half():
    assert cs2.neutral_map_win_prob(1.0, 1.0) == pytest.approx(0.5, abs=1e-9)


def test_map_exact_decided_and_dominant_states():
    assert cs2.map_win_prob_exact(13, 7, True, 1.0, 1.0) == 1.0
    assert cs2.map_win_prob_exact(7, 13, True, 1.0, 1.0) == 0.0
    assert cs2.map_win_prob_exact(12, 0, True, 1.0, 1.0) > 0.99
    assert cs2.map_win_prob_exact(0, 12, True, 1.0, 1.0) < 0.01


def test_map_exact_monotone_in_strength():
    probs = [cs2.map_win_prob_exact(6, 6, True, s, 1.0) for s in (0.6, 0.8, 1.0, 1.2, 1.5)]
    assert probs == sorted(probs)
    assert probs[0] < 0.5 < probs[-1]


def test_map_exact_lead_helps():
    base = cs2.map_win_prob_exact(6, 6, True, 1.0, 1.0)
    assert cs2.map_win_prob_exact(8, 6, True, 1.0, 1.0) > base > cs2.map_win_prob_exact(6, 8, True, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Map level: Monte Carlo
# ---------------------------------------------------------------------------


def test_map_mc_deterministic_with_seed():
    a = cs2.map_win_prob(8, 7, True, 1.2, 1.0, n_sims=500, seed=42)
    b = cs2.map_win_prob(8, 7, True, 1.2, 1.0, n_sims=500, seed=42)
    assert a == b
    assert 0.0 <= a <= 1.0


@pytest.mark.parametrize("score,start_ct,sa,sb", [((8, 7), True, 1.2, 1.0), ((0, 0), False, 0.9, 1.1), ((12, 12), True, 1.0, 1.0)])
def test_map_mc_agrees_with_exact(score, start_ct, sa, sb):
    exact = cs2.map_win_prob_exact(score[0], score[1], start_ct, sa, sb)
    mc = cs2.map_win_prob(score[0], score[1], start_ct, sa, sb, n_sims=4000, seed=7)
    # std error ~ 0.008 at n=4000; 0.03 is ~4 sigma
    assert abs(mc - exact) < 0.03


def test_map_mc_monotone_in_strength():
    probs = [cs2.map_win_prob(6, 6, True, s, 1.0, n_sims=3000, seed=3) for s in (0.7, 1.0, 1.4)]
    assert probs[0] < probs[1] < probs[2]


def test_map_mc_decided_state_short_circuits():
    assert cs2.map_win_prob(13, 5, True, 1.0, 1.0, n_sims=1, seed=0) == 1.0
    assert cs2.map_win_prob(5, 13, True, 1.0, 1.0, n_sims=1, seed=0) == 0.0


# ---------------------------------------------------------------------------
# Series level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.3, 0.5, 0.6, 0.75])
def test_series_closed_forms(p):
    q = 1.0 - p
    assert cs2.series_win_prob(0, 0, p, best_of=1) == pytest.approx(p)
    assert cs2.series_win_prob(0, 0, p, best_of=3) == pytest.approx(p**2 * (3 - 2 * p))
    assert cs2.series_win_prob(0, 0, p, best_of=5) == pytest.approx(p**3 * (1 + 3 * q + 6 * q**2))


def test_series_bo5_amplifies_favourite():
    p = 0.6
    bo1 = cs2.series_win_prob(0, 0, p, best_of=1)
    bo3 = cs2.series_win_prob(0, 0, p, best_of=3)
    bo5 = cs2.series_win_prob(0, 0, p, best_of=5)
    assert bo1 < bo3 < bo5
    # ... and punishes the underdog
    u1, u3, u5 = (cs2.series_win_prob(0, 0, 1 - p, best_of=n) for n in (1, 3, 5))
    assert u1 > u3 > u5


def test_series_monotone_in_both_probabilities():
    grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    for q in grid:
        vals = [cs2.series_win_prob(0, 1, q, p, best_of=3) for p in grid]
        assert vals == sorted(vals)
    for p in grid:
        vals = [cs2.series_win_prob(1, 0, q, p, best_of=5) for q in grid]
        assert vals == sorted(vals)


@pytest.mark.parametrize("a,b,bo", [(0, 0, 3), (1, 0, 3), (0, 1, 3), (1, 1, 3), (2, 1, 5), (0, 2, 5)])
def test_series_complementary(a, b, bo):
    p_cur, p_fut = 0.62, 0.55
    assert cs2.series_win_prob(a, b, p_fut, p_cur, bo) + cs2.series_win_prob(b, a, 1 - p_fut, 1 - p_cur, bo) == pytest.approx(1.0)


def test_series_decided_and_match_point():
    assert cs2.series_win_prob(2, 0, 0.1, best_of=3) == 1.0
    assert cs2.series_win_prob(0, 2, 0.9, best_of=3) == 0.0
    # 1-1 in a BO3: series == current map
    assert cs2.series_win_prob(1, 1, 0.4, 0.7, best_of=3) == pytest.approx(0.7)


def test_series_rejects_bad_inputs():
    with pytest.raises(ValueError):
        cs2.series_win_prob(0, 0, 0.5, best_of=4)
    with pytest.raises(ValueError):
        cs2.series_win_prob(0, 0, 1.5)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_match_win_prob_monotone_in_strength():
    probs = [cs2.match_win_prob(s, 1.0, best_of=3) for s in (0.7, 0.9, 1.0, 1.1, 1.4)]
    assert probs == sorted(probs)
    assert probs[2] == pytest.approx(0.5, abs=1e-9)


def test_match_win_prob_live_map_and_methods_agree():
    exact = cs2.match_win_prob(1.2, 1.0, maps_won_a=1, map_in_progress=True, score_a=8, score_b=7, method="exact")
    mc = cs2.match_win_prob(1.2, 1.0, maps_won_a=1, map_in_progress=True, score_a=8, score_b=7, method="mc", n_sims=4000, seed=1)
    assert 0.0 <= exact <= 1.0
    assert abs(exact - mc) < 0.03
    assert exact > cs2.match_win_prob(1.2, 1.0, best_of=3)  # leading 1-0 and 8-7 helps
    assert not math.isnan(exact)
