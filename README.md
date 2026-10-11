# sports-prob-models

[![tests](https://github.com/gtsmmst/sports-prob-models/actions/workflows/tests.yml/badge.svg)](https://github.com/gtsmmst/sports-prob-models/actions/workflows/tests.yml)

Small, tested, dependency-free Python package of **sports probability models and
bet-sizing math**. It is the maths layer extracted from a personal research
stack that compared model probabilities with prediction-market (Polymarket)
and bookmaker prices. Everything here is pure functions over numbers: no network
code, no exchange or bookmaker integrations, no order placement.

```
pip install .            # or:  pip install -e ".[dev]"  for pytest
```

Requires Python 3.10+. No runtime dependencies.

## Modules

| Module | What it does |
|---|---|
| `sports_prob_models.cs2` | Counter-Strike 2 in-play model: round -> map -> best-of-N series |
| `sports_prob_models.football` | Poisson / xG scoreline matrix -> 1X2, BTTS, totals, Asian handicap; live state |
| `sports_prob_models.kelly` | Full / fractional Kelly for binary contracts, with fee and cap |
| `sports_prob_models.edge` | Model probability vs market price: fair price, edge, EV per unit, side selection |
| `sports_prob_models.arb` | n-way and two-venue arbitrage checks, fee-aware delta-neutral stake split, hedging |
| `sports_prob_models.matching` | Team-name normalisation, fixture signatures, fuzzy matching, market-title parsing |

### `cs2` - round, map, series

* **Round**: `p = s_a' / (s_a' + s_b')`, where the CT-side team's strength is
  multiplied by `ct_advantage` (default 1.15, about 53.5 % for equal teams).
  Pistol rounds shrink the strength gap (`1 + (s - 1) * 0.7`) and use a smaller
  CT bonus.
* **Map**: `map_win_prob` runs a seeded Monte-Carlo over the remaining rounds;
  `map_win_prob_exact` solves the same process by dynamic programming over the
  `(score_a, score_b)` state. Both model MR12 / MR15 halves, the half-time side
  swap and MR3 overtime periods. The two agree to Monte-Carlo noise (tested).
* **Series**: `series_win_prob` is a DP over maps won. For a fresh series with
  per-map probability `p`: BO3 = `p^2 (3 - 2p)`, BO5 = `p^3 (1 + 3q + 6q^2)`.

```python
from sports_prob_models import cs2

# Team A rated 1.2 vs team B rated 1.0, BO3, A leads 1-0 and is up 8-7 on a map it started CT
p = cs2.match_win_prob(1.2, 1.0, best_of=3, maps_won_a=1, map_in_progress=True,
                       score_a=8, score_b=7, a_starts_ct=True)
print(round(p, 3))                                            # ~0.899

cs2.map_win_prob(8, 7, True, 1.2, 1.0, n_sims=3000, seed=42)  # Monte-Carlo
cs2.map_win_prob_exact(8, 7, True, 1.2, 1.0)                  # exact DP, ~0.676
cs2.series_win_prob(0, 0, p_future_map=0.6, best_of=5)        # 0.683
```

Ratings are a free scale with 1.0 = median team; supply your own table and
use `lookup_strength(name, table)`. No ratings ship with the package.

### `football` - Poisson / xG

Goals for each side are independent Poisson variables. The scoreline matrix
`P[i][j] = Pois(i; lambda_home) * Pois(j; lambda_away)` feeds every market:

* `match_result_probs` - 1X2 by comparing `home_goals + i` with `away_goals + j`
* `btts_prob` - `1 - P(home blank) - P(away blank) + P(0-0)`
* `over_under_probs(matrix, line)` - (over, push, under) for half, integer and
  quarter lines (quarter lines split across the two neighbours)
* `asian_handicap_probs(matrix, handicap)` - (win, push, lose) for the home side
* `LiveMatchState` + `live_goal_rates` - expected goals still to come, blending
  the observed xG pace (70 %) with the pre-match rate (30 %), scaled by time left

```python
from sports_prob_models import football as fb

m = fb.scoreline_matrix(1.6, 1.1)
fb.match_result_probs(m)             # MatchResultProbs(home=0.49, draw=0.25, away=0.26)
fb.btts_prob(m)                      # 0.53
fb.over_under_probs(m, 2.5).win      # P(over 2.5) = 0.51
fb.asian_handicap_probs(m, -0.75)    # ThreeWay(win=..., push=..., lose=...)

state = fb.LiveMatchState(minute=60, home_goals=1, away_goals=0, xg_home=1.4, xg_away=0.5,
                          avg_goals_home=1.6, avg_goals_away=1.1)
fb.live_probs(state)                 # 1X2, BTTS, over 2.5 and remaining lambdas
```

### `kelly` - sizing

For a binary contract at `price` (pays 1), net odds are `b = (1 - price) / price`
(a fee rate on profit scales this by `1 - fee`). Full Kelly:

```
f* = (p (b + 1) - 1) / b = p - (1 - p) / b
```

```python
from sports_prob_models import kelly

kelly.kelly_fraction(0.6, 0.5)                                  # 0.20
kelly.kelly_fraction(0.6, 0.5, fee=0.02)                        # 0.19  (fee on profit)
kelly.kelly_fraction(0.6, 0.5, fraction=0.25, max_fraction=0.05)  # quarter Kelly, capped
kelly.kelly_stake(10_000, 0.6, 0.5, fraction=0.25)              # 500.0
kelly.kelly_fraction_from_odds(0.6, 2.0)                        # same as price 0.5
```

### `edge` - model vs market

Fees charged on profit make a contract at `price` cost
`price / (1 - (1 - price) * fee)` per unit of *net* payout (`effective_price`).

* `edge_abs(p, price, fee)` = `p - effective_price`
* `ev_per_dollar(p, price, fee)` = `p / effective_price - 1`
* `evaluate_binary_market(model_p, yes_price, no_price=None, fee=0, min_edge=0)`
  picks YES / NO / PASS and reports edge, EV and the full-Kelly fraction
* `implied_prob`, `fair_decimal_odds`, `remove_overround` for bookmaker odds

```python
from sports_prob_models import edge

sig = edge.evaluate_binary_market(model_p=0.62, yes_price=0.55, fee=0.02, min_edge=0.02)
sig.side, round(sig.edge, 3), round(sig.kelly_full, 3)   # ('YES', 0.065, 0.146)
```

### `arb` - arbitrage and hedging

* `arbitrage_from_odds([2.1, 2.1], total_stake=100)` - flags `sum(1/odds) < 1`
  and splits the stake so every outcome pays the same (here 105, profit 5).
* `two_venue_arb(book_odds, exchange_price, exchange_fee, bankroll)` - outcome A
  at a bookmaker vs the complementary contract on an exchange with a fee on
  profit. With `g = 1 - (1 - q) fee`, payouts are equal when
  `X = bankroll / (1 + o q / g)` and `N = X o / g`; it is an arbitrage iff
  `1/o + q/g < 1`.
* `hedge_stake(contracts, cost, hedge_odds, fee)` - lock in an existing
  exchange position with a bookmaker bet on the opposite outcome.

```python
from sports_prob_models import arb

r = arb.two_venue_arb(2.3, 0.52, exchange_fee=0.02, bankroll=1000)
r.is_arb, round(r.edge, 4), round(r.stake_book, 2), round(r.cost_exchange, 2), round(r.profit, 2)
# (True, 0.0402, 452.98, 547.02, 41.86)   -- same payout whichever side wins
```

### `matching` - pairing markets across venues

```python
from sports_prob_models import matching

aliases = {"natus vincere": "navi", "na'vi": "navi"}
matching.normalize_name("FaZe Clan")                              # 'faze'
matching.match_signature("NaVi", "FaZe Clan", aliases)            # 'faze__navi'
matching.fuzzy_match("Fase", ["faze", "navi", "liquid"])          # 'faze'
matching.extract_teams_from_question("Will FaZe beat NaVi in Map 1?")  # ('FaZe', 'NaVi')
```

## Status / validation

Read this before using any of it with money.

* **Not calibrated.** None of the models has been fitted to, or back-tested
  against, real match outcomes. The CS2 strength scale, the CT bonus, the pistol
  adjustments, the xG blend weights and the shots-on-target bonus are plausible
  hand-set constants, not estimated parameters.
* **Reference maths, not a strategy.** The package shows *how* round-to-series
  aggregation, Poisson scoreline markets, Kelly sizing and fee-aware arbitrage
  are computed. It does not claim that any of these produce an edge over a
  market.
* **Known simplifications.** Independent Poisson goals (no Dixon-Coles
  correlation, no time-varying intensity); i.i.d. rounds and maps (no momentum,
  economy, map pool or veto effects); a single overtime side convention; no
  liquidity, slippage or settlement-rule modelling in the arbitrage maths.
* **What is tested.** Probabilities stay in `[0, 1]` and sum to 1; symmetry /
  complementarity; monotonicity in strength and score; closed-form Kelly and
  series values; Monte-Carlo vs exact DP agreement; hand-computed Poisson values;
  equal payouts in every arbitrage split. Run `pytest`.
* Built with AI assistance (Claude) from an earlier unfinished side project;
  the code was reviewed and the tests were run, but treat it as a starting
  point.

## Layout

```
sports_prob_models/
  __init__.py
  cs2.py        football.py   kelly.py
  edge.py       arb.py        matching.py
tests/
  test_cs2.py   test_football.py   test_kelly.py
  test_edge.py  test_arb.py        test_matching.py
```

## License

MIT - see `LICENSE`.
