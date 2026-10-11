"""sports-prob-models: probability models and bet-sizing math for sports markets.

Pure-Python, dependency-free reference implementations:

- :mod:`sports_prob_models.cs2`       round -> map -> series model for Counter-Strike 2
- :mod:`sports_prob_models.football`  Poisson / xG scoreline matrix -> 1X2, BTTS, totals, handicaps
- :mod:`sports_prob_models.kelly`     full / fractional Kelly sizing for binary markets
- :mod:`sports_prob_models.edge`      model probability vs market price -> edge, fair price, EV
- :mod:`sports_prob_models.arb`       cross-venue arbitrage checks and delta-neutral stake splits
- :mod:`sports_prob_models.matching`  team-name normalisation and fuzzy matching helpers

Nothing here talks to a network, places bets, or depends on third-party packages.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
