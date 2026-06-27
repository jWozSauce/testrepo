"""Team strengths -> expected goals -> a full scoreline probability matrix.

Expected goals for each side come from a multiplicative model (the standard
Dixon-Coles / Maher form):

    lambda_home = base * attack_index[home] * defense_index[away] * home_adv
    lambda_away = base * attack_index[away] * defense_index[home] / home_adv

The scoreline distribution is the outer product of two Poissons with the
Dixon-Coles low-score correction (rho), which fixes the well-known
underestimation of 0-0/1-0/0-1/1-1 results in a plain independent Poisson.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ratings import LEAGUE_AVG_XG, TeamStrength


@dataclass
class MatchModel:
    home: str
    away: str
    lambda_home: float
    lambda_away: float
    matrix: np.ndarray  # matrix[i, j] = P(home scores i, away scores j)


def expected_goals(
    home: TeamStrength,
    away: TeamStrength,
    home_adv: float = 1.0,
    base: float = LEAGUE_AVG_XG,
) -> tuple[float, float]:
    """Opponent-adjusted expected goals for each side.

    `home_adv` defaults to 1.0 (neutral venue) which is appropriate for most
    World Cup fixtures; pass e.g. 1.10 for a host-nation match.
    """
    lam_home = base * home.attack_index * away.defense_index * home_adv
    lam_away = base * away.attack_index * home.defense_index / home_adv
    return float(lam_home), float(lam_away)


def _poisson_pmf(lam: float, k: np.ndarray) -> np.ndarray:
    # P(X=k) = lam^k e^-lam / k!  computed in log space for stability
    from math import lgamma

    log_pmf = k * np.log(lam) - lam - np.array([lgamma(int(i) + 1) for i in k])
    return np.exp(log_pmf)


def _dixon_coles_tau(i: int, j: int, lam: float, mu: float, rho: float) -> float:
    """Low-score dependency correction."""
    if i == 0 and j == 0:
        return 1.0 - lam * mu * rho
    if i == 0 and j == 1:
        return 1.0 + lam * rho
    if i == 1 and j == 0:
        return 1.0 + mu * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
    rho: float = -0.05,
) -> np.ndarray:
    """Return a (max_goals+1) x (max_goals+1) scoreline probability matrix."""
    ks = np.arange(max_goals + 1)
    home_p = _poisson_pmf(lambda_home, ks)
    away_p = _poisson_pmf(lambda_away, ks)
    matrix = np.outer(home_p, away_p)

    for i in (0, 1):
        for j in (0, 1):
            matrix[i, j] *= _dixon_coles_tau(i, j, lambda_home, lambda_away, rho)

    matrix /= matrix.sum()  # renormalise (correction + truncation)
    return matrix


def build_match(
    home: TeamStrength,
    away: TeamStrength,
    home_adv: float = 1.0,
    max_goals: int = 10,
    rho: float = -0.05,
) -> MatchModel:
    lam_h, lam_a = expected_goals(home, away, home_adv=home_adv)
    matrix = score_matrix(lam_h, lam_a, max_goals=max_goals, rho=rho)
    return MatchModel(home.team, away.team, lam_h, lam_a, matrix)
