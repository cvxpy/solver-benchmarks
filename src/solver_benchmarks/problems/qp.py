"""QP benchmark problems."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from solver_benchmarks.problems import register_problem


@register_problem("qp/portfolio_small", tags=["qp", "small"], description="Small portfolio optimization (50 assets)")
def portfolio_small(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    n = 50
    mu = rng.standard_normal(n) * 0.05
    F = rng.standard_normal((n, 10)) * 0.1
    Sigma_sqrt = F
    gamma = 1.0

    x = cp.Variable(n)
    ret = mu @ x
    risk = cp.sum_squares(Sigma_sqrt.T @ x)
    constraints = [cp.sum(x) == 1, x >= 0]
    return cp.Problem(cp.Minimize(-ret + gamma * risk), constraints)


@register_problem("qp/portfolio_medium", tags=["qp", "medium"], description="Medium portfolio optimization (500 assets)")
def portfolio_medium(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    n = 500
    mu = rng.standard_normal(n) * 0.05
    F = rng.standard_normal((n, 30)) * 0.1
    Sigma_sqrt = F
    gamma = 1.0

    x = cp.Variable(n)
    ret = mu @ x
    risk = cp.sum_squares(Sigma_sqrt.T @ x)
    constraints = [cp.sum(x) == 1, x >= 0]
    return cp.Problem(cp.Minimize(-ret + gamma * risk), constraints)


@register_problem("qp/lasso_medium", tags=["qp", "medium"], description="Lasso regression (m=200, n=500)")
def lasso_medium(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    m, n = 200, 500
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    lam = 0.1

    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + lam * cp.norm1(x)))
