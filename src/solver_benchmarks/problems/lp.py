"""LP benchmark problems."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from solver_benchmarks.problems import register_problem


@register_problem("lp/diet_small", tags=["lp", "small"], description="Small diet problem (20 foods, 10 nutrients)")
def diet_small(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    n_foods, n_nutrients = 20, 10
    costs = rng.uniform(1, 10, size=n_foods)
    A = rng.uniform(0, 1, size=(n_nutrients, n_foods))
    b_low = rng.uniform(1, 3, size=n_nutrients)
    b_high = b_low + rng.uniform(1, 5, size=n_nutrients)

    x = cp.Variable(n_foods, nonneg=True)
    constraints = [A @ x >= b_low, A @ x <= b_high, x <= 10]
    return cp.Problem(cp.Minimize(costs @ x), constraints)


@register_problem("lp/transportation_medium", tags=["lp", "medium"], description="Transportation problem (50 sources, 100 sinks)")
def transportation_medium(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    m, n = 50, 100
    costs = rng.uniform(1, 20, size=(m, n))
    supply = rng.uniform(10, 50, size=m)
    demand = rng.uniform(1, 10, size=n)
    # Scale demand to match total supply
    demand = demand * (supply.sum() / demand.sum())

    X = cp.Variable((m, n), nonneg=True)
    constraints = [
        cp.sum(X, axis=1) <= supply,
        cp.sum(X, axis=0) >= demand,
    ]
    return cp.Problem(cp.Minimize(cp.sum(cp.multiply(costs, X))), constraints)


@register_problem("lp/basis_pursuit_large", tags=["lp", "large"], description="Basis pursuit via LP (m=200, n=1000)")
def basis_pursuit_large(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    m, n = 200, 1000
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    x_true[rng.random(n) > 0.1] = 0  # ~90% sparse
    b = A @ x_true

    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(cp.norm1(x)), [A @ x == b])
