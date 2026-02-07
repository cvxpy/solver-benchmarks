"""Mixed-integer benchmark problems."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from solver_benchmarks.problems import register_problem


@register_problem("mip/knapsack_small", tags=["mip", "small"], description="0-1 knapsack (30 items)")
def knapsack_small(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    n = 30
    values = rng.uniform(1, 100, size=n)
    weights = rng.uniform(1, 50, size=n)
    capacity = weights.sum() * 0.4

    x = cp.Variable(n, boolean=True)
    constraints = [weights @ x <= capacity]
    return cp.Problem(cp.Maximize(values @ x), constraints)


@register_problem("mip/facility_location", tags=["mip", "medium"], description="Facility location (10 facilities, 50 customers)")
def facility_location(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    n_facilities = 10
    n_customers = 50

    # Fixed costs for opening facilities
    fixed_costs = rng.uniform(100, 500, size=n_facilities)
    # Transportation costs
    transport_costs = rng.uniform(1, 20, size=(n_facilities, n_customers))
    # Customer demands
    demands = rng.uniform(1, 10, size=n_customers)
    # Facility capacities
    capacities = rng.uniform(50, 150, size=n_facilities)

    y = cp.Variable(n_facilities, boolean=True)  # open facility
    x = cp.Variable((n_facilities, n_customers), nonneg=True)  # assignment

    constraints = [
        cp.sum(x, axis=0) >= demands,  # meet demand
    ]
    for i in range(n_facilities):
        constraints.append(cp.sum(x[i, :]) <= capacities[i] * y[i])

    cost = fixed_costs @ y + cp.sum(cp.multiply(transport_costs, x))
    return cp.Problem(cp.Minimize(cost), constraints)
