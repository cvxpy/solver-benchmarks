"""SDP benchmark problems."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from solver_benchmarks.problems import register_problem


@register_problem(
    "sdp/max_cut_small",
    tags=["sdp", "small"],
    description="Max-cut SDP relaxation (20 nodes)",
)
def max_cut_small(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    n = 20
    # Random graph adjacency (symmetric, no self-loops)
    W = rng.uniform(0, 1, size=(n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    L = np.diag(W.sum(axis=1)) - W  # Laplacian

    X = cp.Variable((n, n), symmetric=True)
    constraints = [
        cp.diag(X) == 1,
        X >> 0,
    ]
    return cp.Problem(cp.Maximize(0.25 * cp.trace(L @ X)), constraints)


@register_problem(
    "sdp/matrix_completion",
    tags=["sdp", "medium"],
    description="Low-rank matrix completion (30x30, 50% observed)",
)
def matrix_completion(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    n = 30
    rank = 3
    U = rng.standard_normal((n, rank))
    V = rng.standard_normal((rank, n))
    M_true = U @ V

    # Observe 50% of entries
    mask = rng.random((n, n)) < 0.5

    X = cp.Variable((n, n))
    constraints = [X[mask] == M_true[mask]]
    return cp.Problem(cp.Minimize(cp.normNuc(X)), constraints)


@register_problem(
    "sdp/knockoff_ar_model",
    tags=["sdp", "medium"],
    description="Knockoff filter AR(1) model (n=100, rho=0.5)",
)
def knockoff_ar_model_medium(seed: int = 0) -> cp.Problem:
    n = 100
    rho = 0.5
    Sigma = rho ** (np.abs(np.arange(n) - np.arange(n)[:, None]))

    s = cp.Variable(n)
    constraints = [s >= 0, cp.PSD(2 * Sigma - cp.diag(s))]
    objective = cp.Minimize(cp.sum(cp.abs(1 - s)))
    return cp.Problem(objective, constraints)


@register_problem(
    "sdp/knockoff_equi",
    tags=["sdp", "medium"],
    description="Knockoff filter equi correlated model (n=100, rho = 0.9)",
)
def knockoff_equi(seed: int = 0) -> cp.Problem:
    n = 100
    rho = 0.9

    Sigma = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)

    s = cp.Variable(n)
    constraints = [s >= 0, cp.PSD(2 * Sigma - cp.diag(s))]
    objective = cp.Minimize(cp.sum(cp.abs(1 - s)))
    return cp.Problem(objective, constraints)

@register_problem(
    "sdp/knockoff_block_equi",
    tags=["sdp", "medium"],
    description="Knockoff filter equi correlated model (n=100, rho = 0.9)",
)
def knockoff_block_equi(seed: int = 0) -> cp.Problem:
    n = 100
    rho = 0.9

    Sigma = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)

    s = cp.Variable(n)
    constraints = [s >= 0, cp.PSD(2 * Sigma - cp.diag(s))]
    objective = cp.Minimize(cp.sum(cp.abs(1 - s)))
    return cp.Problem(objective, constraints)

