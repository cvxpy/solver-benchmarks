"""QP benchmark problems."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from solver_benchmarks.problems import register_problem
from solver_benchmarks.data.bikeshare import load_bikeshare_data, get_bikeshare_features


@register_problem(
    "qp/portfolio_small",
    tags=["qp", "small"],
    description="Small portfolio optimization (50 assets)",
)
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


@register_problem(
    "qp/portfolio_medium",
    tags=["qp", "medium"],
    description="Medium portfolio optimization (500 assets)",
)
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


@register_problem(
    "qp/lasso_medium",
    tags=["qp", "medium"],
    description="Lasso regression (m=200, n=500)",
)
def lasso_medium(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    m, n = 200, 500
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    lam = 0.1

    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + lam * cp.norm1(x)))


@register_problem(
    "qp/lasso_bikeshare",
    tags=["qp", "large", "bikeshare"],
    description="Lasso regression with bikeshare data (m ~ 1_000_000, n ~ 300)",
)
def lasso_bikeshare(seed: int = 0) -> cp.Problem:
    df_bikeshare = load_bikeshare_data()
    A_sparse, A_dense = get_bikeshare_features(df_bikeshare)
    b = np.log(df_bikeshare["Duration"].to_numpy())

    m, n_sparse = A_sparse.shape
    _, n_dense = A_dense.shape

    x_sparse = cp.Variable(n_sparse)
    x_dense = cp.Variable(n_dense)
    x_intercept = cp.Variable(1)
    b_hat = A_sparse @ x_sparse + A_dense @ x_dense + x_intercept
    SSE = 0.5 * cp.sum_squares(b_hat - b) / m
    penalty = cp.norm1(x_sparse) / n_sparse + cp.norm1(x_dense)

    return cp.Problem(cp.Minimize(SSE + penalty))

@register_problem(
    "qp/ridge_bikeshare",
    tags=["qp", "large", "bikeshare"],
    description="Ridge regression with bikeshare data (m ~ 1_000_000, n ~ 300)",
)
def ridge_bikeshare(seed: int = 0) -> cp.Problem:
    df_bikeshare = load_bikeshare_data()
    A_sparse, A_dense = get_bikeshare_features(df_bikeshare)
    b = np.log(df_bikeshare["Duration"].to_numpy())

    m, n_sparse = A_sparse.shape
    _, n_dense = A_dense.shape

    x_sparse = cp.Variable(n_sparse)
    x_dense = cp.Variable(n_dense)
    x_intercept = cp.Variable(1)
    b_hat = A_sparse @ x_sparse + A_dense @ x_dense + x_intercept
    SSE = 0.5 * cp.sum_squares(b_hat - b) / m
    penalty = cp.norm2(x_sparse) / n_sparse + cp.norm2(x_dense)

    return cp.Problem(cp.Minimize(SSE + penalty))

@register_problem(
    "qp/elastic_net_bikeshare",
    tags=["qp", "large", "bikeshare"],
    description="Elastic net regression with bikeshare data (m ~ 1_000_000, n ~ 300)",
)
def elastic_net_bikeshare(seed: int = 0) -> cp.Problem:
    df_bikeshare = load_bikeshare_data()
    A_sparse, A_dense = get_bikeshare_features(df_bikeshare)
    b = np.log(df_bikeshare["Duration"].to_numpy())

    m, n_sparse = A_sparse.shape
    _, n_dense = A_dense.shape

    x_sparse = cp.Variable(n_sparse)
    x_dense = cp.Variable(n_dense)
    x_intercept = cp.Variable(1)
    b_hat = A_sparse @ x_sparse + A_dense @ x_dense + x_intercept
    SSE = 0.5 * cp.sum_squares(b_hat - b) / m
    penalty1 = cp.norm1(x_sparse) / n_sparse + cp.norm1(x_dense)
    penalty2 = cp.norm2(x_sparse) / n_sparse + cp.norm2(x_dense)

    return cp.Problem(cp.Minimize(SSE + penalty1 + penalty2))