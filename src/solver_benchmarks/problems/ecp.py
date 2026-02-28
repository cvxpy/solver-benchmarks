"""ECP benchmark problems."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from solver_benchmarks.problems import register_problem
from solver_benchmarks.data.bikeshare import load_bikeshare_data, get_bikeshare_features


@register_problem(
    "ecp/logistic_bikeshare",
    tags=["ecp", "large", "bikeshare"],
    description="Logistic regression with bikeshare data (m ~ 1_000_000, n ~ 300)",
)
def logistic_bikeshare(seed: int = 0) -> cp.Problem:
    df_bikeshare = load_bikeshare_data()
    A_sparse, A_dense = get_bikeshare_features(df_bikeshare)
    y = (df_bikeshare["Member type"] == "Member").to_numpy().astype(float)

    m, n_sparse = A_sparse.shape
    _, n_dense = A_dense.shape

    x_sparse = cp.Variable(n_sparse)
    x_dense = cp.Variable(n_dense)
    x_intercept = cp.Variable(1)

    logit = A_sparse @ x_sparse + A_dense @ x_dense + x_intercept
    loss = cp.sum(cp.logistic(-cp.multiply(y, logit))) / m
    reg = 0.1 * (cp.norm1(x_sparse) + cp.norm1(x_dense))
    return cp.Problem(cp.Minimize(loss + reg))
