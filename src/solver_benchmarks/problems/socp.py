"""SOCP benchmark problems."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from solver_benchmarks.problems import register_problem


@register_problem("socp/robust_portfolio", tags=["socp", "medium"], description="Robust portfolio with SOC uncertainty (100 assets)")
def robust_portfolio(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    n = 100
    mu = rng.standard_normal(n) * 0.05
    F = rng.standard_normal((n, 20)) * 0.1
    kappa = 0.1  # uncertainty radius

    x = cp.Variable(n)
    ret = mu @ x
    constraints = [
        cp.sum(x) == 1,
        x >= 0,
        cp.norm2(F.T @ x) <= 0.2,  # risk budget
        ret - kappa * cp.norm2(x) >= 0.01,  # robust return
    ]
    return cp.Problem(cp.Maximize(ret), constraints)


@register_problem("socp/antenna_array", tags=["socp", "medium"], description="Antenna array weight design (40 elements)")
def antenna_array(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)
    n = 40
    m = 100  # number of look directions
    # Steering vectors
    theta = np.linspace(-np.pi / 2, np.pi / 2, m)
    d = 0.5  # half-wavelength spacing
    A = np.exp(1j * 2 * np.pi * d * np.outer(np.arange(n), np.sin(theta)))

    # Desired: unit response at broadside (index m//2), minimize elsewhere
    # Work with real/imag parts for SOCP formulation
    A_real = np.real(A).T  # (m, n)
    A_imag = np.imag(A).T

    w_re = cp.Variable(n)
    w_im = cp.Variable(n)
    t = cp.Variable(m, nonneg=True)

    constraints = [
        A_real[m // 2] @ w_re - A_imag[m // 2] @ w_im == 1,
        A_real[m // 2] @ w_im + A_imag[m // 2] @ w_re == 0,
    ]
    for i in range(m):
        if i == m // 2:
            continue
        resp_re = A_real[i] @ w_re - A_imag[i] @ w_im
        resp_im = A_real[i] @ w_im + A_imag[i] @ w_re
        constraints.append(cp.norm2(cp.hstack([resp_re, resp_im])) <= t[i])

    return cp.Problem(cp.Minimize(cp.sum(t)), constraints)
