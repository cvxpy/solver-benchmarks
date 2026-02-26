import numpy as np
import cvxpy as cp

from solver_benchmarks.problems import register_problem


@register_problem(
    "sdp/nearest_correlation_small",
    tags=["sdp", "small"],
    description="Nearest correlation matrix repair (n=30, Higham 2002)",
)
def nearest_correlation_small(seed: int = 0) -> cp.Problem:
    rng = np.random.default_rng(seed)

    n = 30

    A = rng.standard_normal((n, n))
    base = A.T @ A

    d = np.sqrt(np.diag(base))
    d[d == 0] = 1.0
    valid = base / np.outer(d, d)

    noise = rng.normal(0, 0.2, size=(n, n))
    noise = (noise + noise.T) / 2

    G = valid + noise
    np.fill_diagonal(G, 1.0)

    X = cp.Variable((n, n), symmetric=True)

    return cp.Problem(
        cp.Minimize(cp.sum_squares(X - G)),
        [X >> 0, cp.diag(X) == 1],
    )