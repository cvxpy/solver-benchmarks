# Contributing to solver-benchmarks

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
git clone git@github.com:cvxpy/solver-benchmarks.git
cd solver-benchmarks
uv sync --all-extras   # installs dev and analysis extras
```

## Adding a benchmark problem

Problems live in `src/solver_benchmarks/problems/` and are organized by type
(`lp.py`, `qp.py`, `socp.py`, `sdp.py`, `mip.py`). Each problem is a factory
function decorated with `@register_problem`.

### Example

```python
import cvxpy as cp
import numpy as np
from solver_benchmarks.problems import register_problem

@register_problem(
    name="qp/ridge_small",
    tags=["qp", "small"],
    description="Ridge regression with 100 features",
)
def ridge_small(seed: int) -> cp.Problem:
    rng = np.random.default_rng(seed)
    m, n = 80, 100
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    lam = 1.0
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A @ x - b) + lam * cp.sum_squares(x))
    return cp.Problem(objective)
```

### Guidelines

- **Name** must follow `type/descriptive_name` (e.g. `lp/diet_small`).
- **Tags** must include the problem type (`lp`, `qp`, `socp`, `sdp`, `mip`)
  and a size tag (`small`, `medium`, `large`).
- Always use `np.random.default_rng(seed)` for reproducibility — never use
  `np.random.seed()` or global random state.
- The factory must accept a single `seed: int` argument and return a
  `cp.Problem`.

## Running benchmarks

```bash
# Run all problems against all installed solvers
uv run python scripts/run_benchmarks.py --contributor your_name

# Filter by tag or problem name
uv run python scripts/run_benchmarks.py --tags lp qp --contributor your_name
uv run python scripts/run_benchmarks.py --problems lp/diet_small --solvers SCS CLARABEL

# List registered problems
uv run python scripts/run_benchmarks.py --list
```

Results are saved to `results/` as JSONL files named
`YYYYMMDD_contributor_platform.jsonl`.

## Submitting results

1. Run the benchmarks on your machine.
2. Commit the JSONL file in `results/`.
3. Open a pull request.

## Analyzing results

```bash
# Solver comparison table (default)
uv run python scripts/summarize.py --report comparison

# Solver reliability rates
uv run python scripts/summarize.py --report reliability

# Fastest solver per problem
uv run python scripts/summarize.py --report fastest

# Filter by problem type or change metric
uv run python scripts/summarize.py --report comparison --problem-type LP --metric total_time
```

## Testing

```bash
uv run pytest
```

## Code style

- Keep problem definitions self-contained — each factory should construct its
  own data using the seeded RNG.
- Follow existing naming conventions and file organization.
