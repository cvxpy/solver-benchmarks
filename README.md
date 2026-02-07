# solver-benchmarks

A benchmark suite for evaluating and comparing CVXPY solver performance across
a diverse set of convex optimization problems. Each benchmark problem uses
seeded random data for reproducibility, and results are stored as JSONL files
so that contributors can share and compare runs across different machines and
solver versions.

## Quick start

```bash
git clone git@github.com:cvxpy/solver-benchmarks.git
cd solver-benchmarks
uv sync

# List available problems
uv run python scripts/run_benchmarks.py --list

# Run all benchmarks
uv run python scripts/run_benchmarks.py --contributor your_name

# Summarize results
uv run python scripts/summarize.py --report comparison
```

## Project structure

```
src/solver_benchmarks/
  problems/       Problem definitions (LP, QP, MIP, SOCP, SDP)
  runner.py        Benchmark execution engine
  results.py       JSONL serialization for benchmark results
  classify.py      Automatic problem type classification
  analysis.py      Reporting and analysis utilities
scripts/
  run_benchmarks.py   CLI to run benchmarks
  summarize.py        CLI to analyze results
tests/               pytest test suite
results/             Benchmark result files (JSONL)
```

## Available problems

| Type | Problems |
|------|----------|
| LP   | `lp/diet_small`, `lp/transportation_medium`, `lp/basis_pursuit_large` |
| QP   | `qp/portfolio_small`, `qp/portfolio_medium`, `qp/lasso_medium` |
| MIP  | `mip/knapsack_small`, `mip/facility_location` |
| SOCP | `socp/robust_portfolio`, `socp/antenna_array` |
| SDP  | `sdp/max_cut_small`, `sdp/matrix_completion` |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add problems, run
benchmarks, and submit results.
