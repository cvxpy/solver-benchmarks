"""Summary and comparison tools for benchmark results."""

from __future__ import annotations

from collections import defaultdict

from solver_benchmarks.results import BenchmarkResult


def solver_comparison_table(
    results: list[BenchmarkResult],
    metric: str = "solve_time",
    problem_type: str | None = None,
) -> dict[str, dict[str, float | None]]:
    """Build a comparison table: rows=problems, cols=solvers.

    Returns ``{problem_name: {solver_name: metric_value}}``.
    """
    filtered = results
    if problem_type:
        filtered = [r for r in filtered if r.problem_type == problem_type]

    problems = sorted({r.problem_name for r in filtered})
    solvers = sorted({r.solver_name for r in filtered})

    table: dict[str, dict[str, float | None]] = {}
    lookup: dict[tuple[str, str], BenchmarkResult] = {
        (r.problem_name, r.solver_name): r for r in filtered
    }

    for p in problems:
        row: dict[str, float | None] = {}
        for s in solvers:
            r = lookup.get((p, s))
            if r is None:
                row[s] = None
            else:
                row[s] = getattr(r, metric, None)
        table[p] = row
    return table


def solver_reliability_summary(
    results: list[BenchmarkResult],
) -> dict[str, dict[str, dict[str, int]]]:
    """Compute success rates per solver per problem type.

    Returns ``{solver: {problem_type: {"total": N, "optimal": M}}}``.
    """
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "optimal": 0})
    )
    for r in results:
        entry = counts[r.solver_name][r.problem_type]
        entry["total"] += 1
        if r.status == "optimal":
            entry["optimal"] += 1
    return dict(counts)


def fastest_solver_per_problem(
    results: list[BenchmarkResult],
    metric: str = "solve_time",
) -> dict[str, tuple[str, float]]:
    """Return the fastest solver for each problem.

    Returns ``{problem_name: (solver_name, metric_value)}``.
    """
    by_problem: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        if r.status == "optimal":
            by_problem[r.problem_name].append(r)

    best: dict[str, tuple[str, float]] = {}
    for problem, runs in by_problem.items():
        valid = [(r, getattr(r, metric, None)) for r in runs]
        valid = [(r, v) for r, v in valid if v is not None]
        if valid:
            r, v = min(valid, key=lambda x: x[1])
            best[problem] = (r.solver_name, v)
    return best


def results_to_dataframe(results: list[BenchmarkResult]):
    """Convert results to a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame([r.to_dict() for r in results])


def format_comparison_table(
    table: dict[str, dict[str, float | None]],
    metric: str = "solve_time",
) -> str:
    """Format a comparison table as a readable string."""
    if not table:
        return "No results to display."

    solvers = sorted({s for row in table.values() for s in row})
    col_widths = {s: max(len(s), 10) for s in solvers}
    name_width = max(len(p) for p in table)

    header = f"{'Problem':<{name_width}}  " + "  ".join(
        f"{s:>{col_widths[s]}}" for s in solvers
    )
    sep = "-" * len(header)

    lines = [f"Metric: {metric}", sep, header, sep]
    for problem, row in sorted(table.items()):
        vals = []
        for s in solvers:
            v = row.get(s)
            if v is None:
                vals.append(f"{'—':>{col_widths[s]}}")
            else:
                vals.append(f"{v:>{col_widths[s]}.4f}")
        lines.append(f"{problem:<{name_width}}  " + "  ".join(vals))
    lines.append(sep)
    return "\n".join(lines)


def format_reliability_summary(
    summary: dict[str, dict[str, dict[str, int]]],
) -> str:
    """Format reliability summary as a readable string."""
    lines = ["Solver Reliability Summary", "=" * 40]
    for solver in sorted(summary):
        lines.append(f"\n{solver}:")
        for ptype in sorted(summary[solver]):
            entry = summary[solver][ptype]
            total = entry["total"]
            optimal = entry["optimal"]
            pct = (optimal / total * 100) if total > 0 else 0
            lines.append(f"  {ptype:>6}: {optimal}/{total} ({pct:.0f}%)")
    return "\n".join(lines)
