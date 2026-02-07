"""Benchmark execution engine."""

from __future__ import annotations

import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cvxpy as cp

from solver_benchmarks.classify import classify_problem
from solver_benchmarks.problems import ProblemSpec, list_problems
from solver_benchmarks.results import BenchmarkResult, save_results

logger = logging.getLogger(__name__)

SEED = 0


def _env_info() -> dict:
    return {
        "cvxpy_version": cp.__version__,
        "python_version": platform.python_version(),
        "os_info": f"{platform.system()} {platform.release()}",
        "cpu_info": platform.processor() or platform.machine(),
    }


def _solver_version(problem: cp.Problem, solver_name: str) -> str:
    try:
        stats = problem.solver_stats
        if stats is not None and hasattr(stats, "solver_name"):
            return getattr(stats, "solver_name", "")
    except Exception:
        pass
    return ""


def run_single(
    spec: ProblemSpec,
    solver_name: str,
    contributor: str = "",
) -> BenchmarkResult:
    """Run a single (problem, solver) benchmark."""
    env = _env_info()
    problem = spec.func(SEED)
    problem_type = classify_problem(problem)
    metrics = problem.size_metrics

    t0 = time.perf_counter()
    try:
        problem.solve(solver=solver_name)
    except Exception as exc:
        total_time = time.perf_counter() - t0
        logger.warning("Solver %s failed on %s: %s", solver_name, spec.name, exc)
        return BenchmarkResult(
            problem_name=spec.name,
            solver_name=solver_name,
            total_time=total_time,
            status="solver_error",
            problem_type=problem_type,
            num_scalar_variables=metrics.num_scalar_variables,
            num_scalar_eq_constr=metrics.num_scalar_eq_constr,
            num_scalar_leq_constr=metrics.num_scalar_leq_constr,
            timestamp=datetime.now(timezone.utc).isoformat(),
            contributor=contributor,
            **env,
        )
    total_time = time.perf_counter() - t0

    stats = problem.solver_stats
    return BenchmarkResult(
        problem_name=spec.name,
        solver_name=solver_name,
        compilation_time=problem.compilation_time,
        solve_time=stats.solve_time if stats else None,
        setup_time=stats.setup_time if stats else None,
        total_time=total_time,
        status=problem.status,
        objective_value=problem.value if problem.value is not None else None,
        num_iters=stats.num_iters if stats else None,
        problem_type=problem_type,
        num_scalar_variables=metrics.num_scalar_variables,
        num_scalar_eq_constr=metrics.num_scalar_eq_constr,
        num_scalar_leq_constr=metrics.num_scalar_leq_constr,
        solver_version=_solver_version(problem, solver_name),
        timestamp=datetime.now(timezone.utc).isoformat(),
        contributor=contributor,
        **env,
    )


def run_benchmarks(
    problems: list[str] | None = None,
    solvers: list[str] | None = None,
    tags: list[str] | None = None,
    output_dir: str | Path = "results",
    contributor: str = "anonymous",
) -> list[BenchmarkResult]:
    """Run benchmarks for selected problems and solvers."""
    # Select problems
    if problems:
        from solver_benchmarks.problems import get_problem

        specs = [get_problem(p) for p in problems]
    elif tags:
        specs = []
        for tag in tags:
            specs.extend(list_problems(tag=tag))
        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[ProblemSpec] = []
        for s in specs:
            if s.name not in seen:
                seen.add(s.name)
                unique.append(s)
        specs = unique
    else:
        specs = list_problems()

    # Select solvers
    if solvers is None:
        solvers = cp.installed_solvers()

    results: list[BenchmarkResult] = []
    for spec in specs:
        for solver in solvers:
            logger.info("Running %s with %s", spec.name, solver)
            result = run_single(spec, solver, contributor=contributor)
            results.append(result)
            logger.info(
                "  status=%s  total_time=%.3fs", result.status, result.total_time or 0
            )

    # Write results
    output_dir = Path(output_dir)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    plat = platform.system().lower()
    filename = f"{date}_{contributor}_{plat}.jsonl"
    save_results(results, output_dir / filename)

    return results
