#!/usr/bin/env python
"""Summarize benchmark results.

Usage:
    uv run python scripts/summarize.py --report comparison
    uv run python scripts/summarize.py --report reliability
    uv run python scripts/summarize.py --report fastest --metric solve_time
    uv run python scripts/summarize.py --report comparison --problem-type LP --metric total_time
"""

from __future__ import annotations

import argparse

from solver_benchmarks.analysis import (
    fastest_solver_per_problem,
    format_comparison_table,
    format_reliability_summary,
    solver_comparison_table,
    solver_reliability_summary,
)
from solver_benchmarks.results import load_all_results


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument(
        "--report",
        choices=["comparison", "reliability", "fastest"],
        default="comparison",
        help="Report type",
    )
    parser.add_argument("--metric", default="solve_time", help="Metric for comparison/fastest")
    parser.add_argument("--problem-type", default=None, help="Filter by problem type (LP, QP, SOCP, SDP, MIP)")
    parser.add_argument("--results-dir", default="results", help="Directory containing .jsonl files")
    args = parser.parse_args()

    results = load_all_results(args.results_dir)
    if not results:
        print("No results found. Run benchmarks first.")
        return

    if args.problem_type:
        results = [r for r in results if r.problem_type == args.problem_type]

    if args.report == "comparison":
        table = solver_comparison_table(results, metric=args.metric, problem_type=args.problem_type)
        print(format_comparison_table(table, metric=args.metric))

    elif args.report == "reliability":
        summary = solver_reliability_summary(results)
        print(format_reliability_summary(summary))

    elif args.report == "fastest":
        best = fastest_solver_per_problem(results, metric=args.metric)
        print(f"Fastest solver per problem (metric: {args.metric})")
        print("=" * 60)
        for problem in sorted(best):
            solver, value = best[problem]
            print(f"  {problem:<35} {solver:<15} {value:.4f}s")


if __name__ == "__main__":
    main()
