#!/usr/bin/env python
"""Run solver benchmarks.

Usage:
    uv run python scripts/run_benchmarks.py --list
    uv run python scripts/run_benchmarks.py --contributor username
    uv run python scripts/run_benchmarks.py --tags lp qp --contributor username
    uv run python scripts/run_benchmarks.py --problems lp/diet_small qp/lasso_medium --solvers SCS CLARABEL
"""

from __future__ import annotations

import argparse
import logging

from solver_benchmarks.problems import list_problems
from solver_benchmarks.runner import run_benchmarks


def main():
    parser = argparse.ArgumentParser(description="Run CVXPY solver benchmarks")
    parser.add_argument("--list", action="store_true", help="List available problems and exit")
    parser.add_argument("--problems", nargs="+", help="Specific problem names to run")
    parser.add_argument("--solvers", nargs="+", help="Specific solver names to run")
    parser.add_argument("--tags", nargs="+", help="Run problems matching these tags")
    parser.add_argument("--contributor", default="anonymous", help="Contributor name for results file")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.list:
        problems = list_problems()
        print(f"{'Name':<35} {'Tags':<25} Description")
        print("-" * 80)
        for p in problems:
            tags = ", ".join(p.tags)
            print(f"{p.name:<35} {tags:<25} {p.description}")
        return

    results = run_benchmarks(
        problems=args.problems,
        solvers=args.solvers,
        tags=args.tags,
        output_dir=args.output_dir,
        contributor=args.contributor,
    )

    # Print summary
    n_optimal = sum(1 for r in results if r.status == "optimal")
    n_total = len(results)
    print(f"\nCompleted {n_total} benchmark runs ({n_optimal} optimal)")

    if results:
        from pathlib import Path
        output_dir = Path(args.output_dir)
        jsonl_files = sorted(output_dir.glob("*.jsonl"))
        if jsonl_files:
            print(f"Results written to: {jsonl_files[-1]}")


if __name__ == "__main__":
    main()
