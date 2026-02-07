"""Benchmark result dataclass and JSONL serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class BenchmarkResult:
    # Identity
    problem_name: str
    solver_name: str

    # Timing (seconds)
    compilation_time: float | None = None
    solve_time: float | None = None
    setup_time: float | None = None
    total_time: float | None = None

    # Outcome
    status: str = ""
    objective_value: float | None = None
    num_iters: int | None = None

    # Problem size
    problem_type: str = ""
    num_scalar_variables: int | None = None
    num_scalar_eq_constr: int | None = None
    num_scalar_leq_constr: int | None = None

    # Environment
    cvxpy_version: str = ""
    solver_version: str = ""
    python_version: str = ""
    os_info: str = ""
    cpu_info: str = ""
    timestamp: str = ""
    contributor: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkResult:
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


def save_results(results: list[BenchmarkResult], path: str | Path) -> None:
    """Append results to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


def load_results(path: str | Path) -> list[BenchmarkResult]:
    """Load results from a single JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(BenchmarkResult.from_dict(json.loads(line)))
    return results


def load_all_results(directory: str | Path = "results") -> list[BenchmarkResult]:
    """Load results from all .jsonl files in a directory."""
    directory = Path(directory)
    results = []
    for path in sorted(directory.glob("*.jsonl")):
        results.extend(load_results(path))
    return results
