"""Aggregate independent benchmark processes without hiding terminal failures."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npu_study.artifacts import sha256_file, utc_now, write_json  # noqa: E402

T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "ci95_low": None,
            "ci95_high": None,
            "minimum": None,
            "maximum": None,
        }
    mean = statistics.fmean(values)
    if len(values) == 1:
        standard_deviation = None
        ci_low = None
        ci_high = None
    else:
        standard_deviation = statistics.stdev(values)
        critical = T_CRITICAL_975.get(len(values) - 1, 1.96)
        half_width = critical * standard_deviation / (len(values) ** 0.5)
        ci_low = mean - half_width
        ci_high = mean + half_width
    return {
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "standard_deviation": standard_deviation,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "minimum": min(values),
        "maximum": max(values),
    }


def scheduled_run_id(study_id: str, entry: dict[str, Any]) -> str:
    return (
        f"{study_id}-{entry['provider']}-{entry['precision']}-"
        f"b{entry['batch_size']}-s{entry['sequence_length']}-r{entry['repeat']}"
    )


def run_outcome(run_dir: Path) -> str | None:
    markers = {
        "result": run_dir / "result.json",
        "failure": run_dir / "failure.json",
        "skipped": run_dir / "skipped.json",
    }
    present = [name for name, path in markers.items() if path.is_file()]
    if len(present) > 1:
        raise ValueError(f"run has conflicting terminal markers: {run_dir}")
    if not present:
        return None
    if present[0] in {"result", "failure"} and not (run_dir / "process.log").is_file():
        return None
    return present[0]


def aggregate(
    study_dir: Path, allow_incomplete: bool = False, allow_dirty: bool = False
) -> dict[str, Any]:
    study_path = study_dir / "study.json"
    study = load_json(study_path)
    study_id = str(study["study_id"])
    outcomes: dict[str, list[str]] = defaultdict(list)
    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing: list[str] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry in study["schedule"]:
        run_id = scheduled_run_id(study_id, entry)
        run_dir = study_dir / run_id
        outcome = run_outcome(run_dir)
        if outcome is None:
            missing.append(run_id)
            continue
        outcomes[outcome].append(run_id)
        if outcome == "result":
            result = load_json(run_dir / "result.json")
            expected = {
                "run_id": run_id,
                "precision": entry["precision"],
                "provider": entry["provider"],
                "batch_size": entry["batch_size"],
                "sequence_length": entry["sequence_length"],
                "repeat": entry["repeat"],
            }
            actual = {
                "run_id": result["run_id"],
                "precision": result["model"]["precision"],
                "provider": result["execution"]["requested"],
                "batch_size": result["workload"]["batch_size"],
                "sequence_length": result["workload"]["sequence_length"],
                "repeat": result["workload"]["repeat"],
            }
            if actual != expected:
                raise ValueError(f"result does not match scheduled cell: {run_id}")
            results.append((result, load_json(run_dir / "environment.json")))
        elif outcome == "failure":
            failures.append(load_json(run_dir / "failure.json"))
        else:
            skipped.append(load_json(run_dir / "skipped.json"))
    if missing and not allow_incomplete:
        raise ValueError(
            f"study has {len(missing)} non-terminal scheduled runs; first missing run: {missing[0]}"
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    git_commits: set[str] = set()
    dirty_runs: list[str] = []
    for result, environment in results:
        workload = result["workload"]
        provider = result["execution"]["requested"]
        precision = result["model"]["precision"]
        group_id = (
            f"{precision}-{provider}-b{workload['batch_size']}-s{workload['sequence_length']}"
        )
        assignment = result["execution"].get("assignment")
        git_commits.add(str(environment.get("git_commit")))
        if environment.get("git_tracked_dirty"):
            dirty_runs.append(result["run_id"])
        groups[group_id].append(
            {
                "run_id": result["run_id"],
                "repeat": workload["repeat"],
                "precision": precision,
                "provider": provider,
                "batch_size": workload["batch_size"],
                "sequence_length": workload["sequence_length"],
                "inference_mean_ms": result["metrics"]["inference_only"]["mean_ms"],
                "inference_p95_ms": result["metrics"]["inference_only"]["p95_ms"],
                "inference_throughput_docs_per_second": result["metrics"]["inference_only"][
                    "throughput_docs_per_second"
                ],
                "end_to_end_mean_ms": result["metrics"]["end_to_end"]["mean_ms"],
                "session_creation_ms": result["metrics"]["session_creation_ms"],
                "cache_present_before_session": (result["execution"].get("npu_cache") or {}).get(
                    "present_before_session"
                ),
                "npu_node_coverage": assignment.get("npu_node_coverage") if assignment else None,
                "npu_subgraphs": assignment.get("npu_subgraphs") if assignment else None,
                "git_commit": environment.get("git_commit"),
            }
        )
    if dirty_runs and not allow_dirty:
        raise ValueError(
            f"study contains {len(dirty_runs)} run(s) from a dirty tracked tree; "
            f"first dirty run: {dirty_runs[0]}"
        )

    group_summaries: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    for group_id, processes in sorted(groups.items()):
        first = processes[0]
        summary = {
            "precision": first["precision"],
            "provider": first["provider"],
            "batch_size": first["batch_size"],
            "sequence_length": first["sequence_length"],
            "process_count": len(processes),
            "inference_mean_ms": distribution(
                [float(item["inference_mean_ms"]) for item in processes]
            ),
            "inference_p95_ms": distribution(
                [float(item["inference_p95_ms"]) for item in processes]
            ),
            "inference_throughput_docs_per_second": distribution(
                [float(item["inference_throughput_docs_per_second"]) for item in processes]
            ),
            "end_to_end_mean_ms": distribution(
                [float(item["end_to_end_mean_ms"]) for item in processes]
            ),
            "session_creation_ms": distribution(
                [float(item["session_creation_ms"]) for item in processes]
            ),
            "cold_session_creation_ms": distribution(
                [
                    float(item["session_creation_ms"])
                    for item in processes
                    if item["cache_present_before_session"] is False
                ]
            ),
            "warm_session_creation_ms": distribution(
                [
                    float(item["session_creation_ms"])
                    for item in processes
                    if item["cache_present_before_session"] is True
                ]
            ),
            "minimum_npu_node_coverage": min(
                (
                    float(item["npu_node_coverage"])
                    for item in processes
                    if item["npu_node_coverage"] is not None
                ),
                default=None,
            ),
            "processes": sorted(processes, key=lambda item: int(item["repeat"])),
        }
        group_summaries[group_id] = summary
        csv_rows.append(
            {
                "group_id": group_id,
                "precision": summary["precision"],
                "provider": summary["provider"],
                "batch_size": summary["batch_size"],
                "sequence_length": summary["sequence_length"],
                "process_count": summary["process_count"],
                "inference_mean_ms": summary["inference_mean_ms"]["mean"],
                "inference_ci95_low_ms": summary["inference_mean_ms"]["ci95_low"],
                "inference_ci95_high_ms": summary["inference_mean_ms"]["ci95_high"],
                "throughput_docs_per_second": summary["inference_throughput_docs_per_second"][
                    "mean"
                ],
                "end_to_end_mean_ms": summary["end_to_end_mean_ms"]["mean"],
                "minimum_npu_node_coverage": summary["minimum_npu_node_coverage"],
            }
        )

    speedups: dict[str, dict[str, Any]] = {}
    for group_id, cpu in group_summaries.items():
        if cpu["provider"] != "cpu":
            continue
        npu_id = group_id.replace("-cpu-", "-npu-", 1)
        npu = group_summaries.get(npu_id)
        if npu is None:
            continue
        cpu_mean = float(cpu["inference_mean_ms"]["mean"])
        npu_mean = float(npu["inference_mean_ms"]["mean"])
        speedup_id = f"{cpu['precision']}-b{cpu['batch_size']}-s{cpu['sequence_length']}"
        speedups[speedup_id] = {
            "cpu_group": group_id,
            "npu_group": npu_id,
            "cpu_over_npu_inference_speedup": cpu_mean / npu_mean,
        }

    return {
        "schema_version": "1.0.0",
        "generated_at_utc": utc_now(),
        "study_id": study_id,
        "study_sha256": sha256_file(study_path),
        "study_status": study.get("status"),
        "terminal_outcomes": {name: len(values) for name, values in sorted(outcomes.items())},
        "missing_run_ids": missing,
        "failure_records": failures,
        "skipped_records": skipped,
        "reproducibility": {
            "git_commits": sorted(git_commits),
            "tracked_dirty_run_ids": dirty_runs,
        },
        "groups": group_summaries,
        "cpu_npu_speedups": speedups,
        "csv_rows": csv_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["group_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate a publication benchmark study")
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.study_dir
    result = aggregate(
        args.study_dir,
        allow_incomplete=args.allow_incomplete,
        allow_dirty=args.allow_dirty,
    )
    csv_rows = result.pop("csv_rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", result)
    write_csv(output_dir / "summary.csv", csv_rows)
    print(
        f"aggregated {result['terminal_outcomes'].get('result', 0)} results into "
        f"{len(result['groups'])} groups: {output_dir}"
    )


if __name__ == "__main__":
    main()
