"""Build the reviewed, compact Ryzen AI 1.8 publication snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ID = "rai180-20260813"
OUTPUT = REPO_ROOT / "benchmarks" / "results" / "published" / SNAPSHOT_ID

NOMIC_STUDY = REPO_ROOT / "benchmarks" / "results" / "local" / "nomic-rai180-fp32-20260813"
NOMIC_EXTENSION = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "local"
    / "nomic-rai180-fp32-cpu-b32-s512-extension-20260813"
)
BGE_STUDY = REPO_ROOT / "benchmarks" / "results" / "local" / "bge-rai180-compact-20260813"
NOMIC_FIDELITY = REPO_ROOT / "evaluation" / "results" / "local" / "beir-scifact-rai180-fp32"
BGE_FIDELITY = REPO_ROOT / "evaluation" / "results" / "local" / "beir-scifact-bge-rai180-fp32"
PILOT_FIDELITY = REPO_ROOT / "evaluation" / "results" / "local" / "pilot-rai180"
INT8_FAILURE = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "local"
    / "pilot-rai180"
    / "pilot-rai180-npu-int8pc-b1-s128-failure"
    / "failure.json"
)
QWEN_STUDY = REPO_ROOT / "compatibility" / "results" / "local" / "qwen3-rai171-rai180-20260813"
ENVIRONMENT = REPO_ROOT / "environment.local.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def compact_fidelity(path: Path) -> dict[str, Any]:
    raw = read_json(path / "fidelity.json")
    drifts = {}
    for name, drift in raw["drift"].items():
        drifts[name] = {
            key: drift[key]
            for key in (
                "min_cosine",
                "mean_cosine",
                "max_absolute_error",
                "worst_cosine_item",
                "worst_absolute_error_item",
            )
            if key in drift
        }
    evidence = {}
    for name, item in raw["execution_evidence"].items():
        evidence[name] = {
            "model_filename": item["model_filename"],
            "model_sha256": item["model_sha256"],
            "requested_provider": item["requested_provider"],
            "session_providers": item["session_providers"],
            "session_creation_ms": item["session_creation_ms"],
            "npu_verified": item["npu_verified"],
            "assignment": item["assignment"],
        }
    return {
        "recorded_at_utc": raw["recorded_at_utc"],
        "model": raw["model"],
        "dataset": raw["dataset"],
        "decision": raw["decision"],
        "stage_timings_seconds": raw["stage_timings_seconds"],
        "drift": drifts,
        "retrieval": raw["retrieval"],
        "execution_evidence": evidence,
        "source_fidelity_sha256": sha256(path / "fidelity.json"),
    }


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_nomic_matrix() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_summary_rows(NOMIC_STUDY / "summary.csv"):
        rows.append(
            {
                **raw,
                "model": "nomic-ai/nomic-embed-text-v1.5",
                "evidence_level": "confirmatory_n5",
                "notes": "",
            }
        )
    for raw in read_summary_rows(NOMIC_EXTENSION / "summary.csv"):
        rows.append(
            {
                **raw,
                "model": "nomic-ai/nomic-embed-text-v1.5",
                "evidence_level": "exploratory_censored_n2",
                "notes": (
                    "Two long-ceiling samples only; original randomized process timed out at "
                    "2700 seconds. Do not interpret its confidence interval as confirmatory."
                ),
            }
        )
    fieldnames = [
        "model",
        "group_id",
        "precision",
        "provider",
        "batch_size",
        "sequence_length",
        "process_count",
        "inference_mean_ms",
        "inference_ci95_low_ms",
        "inference_ci95_high_ms",
        "throughput_docs_per_second",
        "end_to_end_mean_ms",
        "minimum_npu_node_coverage",
        "evidence_level",
        "notes",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "nomic-matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = read_json(NOMIC_STUDY / "summary.json")
    return rows, summary["cpu_npu_speedups"]


def write_bge_corners() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(BGE_STUDY.glob("*/result.json")):
        raw = read_json(result_path)
        assignment = raw["execution"].get("assignment")
        metrics = raw["metrics"]
        rows.append(
            {
                "model": raw["model"]["source_model"],
                "revision": raw["model"]["source_revision"],
                "provider": raw["execution"]["requested"],
                "batch_size": raw["workload"]["batch_size"],
                "sequence_length": raw["workload"]["sequence_length"],
                "process_count": 1,
                "warmup_iterations": raw["workload"]["warmup_iterations"],
                "timed_iterations": raw["workload"]["timed_iterations"],
                "session_creation_ms": metrics["session_creation_ms"],
                "inference_mean_ms": metrics["inference_only"]["mean_ms"],
                "throughput_docs_per_second": metrics["inference_only"][
                    "throughput_docs_per_second"
                ],
                "end_to_end_mean_ms": metrics["end_to_end"]["mean_ms"],
                "npu_verified": raw["execution"]["npu_verified"],
                "npu_node_coverage": assignment["npu_node_coverage"] if assignment else "",
                "evidence_level": "exploratory_single_process",
            }
        )
    fieldnames = list(rows[0])
    with (OUTPUT / "bge-corners.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def int8_rejection() -> dict[str, Any]:
    raw = read_json(PILOT_FIDELITY / "fidelity.json")
    failure = read_json(INT8_FAILURE)
    drift = raw["drift"]["onnx_dynamic_int8_cpu"]
    retrieval = raw["retrieval"]["onnx_dynamic_int8_cpu"]
    reference = raw["retrieval"]["pytorch_fp32"]
    return {
        "decision": "rejected",
        "candidate": "dynamic INT8 per-channel weights",
        "dataset": raw["dataset"],
        "cpu_fidelity": {
            "minimum_cosine": drift["min_cosine"],
            "maximum_absolute_error": drift["max_absolute_error"],
            "recall_at_10": retrieval["recall_at_10"],
            "reference_recall_at_10": reference["recall_at_10"],
            "ndcg_at_10": retrieval["ndcg_at_10"],
            "reference_ndcg_at_10": reference["ndcg_at_10"],
        },
        "npu_rejection": {
            "error_type": failure["error_type"],
            "error_message": failure["error_message"],
            "assignment_report_sha256": failure["execution"]["assignment_report"]["sha256"],
        },
        "source_fidelity_sha256": sha256(PILOT_FIDELITY / "fidelity.json"),
        "source_failure_sha256": sha256(INT8_FAILURE),
    }


def failure_signature(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "OrtValue shape verification failed" in text:
        return {
            "class": "generation_logits_shape_mismatch",
            "current_shape": [1, 1, 151669],
            "requested_shape": [1, 15, 151669],
        }
    if "SkipSimplifiedLayerNormalizationBf" in text and "tensor(float16)" in text:
        return {
            "class": "hybrid_model_load_dtype_rejection",
            "operator": "SkipSimplifiedLayerNormalizationBf",
            "rejected_input_type": "tensor(float16)",
        }
    return {"class": "unclassified_probe_failure", "log_sha256": sha256(log_path)}


def write_qwen() -> dict[str, Any]:
    raw = read_json(QWEN_STUDY / "matrix.json")
    cells = []
    for cell in raw["cells"]:
        packages = cell["environment"]["environment"]["packages"]
        cell_dir = QWEN_STUDY / cell["cell_id"]
        cells.append(
            {
                "cell_id": cell["cell_id"],
                "sdk_version": cell["sdk_version"],
                "include_hidden_states": cell["include_hidden_states"],
                "precision": cell["precision"],
                "oga_execution_provider": cell["oga_execution_provider"],
                "packages": packages,
                "stages": cell["stages"],
                "passed": cell["passed"],
                "failure_signature": failure_signature(cell_dir / "hybrid-probe.log"),
            }
        )
    result = {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "model_snapshot_revision": raw["model_snapshot_revision"],
        "scope_note": (
            "Controlled FP16 OGA DirectML-to-hybrid reproducer. This is not the canonical "
            "AWQ flow and must not be generalized beyond the recorded commands."
        ),
        "cells": cells,
        "source_matrix_sha256": sha256(QWEN_STUDY / "matrix.json"),
    }
    write_json(OUTPUT / "qwen-compatibility.json", result)
    return result


def svg_text(x: float, y: float, value: str, size: int = 13, anchor: str = "middle") -> str:
    escaped = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Segoe UI,Arial,sans-serif" font-size="{size}">{escaped}</text>'
    )


def write_speedup_svg(speedups: dict[str, Any]) -> None:
    ordered = sorted(
        speedups.items(),
        key=lambda item: (
            int(item[0].split("-b", 1)[1].split("-", 1)[0]),
            int(item[0].rsplit("-s", 1)[1]),
        ),
    )
    width, height = 960, 520
    left, right, top, bottom = 75, 25, 70, 110
    chart_width, chart_height = width - left - right, height - top - bottom
    maximum = 7.0
    bar_width = chart_width / len(ordered) * 0.62
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 32, "Nomic FP32 CPU / verified-NPU inference speedup", 20),
        svg_text(width / 2, 54, "Five independent processes per displayed cell", 12),
    ]
    for tick in range(0, 8):
        y = top + chart_height * (1 - tick / maximum)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(svg_text(left - 10, y + 4, f"{tick}×", 11, "end"))
    slot = chart_width / len(ordered)
    for index, (key, item) in enumerate(ordered):
        value = float(item["cpu_over_npu_inference_speedup"])
        x = left + slot * (index + 0.5)
        y = top + chart_height * (1 - value / maximum)
        parts.append(
            f'<rect x="{x - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{top + chart_height - y:.1f}" rx="3" fill="#7c3aed"/>'
        )
        parts.append(svg_text(x, y - 7, f"{value:.2f}×", 11))
        label = key.replace("fp32-", "").replace("-s", "/s").replace("b", "b")
        parts.append(svg_text(x, top + chart_height + 24, label, 11))
    parts.append(svg_text(width / 2, height - 28, "Batch / sequence length", 13))
    parts.append("</svg>\n")
    (OUTPUT / "nomic-speedup.svg").write_text("\n".join(parts), encoding="utf-8", newline="\n")


def write_fidelity_svg(fidelity: dict[str, Any]) -> None:
    width, height = 960, 520
    left, right, top, bottom = 80, 30, 75, 95
    chart_width, chart_height = width - left - right, height - top - bottom
    series = []
    for model_key, model_label in (("nomic", "Nomic"), ("bge", "BGE")):
        retrieval = fidelity[model_key]["retrieval"]
        for runtime, runtime_label in (
            ("pytorch_fp32", "PyTorch"),
            ("onnx_fp32_cpu", "ONNX CPU"),
            ("onnx_fp32_npu", "ONNX NPU"),
        ):
            series.append(
                (
                    f"{model_label}\n{runtime_label}",
                    float(retrieval[runtime]["ndcg_at_10"]),
                    float(retrieval[runtime]["recall_at_10"]),
                )
            )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 32, "Official BEIR SciFact retrieval fidelity", 20),
        svg_text(width / 2, 55, "FP32 reference, ONNX CPU, and fail-closed NPU", 12),
    ]
    y_min, y_max = 0.55, 0.85
    for index in range(7):
        value = y_min + index * 0.05
        y = top + chart_height * (1 - (value - y_min) / (y_max - y_min))
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(svg_text(left - 10, y + 4, f"{value:.2f}", 11, "end"))
    slot = chart_width / len(series)
    bar_width = slot * 0.28
    for index, (label, ndcg, recall) in enumerate(series):
        x = left + slot * (index + 0.5)
        for offset, value, color in (
            (-bar_width / 2, ndcg, "#7c3aed"),
            (bar_width / 2, recall, "#0891b2"),
        ):
            y = top + chart_height * (1 - (value - y_min) / (y_max - y_min))
            parts.append(
                f'<rect x="{x + offset - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{top + chart_height - y:.1f}" fill="{color}"/>'
            )
        first, second = label.split("\n")
        parts.append(svg_text(x, top + chart_height + 22, first, 11))
        parts.append(svg_text(x, top + chart_height + 38, second, 10))
    parts.extend(
        [
            '<rect x="350" y="485" width="14" height="14" fill="#7c3aed"/>',
            svg_text(371, 497, "nDCG@10", 12, "start"),
            '<rect x="480" y="485" width="14" height="14" fill="#0891b2"/>',
            svg_text(501, 497, "Recall@10", 12, "start"),
            "</svg>\n",
        ]
    )
    (OUTPUT / "fidelity.svg").write_text("\n".join(parts), encoding="utf-8", newline="\n")


def main() -> None:
    required = [
        NOMIC_STUDY / "summary.json",
        NOMIC_EXTENSION / "summary.json",
        NOMIC_FIDELITY / "fidelity.json",
        BGE_FIDELITY / "fidelity.json",
        QWEN_STUDY / "matrix.json",
        ENVIRONMENT,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing publication inputs: " + ", ".join(map(str, missing)))

    matrix_rows, speedups = write_nomic_matrix()
    bge_rows = write_bge_corners()
    fidelity = {
        "nomic": compact_fidelity(NOMIC_FIDELITY),
        "bge": compact_fidelity(BGE_FIDELITY),
        "dynamic_int8_negative_result": int8_rejection(),
    }
    write_json(OUTPUT / "fidelity.json", fidelity)
    qwen = write_qwen()
    write_speedup_svg(speedups)
    write_fidelity_svg(fidelity)

    environment = read_json(ENVIRONMENT)
    source_paths = [
        NOMIC_STUDY / "summary.json",
        NOMIC_STUDY / "study.json",
        NOMIC_EXTENSION / "summary.json",
        NOMIC_FIDELITY / "fidelity.json",
        BGE_FIDELITY / "fidelity.json",
        PILOT_FIDELITY / "fidelity.json",
        INT8_FAILURE,
        QWEN_STUDY / "matrix.json",
        ENVIRONMENT,
    ]
    recorded_times = [
        fidelity["nomic"]["recorded_at_utc"],
        fidelity["bge"]["recorded_at_utc"],
        read_json(QWEN_STUDY / "matrix.json")["recorded_at_utc"],
    ]
    manifest = {
        "schema_version": "1.0.0",
        "snapshot_id": SNAPSHOT_ID,
        "recorded_through_utc": max(recorded_times),
        "scope": {
            "nomic_confirmatory_groups": sum(
                row["evidence_level"] == "confirmatory_n5" for row in matrix_rows
            ),
            "nomic_censored_groups": sum(
                row["evidence_level"] == "exploratory_censored_n2" for row in matrix_rows
            ),
            "bge_exploratory_groups": len(bge_rows),
            "qwen_compatibility_cells": len(qwen["cells"]),
        },
        "environment": environment,
        "limitations": [
            "One laptop; no cross-machine replication.",
            "Balanced AC power profile; resident desktop applications were present.",
            "No calibrated power, energy, temperature, or sustained-load measurement.",
            "CPU b32/s512 is censored: one 45-minute timeout and two long-ceiling samples.",
            "BGE corner timings are single-process exploratory checks.",
            "Qwen compatibility uses a controlled FP16 OGA DirectML-to-hybrid path, not AWQ.",
        ],
        "source_artifacts": [
            {"path": relative(path), "sha256": sha256(path)} for path in source_paths
        ],
        "published_files": [],
    }
    write_json(OUTPUT / "manifest.json", manifest)
    published = sorted(path for path in OUTPUT.iterdir() if path.name != "manifest.json")
    manifest["published_files"] = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in published
    ]
    write_json(OUTPUT / "manifest.json", manifest)
    print(f"publication snapshot written: {OUTPUT}")


if __name__ == "__main__":
    main()
