"""Run the versioned Qwen3 OGA hidden-state compatibility matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npu_study.artifacts import sha256_file, utc_now, write_json  # noqa: E402

PROBE = Path(__file__).parent / "probe_oga.py"
SDK_ENVS = {"1.7.1": "ryzen-ai-1.7.1", "1.8.0": "ryzen-ai-1.8.0"}
HIDDEN_STATES = (False, True)


def sanitized(text: str, replacements: dict[str, str]) -> str:
    for source, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        text = re.sub(re.escape(source), replacement, text, flags=re.IGNORECASE)
    return text


def terminate_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        process.kill()


def run_stage(
    *,
    name: str,
    command: list[str],
    cell_dir: Path,
    timeout_seconds: float,
    replacements: dict[str, str],
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.perf_counter()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_tree(process)
        stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started
    return_code: int | str = "timeout" if timed_out else int(process.returncode)
    log = (
        f"command: {subprocess.list2cmdline(command)}\n"
        f"started_at_utc: {started_at}\n"
        f"elapsed_seconds: {elapsed}\n"
        f"return_code: {return_code}\n\n"
        f"[stdout]\n{stdout}\n[stderr]\n{stderr}"
    )
    log_path = cell_dir / f"{name}.log"
    log_path.write_text(sanitized(log, replacements), encoding="utf-8", newline="\n")
    result = {
        "stage": name,
        "started_at_utc": started_at,
        "elapsed_seconds": elapsed,
        "return_code": return_code,
        "timed_out": timed_out,
        "log_filename": log_path.name,
        "log_sha256": sha256_file(log_path),
        "passed": not timed_out and process.returncode == 0,
    }
    write_json(cell_dir / f"{name}.json", result)
    return result


def conda_prefix(conda: str, environment: str) -> list[str]:
    return [conda, "run", "--no-capture-output", "-n", environment]


def build_command(
    conda: str,
    environment: str,
    model_snapshot: Path,
    output_dir: Path,
    hidden_states: bool,
) -> list[str]:
    command = conda_prefix(conda, environment) + [
        "python",
        "-m",
        "onnxruntime_genai.models.builder",
        "--input",
        str(model_snapshot),
        "--output",
        str(output_dir),
        "--precision",
        "fp16",
        "--execution_provider",
        "dml",
    ]
    if hidden_states:
        command.extend(["--extra_options", "include_hidden_states=true"])
    return command


def hybrid_command(
    conda: str,
    environment: str,
    sdk_version: str,
    input_dir: Path,
    output_dir: Path,
) -> list[str]:
    prefix = conda_prefix(conda, environment) + ["model_generate"]
    if sdk_version == "1.7.1":
        return prefix + ["--hybrid", str(output_dir), str(input_dir)]
    return prefix + [
        "--hybrid",
        "--eager",
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
    ]


def inspect_onnx_outputs(model_path: Path) -> list[dict[str, Any]]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    return [
        {
            "name": output.name,
            "element_type": int(output.type.tensor_type.elem_type),
            "shape": [
                dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
                for dimension in output.type.tensor_type.shape.dim
            ],
        }
        for output in model.graph.output
    ]


def artifact_files(path: Path) -> list[dict[str, Any]]:
    names = ("model.onnx", "model.onnx.data", "genai_config.json")
    return [
        {
            "filename": name,
            "bytes": (path / name).stat().st_size,
            "sha256": sha256_file(path / name),
        }
        for name in names
        if (path / name).is_file()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3 OGA compatibility by SDK version")
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=90 * 60)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    conda = shutil.which("conda")
    if conda is None:
        raise FileNotFoundError("conda executable not found")
    snapshot = args.model_snapshot.resolve()
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"model snapshot is missing config.json: {snapshot}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replacements = {
        str(REPO_ROOT.resolve()): "<REPO_ROOT>",
        str(snapshot): "<MODEL_SNAPSHOT>",
        str(Path.home().resolve()): "<USER_HOME>",
        str(Path(conda).resolve().parents[1]): "<CONDA_ROOT>",
    }
    cells: list[dict[str, Any]] = []
    for sdk_version, environment in SDK_ENVS.items():
        for hidden_states in HIDDEN_STATES:
            cell_id = f"rai-{sdk_version}-hidden-{'on' if hidden_states else 'off'}"
            cell_dir = args.output_dir / cell_id
            cell_path = cell_dir / "cell.json"
            if args.resume and cell_path.is_file():
                cell = json.loads(cell_path.read_text(encoding="utf-8"))
                environment_details_path = cell_dir / "environment-details.json"
                if not environment_details_path.is_file():
                    backfill = run_stage(
                        name="environment-backfill",
                        command=conda_prefix(conda, environment)
                        + [
                            "python",
                            str(PROBE),
                            "--environment-only",
                            "--output",
                            str(environment_details_path),
                        ],
                        cell_dir=cell_dir,
                        timeout_seconds=args.timeout_seconds,
                        replacements=replacements,
                    )
                    cell["environment_backfill_stage"] = backfill
                    if environment_details_path.is_file():
                        cell["environment"] = json.loads(
                            environment_details_path.read_text(encoding="utf-8")
                        )
                    write_json(cell_path, cell)
                cells.append(cell)
                print(f"resuming: complete: {cell_id}", flush=True)
                continue
            if cell_dir.exists() and any(cell_dir.iterdir()):
                raise FileExistsError(
                    f"cell is non-empty; use a new output or --resume: {cell_dir}"
                )
            cell_dir.mkdir(parents=True, exist_ok=True)
            oga_dir = cell_dir / "oga"
            hybrid_dir = cell_dir / "hybrid"
            probe_path = cell_dir / "probe.json"
            cell: dict[str, Any] = {
                "schema_version": "1.0.0",
                "cell_id": cell_id,
                "sdk_version": sdk_version,
                "conda_environment": environment,
                "model_snapshot_revision": snapshot.name,
                "precision": "fp16",
                "oga_execution_provider": "dml",
                "include_hidden_states": hidden_states,
                "stages": [],
            }
            environment_path = cell_dir / "environment-details.json"
            environment_stage = run_stage(
                name="environment",
                command=conda_prefix(conda, environment)
                + [
                    "python",
                    str(PROBE),
                    "--environment-only",
                    "--output",
                    str(environment_path),
                ],
                cell_dir=cell_dir,
                timeout_seconds=args.timeout_seconds,
                replacements=replacements,
            )
            cell["stages"].append(environment_stage)
            if environment_path.is_file():
                cell["environment"] = json.loads(environment_path.read_text(encoding="utf-8"))
            build = run_stage(
                name="oga-build",
                command=build_command(conda, environment, snapshot, oga_dir, hidden_states),
                cell_dir=cell_dir,
                timeout_seconds=args.timeout_seconds,
                replacements=replacements,
            )
            cell["stages"].append(build)
            if build["passed"]:
                try:
                    cell["oga_artifacts"] = artifact_files(oga_dir)
                    cell["oga_outputs"] = inspect_onnx_outputs(oga_dir / "model.onnx")
                except (FileNotFoundError, OSError, ValueError) as exc:
                    cell["artifact_inspection_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    hybrid = run_stage(
                        name="hybrid-partition",
                        command=hybrid_command(
                            conda, environment, sdk_version, oga_dir, hybrid_dir
                        ),
                        cell_dir=cell_dir,
                        timeout_seconds=args.timeout_seconds,
                        replacements=replacements,
                    )
                    cell["stages"].append(hybrid)
                    if hybrid["passed"]:
                        probe_command = conda_prefix(conda, environment) + [
                            "python",
                            str(PROBE),
                            "--model",
                            str(hybrid_dir),
                            "--output",
                            str(probe_path),
                        ]
                        if hidden_states:
                            probe_command.append("--expect-hidden")
                        probe = run_stage(
                            name="hybrid-probe",
                            command=probe_command,
                            cell_dir=cell_dir,
                            timeout_seconds=args.timeout_seconds,
                            replacements=replacements,
                        )
                        cell["stages"].append(probe)
                        if probe_path.is_file():
                            cell["probe"] = json.loads(probe_path.read_text(encoding="utf-8"))
            cell["passed"] = bool(
                len(cell["stages"]) == 4
                and all(stage["passed"] for stage in cell["stages"])
                and cell.get("probe", {}).get("probe", {}).get("passed")
            )
            cell["recorded_at_utc"] = utc_now()
            write_json(cell_path, cell)
            cells.append(cell)
            write_json(
                args.output_dir / "matrix.json",
                {
                    "schema_version": "1.0.0",
                    "recorded_at_utc": utc_now(),
                    "model_snapshot_revision": snapshot.name,
                    "cells": cells,
                },
            )
    write_json(
        args.output_dir / "matrix.json",
        {
            "schema_version": "1.0.0",
            "recorded_at_utc": utc_now(),
            "model_snapshot_revision": snapshot.name,
            "cells": cells,
        },
    )
    print(f"decoder compatibility matrix complete: {args.output_dir}")


if __name__ == "__main__":
    main()
