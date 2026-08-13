"""Launch the publication Nomic ablation as independent warmed processes."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npu_study.artifacts import sha256_file, utc_now, write_json  # noqa: E402

HARNESS = Path(__file__).parent / "run_benchmark.py"
PUBLICATION_PRECISIONS = ("fp32",)
PROVIDERS = ("cpu", "npu")
SEQUENCE_LENGTHS = (32, 128, 512)
BATCH_SIZES = (1, 8, 32)
DEFAULT_PROCESS_TIMEOUT_SECONDS = 45 * 60


def model_path(models_dir: Path, precision: str, batch_size: int, seq_len: int) -> Path:
    suffix = "_int8_per_channel" if precision == "dynamic-int8" else ""
    return models_dir / f"nomic-embed-v1.5_b{batch_size}_seq{seq_len}{suffix}.onnx"


def command_for(
    args: argparse.Namespace,
    *,
    precision: str,
    provider: str,
    batch_size: int,
    seq_len: int,
    repeat: int,
    run_id: str,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(HARNESS),
        "--model",
        str(model_path(args.models_dir, precision, batch_size, seq_len)),
        "--precision",
        precision,
        "--quantization-granularity",
        "per-channel" if precision == "dynamic-int8" else "none",
        "--require-model-metadata",
        "--provider",
        provider,
        "--batch-size",
        str(batch_size),
        "--seq-len",
        str(seq_len),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--repeat",
        str(repeat),
        "--run-id",
        run_id,
        "--texts-file",
        str(args.texts_file),
        "--text-prefix",
        args.text_prefix,
        "--intra-op-threads",
        str(args.intra_op_threads),
        "--inter-op-threads",
        str(args.inter_op_threads),
        "--mode",
        "both",
        "--output-dir",
        str(output_dir),
    ]
    if args.manifest_input:
        command.extend(["--manifest-input", str(args.manifest_input)])
    if provider == "npu":
        command.extend(["--require-npu", "--cache-dir", str(args.cache_dir)])
    return command


def sanitized_process_log(
    *,
    command: list[str],
    stdout: str,
    stderr: str,
    returncode: int | str,
    args: argparse.Namespace,
) -> str:
    content = (
        f"command: {subprocess.list2cmdline(command)}\n"
        f"return_code: {returncode}\n\n"
        f"[stdout]\n{stdout}\n[stderr]\n{stderr}"
    )
    replacements = {
        str(REPO_ROOT.resolve()): "<REPO_ROOT>",
        str(args.cache_dir.expanduser().resolve()): "<NPU_CACHE_DIR>",
        str(Path.home().resolve()): "<USER_HOME>",
    }
    for value, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        content = re.sub(re.escape(value), replacement, content, flags=re.IGNORECASE)
    return content


def timeout_text(value: str | bytes | None) -> str:
    """Normalize TimeoutExpired output across Python and platform variants."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def cell_id(precision: str, provider: str, batch_size: int, seq_len: int) -> str:
    return f"{provider}-{precision}-b{batch_size}-s{seq_len}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged Nomic ablation matrix")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--texts-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest-input", type=Path)
    parser.add_argument("--processes-per-cell", type=int, default=5)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
        help="Provider subset to include in this recorded study",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        choices=BATCH_SIZES,
        default=list(BATCH_SIZES),
        help="Batch-size subset to include in this recorded study",
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        choices=SEQUENCE_LENGTHS,
        default=list(SEQUENCE_LENGTHS),
        help="Sequence-length subset to include in this recorded study",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--intra-op-threads", type=int, default=8)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    parser.add_argument("--text-prefix", default="search_query: ")
    parser.add_argument(
        "--process-timeout-seconds",
        type=float,
        default=DEFAULT_PROCESS_TIMEOUT_SECONDS,
        help="Terminate one benchmark process after this wall-clock duration",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "local",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--study-id")
    args = parser.parse_args()
    if args.processes_per_cell < 5:
        parser.error("publication matrix requires at least five independent processes per cell")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds cannot be negative")
    if args.intra_op_threads < 1 or args.inter_op_threads < 1:
        parser.error("publication matrix thread counts must be positive")
    if args.process_timeout_seconds <= 0:
        parser.error("--process-timeout-seconds must be positive")
    return args


def build_schedule(
    processes_per_cell: int,
    seed: int,
    precisions: tuple[str, ...] = PUBLICATION_PRECISIONS,
    providers: tuple[str, ...] = PROVIDERS,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    sequence_lengths: tuple[int, ...] = SEQUENCE_LENGTHS,
) -> list[tuple[str, str, int, int, int]]:
    schedule = [
        (precision, provider, batch_size, seq_len, repeat)
        for precision, provider, batch_size, seq_len in product(
            precisions, providers, batch_sizes, sequence_lengths
        )
        for repeat in range(1, processes_per_cell + 1)
    ]
    random.Random(seed).shuffle(schedule)
    return schedule


def main() -> None:
    args = parse_args()
    study_id = args.study_id or time.strftime("%Y%m%dT%H%M%S")
    study_dir = args.output_dir / study_id
    providers = tuple(args.providers)
    batch_sizes = tuple(args.batch_sizes)
    sequence_lengths = tuple(args.sequence_lengths)
    cells = list(product(PUBLICATION_PRECISIONS, providers, batch_sizes, sequence_lengths))
    missing = sorted(
        {
            model_path(args.models_dir, precision, batch_size, seq_len)
            for precision, _, batch_size, seq_len in cells
            if not model_path(args.models_dir, precision, batch_size, seq_len).exists()
        }
    )
    if missing and not args.dry_run:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"matrix model artifacts are missing:\n{formatted}")

    schedule = build_schedule(
        args.processes_per_cell,
        args.seed,
        providers=providers,
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
    )
    study = {
        "schema_version": "1.0.0",
        "study_id": study_id,
        "status": "running",
        "created_at_utc": utc_now(),
        "seed": args.seed,
        "cooldown_seconds": args.cooldown_seconds,
        "processes_per_cell": args.processes_per_cell,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "process_timeout_seconds": args.process_timeout_seconds,
        "cell_scope": {
            "providers": list(providers),
            "batch_sizes": list(batch_sizes),
            "sequence_lengths": list(sequence_lengths),
        },
        "precision_scope": {
            "included": list(PUBLICATION_PRECISIONS),
            "excluded": {
                "dynamic-int8": (
                    "b1/s128 per-tensor failed the predeclared smoke gate; per-channel failed "
                    "NanoSciFact fidelity and the VitisAI report assigned zero NPU nodes"
                )
            },
        },
        "session_options": {
            "execution_mode": "ORT_SEQUENTIAL",
            "intra_op_num_threads": args.intra_op_threads,
            "inter_op_num_threads": args.inter_op_threads,
        },
        "input": {
            "filename": args.texts_file.name,
            "sha256": sha256_file(args.texts_file),
            "text_prefix": args.text_prefix,
        },
        "schedule": [
            {
                "ordinal": ordinal,
                "precision": precision,
                "provider": provider,
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "repeat": repeat,
            }
            for ordinal, (precision, provider, batch_size, seq_len, repeat) in enumerate(
                schedule, start=1
            )
        ],
        "failures": [],
    }
    study_path = study_dir / "study.json"
    if not args.dry_run and study_path.exists():
        if not args.resume:
            raise FileExistsError(f"study already exists; pass --resume: {study_dir}")
        with study_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        protocol_fields = (
            "seed",
            "processes_per_cell",
            "warmup_iterations",
            "timed_iterations",
            "process_timeout_seconds",
            "cell_scope",
            "precision_scope",
            "session_options",
            "input",
            "schedule",
        )
        for key in protocol_fields:
            if key == "process_timeout_seconds" and key not in previous:
                continue
            if previous.get(key) != study[key]:
                raise ValueError(f"cannot resume: study protocol field {key!r} changed")
        study["created_at_utc"] = previous["created_at_utc"]
        study["resumed_at_utc"] = utc_now()
        study["failures"] = previous.get("failures", [])
        study["skipped_runs"] = previous.get("skipped_runs", [])
        if "process_timeout_seconds" not in previous:
            study["protocol_amendments"] = previous.get("protocol_amendments", []) + [
                {
                    "recorded_at_utc": utc_now(),
                    "field": "process_timeout_seconds",
                    "value": args.process_timeout_seconds,
                    "reason": (
                        "Added a fail-safe after observing long first-use VitisAI compilation; "
                        "all completed processes remained below the ceiling"
                    ),
                }
            ]
    if not args.dry_run:
        write_json(study_path, study)

    failures: list[str] = list(study.get("failures", []))
    skipped_runs: list[str] = list(study.get("skipped_runs", []))
    failed_cells: set[str] = set()
    if args.resume:
        for failure_path in study_dir.glob("*/failure.json"):
            with failure_path.open("r", encoding="utf-8") as handle:
                failure = json.load(handle)
            if failure.get("cell_id"):
                failed_cells.add(failure["cell_id"])

    for ordinal, (precision, provider, batch_size, seq_len, repeat) in enumerate(schedule, start=1):
        run_id = f"{study_id}-{provider}-{precision}-b{batch_size}-s{seq_len}-r{repeat}"
        current_cell = cell_id(precision, provider, batch_size, seq_len)
        command = command_for(
            args,
            precision=precision,
            provider=provider,
            batch_size=batch_size,
            seq_len=seq_len,
            repeat=repeat,
            run_id=run_id,
            output_dir=study_dir,
        )
        run_dir = study_dir / run_id
        if (
            args.resume
            and (run_dir / "result.json").is_file()
            and (run_dir / "process.log").is_file()
        ):
            print(f"resuming: already complete: {run_id}")
            continue
        if args.resume and (run_dir / "failure.json").is_file():
            print(f"resuming: terminal failure already recorded: {run_id}")
            failed_cells.add(current_cell)
            if run_id not in failures:
                failures.append(run_id)
            continue
        if current_cell in failed_cells:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                run_dir / "skipped.json",
                {
                    "schema_version": "1.0.0",
                    "study_id": study_id,
                    "run_id": run_id,
                    "cell_id": current_cell,
                    "recorded_at_utc": utc_now(),
                    "reason": "remaining repeat skipped after a terminal failure in this cell",
                },
            )
            if run_id not in skipped_runs:
                skipped_runs.append(run_id)
            study["skipped_runs"] = skipped_runs
            write_json(study_dir / "study.json", study)
            print(f"skipping: cell already failed: {run_id}")
            continue
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            continue
        if ordinal > 1 and args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)
        failure_stage: str | None = None
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=args.process_timeout_seconds,
            )
            returncode: int | str = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
            if completed.returncode:
                failure_stage = "benchmark_process"
        except subprocess.TimeoutExpired as exc:
            returncode = "timeout"
            stdout = timeout_text(exc.stdout)
            stderr = timeout_text(exc.stderr)
            failure_stage = "process_timeout"
            timeout_message = (
                f"benchmark process exceeded {args.process_timeout_seconds:g} seconds and "
                "was terminated\n"
            )
            stderr = f"{stderr}\n{timeout_message}" if stderr else timeout_message
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "process.log").write_text(
            sanitized_process_log(
                command=command,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                args=args,
            ),
            encoding="utf-8",
            newline="\n",
        )
        if failure_stage:
            write_json(
                run_dir / "failure.json",
                {
                    "schema_version": "1.0.0",
                    "study_id": study_id,
                    "run_id": run_id,
                    "cell_id": current_cell,
                    "recorded_at_utc": utc_now(),
                    "stage": failure_stage,
                    "return_code": returncode,
                    "process_timeout_seconds": args.process_timeout_seconds,
                },
            )
            failed_cells.add(current_cell)
            if run_id not in failures:
                failures.append(run_id)
            study["failures"] = failures
            study["status"] = "running_with_failures" if args.continue_on_error else "failed"
            write_json(study_dir / "study.json", study)
            if not args.continue_on_error:
                raise SystemExit(f"matrix process failed: {run_id}; inspect its process.log")

    if failures:
        study["status"] = "complete_with_failures"
        study["completed_at_utc"] = utc_now()
        study["failures"] = failures
        study["skipped_runs"] = skipped_runs
        write_json(study_dir / "study.json", study)
        print(
            f"matrix completed with {len(failures)} failed process(es); "
            f"inspect their failure.json and process.log files"
        )
        return
    if not args.dry_run:
        study["status"] = "complete"
        study["completed_at_utc"] = utc_now()
        write_json(study_dir / "study.json", study)


if __name__ == "__main__":
    main()
