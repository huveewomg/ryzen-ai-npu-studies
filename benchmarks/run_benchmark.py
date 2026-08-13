"""Reproducible CPU/VitisAI benchmark harness for embedding ONNX models.

NPU-labelled publication runs must pass ``--require-npu``. That mode exits
non-zero unless VitisAI is available, active in the session, and emits an
operator-assignment report containing at least one NPU-assigned node.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npu_study.artifacts import (  # noqa: E402
    environment_manifest,
    sha256_file,
    utc_now,
    write_json,
)
from npu_study.evidence import (  # noqa: E402
    DEFAULT_REPORT_NAME,
    VITIS_PROVIDER,
    NPUVerificationError,
    configure_assignment_report,
    locate_assignment_report,
    parse_assignment_report,
    provider_options,
    require_available_provider,
    require_session_provider,
)

WARMUP_ITERATIONS = 20
BENCHMARK_ITERATIONS = 100
SEQUENCE_LENGTH = 128
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"

PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "npu": VITIS_PROVIDER,
}


@dataclass
class LatencySummary:
    latencies_ms: list[float] = field(default_factory=list)
    mean_ms: float = 0.0
    standard_deviation_ms: float = 0.0
    median_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput_docs_per_second: float = 0.0


def summarise(latencies: list[float], batch_size: int) -> LatencySummary:
    values = np.asarray(latencies, dtype=np.float64)
    return LatencySummary(
        latencies_ms=latencies,
        mean_ms=float(np.mean(values)),
        standard_deviation_ms=float(np.std(values)),
        median_ms=float(np.median(values)),
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        throughput_docs_per_second=batch_size * 1000.0 / float(np.mean(values)),
    )


def load_texts(path: Path, prefix: str = "") -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if path.suffix.lower() == ".jsonl":
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                    raise ValueError(f"{path}:{line_number} must contain a string text field")
                texts.append(prefix + value["text"])
            else:
                texts.append(prefix + line)
    if not texts:
        raise ValueError(f"no texts found in {path}")
    return texts


def create_tokenizer(model_id: str, revision: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def encode_batch(tokenizer, texts: list[str], batch_size: int, seq_len: int, index: int):
    selected = [texts[(index * batch_size + offset) % len(texts)] for offset in range(batch_size)]
    encoded = tokenizer(
        selected,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }


def synthetic_batch(session, batch_size: int, seq_len: int) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        shape = [dimension if isinstance(dimension, int) else 1 for dimension in item.shape]
        if len(shape) == 2:
            shape = [batch_size, seq_len]
        dtype = np.float32 if item.type == "tensor(float)" else np.int64
        values[item.name] = np.ones(shape, dtype=dtype)
    return values


def validate_model_shape(session, batch_size: int, seq_len: int) -> None:
    for item in session.get_inputs():
        if len(item.shape) != 2:
            continue
        expected_batch, expected_sequence = item.shape
        if isinstance(expected_batch, int) and expected_batch != batch_size:
            raise ValueError(
                f"model input {item.name} has fixed batch {expected_batch}, requested {batch_size}"
            )
        if isinstance(expected_sequence, int) and expected_sequence != seq_len:
            raise ValueError(
                f"model input {item.name} has fixed sequence {expected_sequence}, requested {seq_len}"
            )


def validate_model_metadata(
    model_path: Path,
    *,
    precision: str,
    model_revision: str,
    quantization_granularity: str,
    required: bool,
) -> tuple[Path | None, str | None]:
    metadata_path = model_path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        if required:
            raise ValueError(f"required model metadata is missing: {metadata_path}")
        return None, None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    model_sha256 = sha256_file(model_path)
    if precision == "fp32":
        recorded_hash = metadata.get("onnx", {}).get("sha256")
        recorded_revision = metadata.get("source_revision")
    else:
        recorded_hash = metadata.get("output_model", {}).get("sha256")
        recorded_revision = metadata.get("source_model", {}).get("huggingface_revision")
        if metadata.get("per_channel") is not (quantization_granularity == "per-channel"):
            raise ValueError("quantization granularity does not match model metadata")
        if not metadata.get("acceptance", {}).get("passed"):
            raise ValueError("quantized model metadata records a failed smoke-fidelity gate")
    if recorded_hash != model_sha256:
        raise ValueError("model SHA-256 does not match its metadata")
    if recorded_revision != model_revision:
        raise ValueError("model revision does not match its metadata")
    return metadata_path, sha256_file(metadata_path)


def create_session(
    model_path: Path,
    provider: str,
    cache_dir: Path | None,
    require_npu: bool,
    intra_op_threads: int = 0,
    inter_op_threads: int = 0,
):
    import onnxruntime as ort

    available = ort.get_available_providers()
    options = None
    report_root = None
    cache_key = None
    cache_present_before_session = False
    if provider == "npu":
        if cache_dir is None:
            raise ValueError("--cache-dir is required for NPU runs")
        if require_npu:
            require_available_provider(available)
            configure_assignment_report(DEFAULT_REPORT_NAME)
        options_dict = provider_options(model_path, cache_dir)
        options = [options_dict]
        cache_key = options_dict["cache_key"]
        report_root = cache_dir / options_dict["cache_key"]
        cache_present_before_session = report_root.exists() and any(report_root.rglob("*"))

    session_options = ort.SessionOptions()
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if intra_op_threads:
        session_options.intra_op_num_threads = intra_op_threads
    if inter_op_threads:
        session_options.inter_op_num_threads = inter_op_threads
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=[PROVIDERS[provider]],
        provider_options=options,
    )
    creation_ms = (time.perf_counter() - started) * 1000.0
    active = session.get_providers()
    if provider == "npu" and require_npu:
        require_session_provider(active)
    return (
        session,
        available,
        active,
        report_root,
        {
            "session_creation_ms": creation_ms,
            "cache_key": cache_key,
            "cache_present_before_session": cache_present_before_session,
        },
    )


def measure_inference(
    session, batches: list[dict[str, np.ndarray]], iterations: int
) -> list[float]:
    latencies: list[float] = []
    for index in range(iterations):
        started = time.perf_counter()
        session.run(None, batches[index % len(batches)])
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def measure_end_to_end(
    session, tokenizer, texts: list[str], batch_size: int, seq_len: int, iterations: int
) -> list[float]:
    latencies: list[float] = []
    for index in range(iterations):
        started = time.perf_counter()
        inputs = encode_batch(tokenizer, texts, batch_size, seq_len, index)
        session.run(None, inputs)
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def manifest_extra(args: argparse.Namespace) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if args.manifest_input:
        with args.manifest_input.open("r", encoding="utf-8") as handle:
            extra = json.load(handle)
    controls = extra.setdefault("controls", {})
    controls["thread_settings"] = {
        "onnxruntime_execution_mode": "ORT_SEQUENTIAL",
        "onnxruntime_intra_op_num_threads": args.intra_op_threads,
        "onnxruntime_inter_op_num_threads": args.inter_op_threads,
    }
    return extra


def run_one(args: argparse.Namespace, provider: str) -> Path:
    model_path = args.model.resolve()
    run_id = args.run_id or (
        f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{provider}-"
        f"{args.precision}-b{args.batch_size}-s{args.seq_len}-r{args.repeat}"
    )
    run_dir = args.output_dir / run_id
    metadata_path, metadata_sha256 = validate_model_metadata(
        model_path,
        precision=args.precision,
        model_revision=args.model_revision,
        quantization_granularity=args.quantization_granularity,
        required=args.require_model_metadata,
    )
    session, available, active, report_root, session_info = create_session(
        model_path,
        provider,
        args.cache_dir,
        args.require_npu,
        args.intra_op_threads,
        args.inter_op_threads,
    )
    validate_model_shape(session, args.batch_size, args.seq_len)

    tokenizer = None
    texts = None
    if args.synthetic_input:
        batches = [synthetic_batch(session, args.batch_size, args.seq_len)]
        input_kind = "fixed-shape synthetic microbenchmark"
    else:
        texts = load_texts(args.texts_file, args.text_prefix)
        tokenizer = create_tokenizer(args.tokenizer, args.model_revision)
        batches = [
            encode_batch(tokenizer, texts, args.batch_size, args.seq_len, index)
            for index in range(max(args.warmup, args.iterations))
        ]
        input_kind = "tokenizer-backed text"

    for index in range(args.warmup):
        session.run(None, batches[index % len(batches)])

    assignment = None
    report_path = None
    if provider == "npu" and args.require_npu:
        search_root = report_root if report_root and report_root.exists() else args.cache_dir
        report_path = locate_assignment_report(search_root)
        try:
            assignment = parse_assignment_report(report_path)
        except NPUVerificationError as exc:
            failure = {
                "schema_version": "1.0.0",
                "status": "failed",
                "run_id": run_id,
                "recorded_at_utc": utc_now(),
                "stage": "npu_assignment_verification",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "model": {
                    "filename": model_path.name,
                    "sha256": sha256_file(model_path),
                    "metadata_filename": metadata_path.name if metadata_path else None,
                    "metadata_sha256": metadata_sha256,
                    "precision": args.precision,
                    "quantization_granularity": args.quantization_granularity,
                },
                "execution": {
                    "available_providers": available,
                    "session_providers": active,
                    "session_options": {
                        "execution_mode": "ORT_SEQUENTIAL",
                        "intra_op_num_threads": args.intra_op_threads,
                        "inter_op_num_threads": args.inter_op_threads,
                    },
                    "npu_cache": {
                        "cache_key": session_info["cache_key"],
                        "present_before_session": session_info["cache_present_before_session"],
                    },
                    "session_creation_ms": session_info["session_creation_ms"],
                    "assignment_report": {
                        "filename": report_path.name,
                        "sha256": sha256_file(report_path),
                    },
                },
            }
            write_json(run_dir / "failure.json", failure)
            write_json(
                run_dir / "environment.json",
                environment_manifest(
                    repo_root=REPO_ROOT,
                    available_providers=available,
                    extra=manifest_extra(args),
                ),
            )
            shutil.copy2(report_path, run_dir / DEFAULT_REPORT_NAME)
            raise

    inference = summarise(measure_inference(session, batches, args.iterations), args.batch_size)
    end_to_end = None
    if args.mode in {"end-to-end", "both"}:
        if tokenizer is None or texts is None:
            raise ValueError("end-to-end mode requires --texts-file, not --synthetic-input")
        end_to_end = summarise(
            measure_end_to_end(
                session, tokenizer, texts, args.batch_size, args.seq_len, args.iterations
            ),
            args.batch_size,
        )

    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "recorded_at_utc": utc_now(),
        "model": {
            "filename": model_path.name,
            "sha256": sha256_file(model_path),
            "source_model": args.tokenizer,
            "source_revision": args.model_revision,
            "precision": args.precision,
            "quantization_granularity": args.quantization_granularity,
            "metadata_filename": metadata_path.name if metadata_path else None,
            "metadata_sha256": metadata_sha256,
        },
        "execution": {
            "requested": provider,
            "available_providers": available,
            "session_providers": active,
            "require_npu": bool(provider == "npu" and args.require_npu),
            "npu_verified": bool(assignment),
            "assignment": assignment.to_dict() if assignment else None,
            "session_options": {
                "execution_mode": "ORT_SEQUENTIAL",
                "intra_op_num_threads": args.intra_op_threads,
                "inter_op_num_threads": args.inter_op_threads,
            },
            "npu_cache": (
                {
                    "cache_key": session_info["cache_key"],
                    "present_before_session": session_info["cache_present_before_session"],
                }
                if provider == "npu"
                else None
            ),
        },
        "workload": {
            "batch_size": args.batch_size,
            "sequence_length": args.seq_len,
            "warmup_iterations": args.warmup,
            "timed_iterations": args.iterations,
            "repeat": args.repeat,
            "input_kind": input_kind,
            "input_file": args.texts_file.name if args.texts_file else None,
            "input_sha256": sha256_file(args.texts_file) if args.texts_file else None,
            "text_prefix": args.text_prefix,
        },
        "metrics": {
            "session_creation_ms": session_info["session_creation_ms"],
            "inference_only": asdict(inference),
            "end_to_end": asdict(end_to_end) if end_to_end else None,
        },
    }
    write_json(run_dir / "result.json", result)
    write_json(
        run_dir / "environment.json",
        environment_manifest(
            repo_root=REPO_ROOT,
            available_providers=available,
            extra=manifest_extra(args),
        ),
    )
    if report_path:
        shutil.copy2(report_path, run_dir / DEFAULT_REPORT_NAME)

    print(
        f"{provider}: {inference.mean_ms:.2f} ms, {inference.throughput_docs_per_second:.2f} docs/s"
    )
    print(f"artifacts: {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible NPU embedding benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", choices=["cpu", "npu", "all"], default="all")
    parser.add_argument("--precision", choices=["fp32", "dynamic-int8"], required=True)
    parser.add_argument(
        "--quantization-granularity",
        choices=["none", "per-tensor", "per-channel"],
        default="none",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--require-npu", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--texts-file", type=Path)
    parser.add_argument("--text-prefix", default="")
    parser.add_argument("--synthetic-input", action="store_true")
    parser.add_argument("--tokenizer", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--mode", choices=["inference", "end-to-end", "both"], default="both")
    parser.add_argument("--manifest-input", type=Path)
    parser.add_argument("--require-model-metadata", action="store_true")
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--inter-op-threads", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "benchmarks" / "results" / "local"
    )
    args = parser.parse_args()
    if not args.synthetic_input and args.texts_file is None:
        parser.error(
            "pass --texts-file for real-text runs or --synthetic-input for a microbenchmark"
        )
    if args.synthetic_input and args.texts_file is not None:
        parser.error("--synthetic-input and --texts-file are mutually exclusive")
    if args.provider in {"npu", "all"} and not args.require_npu:
        parser.error("NPU-labelled runs require --require-npu")
    if args.provider == "cpu" and args.require_npu:
        parser.error("--require-npu cannot be used with --provider cpu")
    if args.batch_size < 1 or args.seq_len < 1 or args.warmup < 1 or args.iterations < 1:
        parser.error("batch size, sequence length, warmups, and iterations must be positive")
    if args.intra_op_threads < 0 or args.inter_op_threads < 0:
        parser.error("thread counts must be zero (ORT default) or positive")
    if args.precision == "fp32" and args.quantization_granularity != "none":
        parser.error("FP32 runs require --quantization-granularity none")
    if args.precision == "dynamic-int8" and args.quantization_granularity == "none":
        parser.error("dynamic INT8 runs must declare per-tensor or per-channel granularity")
    return args


def main() -> None:
    args = parse_args()
    selected = list(PROVIDERS) if args.provider == "all" else [args.provider]
    for provider in selected:
        run_one(args, provider)


if __name__ == "__main__":
    main()
