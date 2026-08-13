"""Run the four-way Nomic CPU/NPU smoke matrix.

This compatibility wrapper delegates to the publication benchmark harness.
NPU runs always use ``--require-npu`` and therefore fail non-zero when provider
or operator-assignment evidence is missing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "benchmarks" / "run_benchmark.py"
FP32_MODEL = Path(__file__).parent / "nomic-embed-v1.5_b1_seq128.onnx"
INT8_MODEL = Path(__file__).parent / "nomic-embed-v1.5_b1_seq128_int8.onnx"


def run_configuration(
    *,
    model: Path,
    precision: str,
    provider: str,
    args: argparse.Namespace,
) -> None:
    if not model.exists():
        raise FileNotFoundError(f"required model is missing: {model}")
    command = [
        sys.executable,
        str(BENCHMARK),
        "--model",
        str(model),
        "--precision",
        precision,
        "--provider",
        provider,
        "--batch-size",
        "1",
        "--seq-len",
        "128",
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--mode",
        args.mode,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.synthetic_input:
        command.append("--synthetic-input")
    else:
        command.extend(["--texts-file", str(args.texts_file)])
    if provider == "npu":
        command.extend(["--require-npu", "--cache-dir", str(args.cache_dir)])
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nomic FP32/INT8 CPU/NPU smoke matrix")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--texts-file", type=Path)
    parser.add_argument("--synthetic-input", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--mode", choices=["inference", "end-to-end", "both"], default="both")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "local",
    )
    args = parser.parse_args()
    if not args.synthetic_input and args.texts_file is None:
        parser.error("pass --texts-file or explicitly choose --synthetic-input")
    if args.synthetic_input and args.texts_file is not None:
        parser.error("--synthetic-input and --texts-file are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    for precision, model in (("fp32", FP32_MODEL), ("dynamic-int8", INT8_MODEL)):
        for provider in ("cpu", "npu"):
            run_configuration(
                model=model,
                precision=precision,
                provider=provider,
                args=args,
            )


if __name__ == "__main__":
    main()
