"""Build fixed-shape Nomic artifacts for validated and exploratory studies."""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER = Path(__file__).parent / "export_nomic_embed.py"
QUANTIZER = Path(__file__).parent / "quantize_nomic.py"
MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
BATCH_SIZES = (1, 8, 32)
SEQUENCE_LENGTHS = (32, 128, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the fixed-shape Nomic model matrix")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--include-rejected-int8-candidates",
        action="store_true",
        help="Also build per-channel dynamic INT8 artifacts rejected by the b1/s128 gate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for batch_size, seq_len in product(BATCH_SIZES, SEQUENCE_LENGTHS):
        fp32_path = args.output_dir / f"nomic-embed-v1.5_b{batch_size}_seq{seq_len}.onnx"
        int8_path = fp32_path.with_name(f"{fp32_path.stem}_int8_per_channel.onnx")
        if not (args.skip_existing and fp32_path.exists()):
            subprocess.run(
                [
                    sys.executable,
                    str(EXPORTER),
                    "--batch-size",
                    str(batch_size),
                    "--seq-len",
                    str(seq_len),
                    "--output-dir",
                    str(args.output_dir),
                    "--model-revision",
                    args.model_revision,
                ],
                cwd=REPO_ROOT,
                check=True,
            )
        if args.include_rejected_int8_candidates and not (
            args.skip_existing and int8_path.exists()
        ):
            subprocess.run(
                [
                    sys.executable,
                    str(QUANTIZER),
                    "--model",
                    str(fp32_path),
                    "--output",
                    str(int8_path),
                    "--model-revision",
                    args.model_revision,
                    "--per-channel",
                ],
                cwd=REPO_ROOT,
                check=True,
            )


if __name__ == "__main__":
    main()
