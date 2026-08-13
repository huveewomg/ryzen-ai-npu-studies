"""Build the predeclared BGE generalization corners."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from export_bert_embed import BGE_MODEL_ID, BGE_REVISION

EXPORTER = Path(__file__).parent / "export_bert_embed.py"
CORNERS = ((1, 32), (32, 128), (1, 512))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BGE generalization ONNX artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for batch_size, seq_len in CORNERS:
        output = args.output_dir / f"bge-small-en-v1.5_b{batch_size}_seq{seq_len}.onnx"
        if (
            args.skip_existing
            and output.is_file()
            and output.with_suffix(".metadata.json").is_file()
        ):
            print(f"already built: {output}")
            continue
        subprocess.run(
            [
                sys.executable,
                str(EXPORTER),
                "--model-id",
                BGE_MODEL_ID,
                "--model-revision",
                BGE_REVISION,
                "--pooling",
                "cls",
                "--batch-size",
                str(batch_size),
                "--seq-len",
                str(seq_len),
                "--output-dir",
                str(args.output_dir),
                "--output-stem",
                "bge-small-en-v1.5",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
