"""Materialize the pinned NanoSciFact snapshot as canonical JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npu_study.artifacts import sha256_file, utc_now, write_json  # noqa: E402

DEFAULT_SPEC = Path(__file__).parent / "datasets" / "nanoscifact.json"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "nanoscifact"


def canonical_row(config: str, row: dict[str, Any]) -> dict[str, Any]:
    if config == "corpus":
        return {"id": str(row["_id"]), "text": str(row["text"])}
    if config == "queries":
        return {"id": str(row["_id"]), "text": str(row["text"])}
    if config == "qrels":
        return {
            "query_id": str(row["query-id"]),
            "document_id": str(row["corpus-id"]),
            "score": float(row.get("score", 1.0)),
        }
    raise ValueError(f"unsupported config: {config}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def materialize(spec_path: Path, output_dir: Path) -> None:
    from datasets import load_dataset

    with spec_path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    outputs: dict[str, dict[str, Any]] = {}
    for config in ("corpus", "queries", "qrels"):
        dataset = load_dataset(
            spec["dataset"],
            config,
            split=spec["split"],
            revision=spec["revision"],
        )
        rows = [canonical_row(config, dict(row)) for row in dataset]
        rows.sort(key=lambda row: tuple(str(value) for value in row.values()))
        expected = int(spec["expected_rows"][config])
        if len(rows) != expected:
            raise RuntimeError(f"{config}: expected {expected} rows, received {len(rows)}")
        path = output_dir / f"{config}.jsonl"
        write_jsonl(path, rows)
        checksum = sha256_file(path)
        expected_checksum = spec["canonical_sha256"][config]
        if checksum != expected_checksum:
            raise RuntimeError(
                f"{config}: canonical SHA-256 mismatch; "
                f"expected {expected_checksum}, received {checksum}"
            )
        outputs[config] = {
            "filename": path.name,
            "rows": len(rows),
            "sha256": checksum,
        }

    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "1.0.0",
            "materialized_at_utc": utc_now(),
            "source": spec,
            "canonical_outputs": outputs,
        },
    )
    print(f"materialized pinned corpus: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the pinned NanoSciFact corpus")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    materialize(args.spec, args.output_dir)
