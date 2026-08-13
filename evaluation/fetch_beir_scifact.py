"""Materialize the checksum-pinned BEIR SciFact test split as canonical JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npu_study.artifacts import sha256_file, utc_now, write_json  # noqa: E402

DEFAULT_SPEC = Path(__file__).parent / "datasets" / "beir-scifact.json"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "beir-scifact"
ARCHIVE_MEMBERS = {
    "corpus": "scifact/corpus.jsonl",
    "queries": "scifact/queries.jsonl",
    "qrels": "scifact/qrels/test.tsv",
}


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def jsonl_member(archive: zipfile.ZipFile, member: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {member}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object in {member}:{line_number}")
            rows.append(row)
    return rows


def canonical_rows(archive_path: Path) -> dict[str, list[dict[str, Any]]]:
    with zipfile.ZipFile(archive_path) as archive:
        missing = set(ARCHIVE_MEMBERS.values()) - set(archive.namelist())
        if missing:
            raise ValueError(f"archive is missing required members: {sorted(missing)}")
        corpus_source = jsonl_member(archive, ARCHIVE_MEMBERS["corpus"])
        queries_source = jsonl_member(archive, ARCHIVE_MEMBERS["queries"])
        with (
            archive.open(ARCHIVE_MEMBERS["qrels"]) as raw,
            io.TextIOWrapper(raw, encoding="utf-8") as handle,
        ):
            qrel_source = list(csv.DictReader(handle, delimiter="\t"))

    qrels = [
        {
            "query_id": str(row["query-id"]),
            "document_id": str(row["corpus-id"]),
            "score": float(row["score"]),
        }
        for row in qrel_source
    ]
    test_query_ids = {row["query_id"] for row in qrels}
    corpus = [
        {
            "id": str(row["_id"]),
            "title": str(row.get("title", "")),
            "text": str(row["text"]),
        }
        for row in corpus_source
    ]
    queries = [
        {"id": str(row["_id"]), "text": str(row["text"])}
        for row in queries_source
        if str(row["_id"]) in test_query_ids
    ]
    corpus.sort(key=lambda row: row["id"])
    queries.sort(key=lambda row: row["id"])
    qrels.sort(key=lambda row: (row["query_id"], row["document_id"], row["score"]))
    return {"corpus": corpus, "queries": queries, "qrels": qrels}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def materialize(spec_path: Path, output_dir: Path, archive_path: Path | None) -> None:
    with spec_path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    source_archive = archive_path or output_dir.parent / "beir-scifact.zip"
    if not source_archive.exists():
        download(spec["url"], source_archive)
    for algorithm in ("md5", "sha256"):
        actual = digest_file(source_archive, algorithm)
        expected = spec[f"archive_{algorithm}"]
        if actual != expected:
            raise ValueError(
                f"archive {algorithm.upper()} mismatch: expected {expected}, received {actual}"
            )

    outputs: dict[str, dict[str, Any]] = {}
    for name, rows in canonical_rows(source_archive).items():
        expected_rows = int(spec["expected_rows"][name])
        if len(rows) != expected_rows:
            raise ValueError(f"{name}: expected {expected_rows} rows, received {len(rows)}")
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        checksum = sha256_file(path)
        expected_checksum = spec["canonical_sha256"][name]
        if expected_checksum != "PENDING" and checksum != expected_checksum:
            raise ValueError(
                f"{name}: canonical SHA-256 mismatch; "
                f"expected {expected_checksum}, received {checksum}"
            )
        outputs[name] = {"filename": path.name, "rows": len(rows), "sha256": checksum}

    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "1.0.0",
            "materialized_at_utc": utc_now(),
            "source": spec,
            "archive": {
                "filename": source_archive.name,
                "md5": digest_file(source_archive, "md5"),
                "sha256": sha256_file(source_archive),
            },
            "canonical_outputs": outputs,
        },
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))
    print(f"materialized pinned corpus: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch the pinned BEIR SciFact test split")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    materialize(args.spec, args.output_dir, args.archive)
