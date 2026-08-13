"""Validate benchmark fixtures and publication results against the JSON schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "benchmark-result.schema.json"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "benchmark-result.json"
PUBLISHED_ROOT = REPO_ROOT / "benchmarks" / "results" / "published"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_publication_manifest(path: Path) -> list[str]:
    path = path.resolve()
    manifest = load_json(path)
    root = path.parent
    failures: list[str] = []
    declared: set[str] = set()
    for entry in manifest.get("published_files", []):
        name = entry.get("path")
        if not isinstance(name, str) or not name or Path(name).name != name:
            failures.append(f"{path}: invalid published file path: {name!r}")
            continue
        if name in declared:
            failures.append(f"{path}: duplicate published file: {name}")
            continue
        declared.add(name)
        candidate = (root / name).resolve()
        if candidate.parent != root or not candidate.is_file():
            failures.append(f"{path}: published file is missing or escapes snapshot: {name}")
            continue
        if candidate.stat().st_size != entry.get("bytes"):
            failures.append(f"{path}: size mismatch for {name}")
        if sha256_file(candidate) != entry.get("sha256"):
            failures.append(f"{path}: SHA-256 mismatch for {name}")
    actual = {item.name for item in root.iterdir() if item.is_file() and item != path}
    if actual != declared:
        failures.append(
            f"{path}: manifest file set differs: missing={sorted(actual - declared)}, "
            f"extra={sorted(declared - actual)}"
        )
    return failures


def main() -> None:
    schema = load_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    paths = [FIXTURE_PATH]
    if PUBLISHED_ROOT.exists():
        paths.extend(sorted(PUBLISHED_ROOT.rglob("result.json")))
    failures = []
    for path in paths:
        errors = sorted(validator.iter_errors(load_json(path)), key=lambda item: list(item.path))
        for error in errors:
            location = ".".join(str(part) for part in error.path) or "<root>"
            failures.append(f"{path}:{location}: {error.message}")
    manifests = sorted(PUBLISHED_ROOT.glob("*/manifest.json")) if PUBLISHED_ROOT.exists() else []
    for path in manifests:
        failures.extend(validate_publication_manifest(path))
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"validated {len(paths)} benchmark result file(s) and "
        f"{len(manifests)} publication manifest(s)"
    )


if __name__ == "__main__":
    main()
