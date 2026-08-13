"""Build a sanitized raw-evidence attachment for the Ryzen AI 1.8 study."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npu_study.artifacts import sha256_file  # noqa: E402

SNAPSHOT_ID = "rai180-20260813"
BUNDLE_ID = f"{SNAPSHOT_ID}-raw-evidence"
DEFAULT_OUTPUT = REPO_ROOT / "dist" / BUNDLE_ID
DEFAULT_ARCHIVE = REPO_ROOT / "dist" / f"{BUNDLE_ID}.zip"

TEXT_SUFFIXES = {".csv", ".json", ".log", ".txt"}
FORBIDDEN_SUFFIXES = {".bin", ".data", ".npz", ".onnx", ".pb", ".xclbin", ".xmodel"}
ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
WINDOWS_BACKSLASH_PATH = re.compile(r"(?i)(?:[A-Z]:\\)(?:[^\\\r\n\t<>|\"']+\\)*[^\\\r\n\t <>|\"']+")
WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?i)(?<![A-Z0-9_])(?:[A-Z]:[\\/])"
    r"(?:[^\\/\r\n\t<>|\"']+[\\/])*[^\\/\r\n\t <>|\"']+"
)
POSIX_PRIVATE_PATH = re.compile(r"(?<![\w:])/(?:home|Users|mnt|tmp)/[^\s\"'<>|]+")

SECRET_PATTERNS = {
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "GitHub token": re.compile(r"\b(?:gh[opsu]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "private key": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    "provider token": re.compile(r"\b(?:sk|hf)_[A-Za-z0-9_-]{24,}\b"),
    "credential assignment": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
        r"\s*[\"']?\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']"
    ),
}


@dataclass(frozen=True)
class SourceGroup:
    """A local evidence tree and its destination within the release bundle."""

    name: str
    source_root: Path
    release_root: Path
    include: Callable[[Path, Path], bool]


def include_text_tree(path: Path, _root: Path) -> bool:
    """Include all supported text artifacts below a source root."""

    return path.is_file() and path.suffix.lower() in TEXT_SUFFIXES


def include_top_level_qwen_evidence(path: Path, root: Path) -> bool:
    """Include matrix and per-cell records, but never generated model trees."""

    if not include_text_tree(path, root):
        return False
    return len(path.relative_to(root).parts) <= 2


def default_source_groups(repo_root: Path = REPO_ROOT) -> list[SourceGroup]:
    """Return the reviewed source set for the Ryzen AI 1.8 evidence attachment."""

    benchmark_local = repo_root / "benchmarks" / "results" / "local"
    evaluation_local = repo_root / "evaluation" / "results" / "local"
    compatibility_local = repo_root / "compatibility" / "results" / "local"
    return [
        SourceGroup(
            "nomic-matrix",
            benchmark_local / "nomic-rai180-fp32-20260813",
            Path("benchmarks/nomic-matrix"),
            include_text_tree,
        ),
        SourceGroup(
            "nomic-censored-corner",
            benchmark_local / "nomic-rai180-fp32-cpu-b32-s512-extension-20260813",
            Path("benchmarks/nomic-censored-corner"),
            include_text_tree,
        ),
        SourceGroup(
            "bge-corners",
            benchmark_local / "bge-rai180-compact-20260813",
            Path("benchmarks/bge-corners"),
            include_text_tree,
        ),
        SourceGroup(
            "dynamic-int8-benchmark-pilot",
            benchmark_local / "pilot-rai180",
            Path("quantization/benchmark-pilot"),
            include_text_tree,
        ),
        SourceGroup(
            "nomic-scifact-fidelity",
            evaluation_local / "beir-scifact-rai180-fp32",
            Path("fidelity/nomic-scifact"),
            include_text_tree,
        ),
        SourceGroup(
            "bge-scifact-fidelity",
            evaluation_local / "beir-scifact-bge-rai180-fp32",
            Path("fidelity/bge-scifact"),
            include_text_tree,
        ),
        SourceGroup(
            "dynamic-int8-fidelity-pilot",
            evaluation_local / "pilot-rai180",
            Path("quantization/fidelity-pilot"),
            include_text_tree,
        ),
        SourceGroup(
            "qwen3-compatibility",
            compatibility_local / "qwen3-rai171-rai180-20260813",
            Path("compatibility/qwen3"),
            include_top_level_qwen_evidence,
        ),
    ]


@lru_cache(maxsize=8)
def _replacement_values(repo_root: Path) -> tuple[tuple[str, str], ...]:
    replacements: list[tuple[str, str]] = []
    repo_paths = [repo_root.resolve()]
    if repo_root.is_absolute():
        repo_paths.append(repo_root)
    candidates = [
        *((str(path), "<repo-root>") for path in repo_paths),
        (str(Path.home().resolve()), "<user-home>"),
        (os.environ.get("USERPROFILE", ""), "<user-home>"),
        (platform.node(), "<host>"),
        (getpass.getuser(), "<user>"),
    ]
    seen: set[str] = set()
    for value, replacement in candidates:
        value = value.strip()
        if len(value) < 3 or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        replacements.append((value, replacement))
        alternate = value.replace("\\", "/")
        if alternate.casefold() not in seen:
            seen.add(alternate.casefold())
            replacements.append((alternate, replacement))
    return tuple(sorted(replacements, key=lambda item: len(item[0]), reverse=True))


def sanitize_text(text: str, *, repo_root: Path = REPO_ROOT) -> str:
    """Remove local identities, absolute paths, and terminal control sequences."""

    sanitized = ANSI_ESCAPE.sub("", text)
    for value, replacement in _replacement_values(repo_root):
        sanitized = re.sub(re.escape(value), replacement, sanitized, flags=re.IGNORECASE)
    sanitized = WINDOWS_BACKSLASH_PATH.sub("<absolute-path>", sanitized)
    sanitized = WINDOWS_ABSOLUTE_PATH.sub("<absolute-path>", sanitized)
    sanitized = POSIX_PRIVATE_PATH.sub("<absolute-path>", sanitized)
    return sanitized.replace("\r\n", "\n").replace("\r", "\n")


def _sanitize_json_value(value: Any, *, repo_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_json_value(item, repo_root=repo_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item, repo_root=repo_root) for item in value]
    if isinstance(value, str):
        return sanitize_text(value, repo_root=repo_root)
    return value


def sanitized_bytes(path: Path, *, repo_root: Path = REPO_ROOT) -> bytes:
    """Return a canonical, sanitized representation of a text artifact."""

    raw_text = path.read_bytes().decode("utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        parsed = json.loads(raw_text)
        sanitized = _sanitize_json_value(parsed, repo_root=repo_root)
        text = json.dumps(sanitized, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    else:
        text = sanitize_text(raw_text, repo_root=repo_root)
        if text and not text.endswith("\n"):
            text += "\n"
    return text.encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git_value(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def tracked_worktree_is_clean(repo_root: Path) -> bool:
    """Return whether tracked files match HEAD; ignored and untracked files do not count."""

    return not _git_value(repo_root, "status", "--porcelain", "--untracked-files=no")


def _safe_reset_directory(output: Path, *, allowed_root: Path) -> None:
    resolved_output = output.resolve()
    resolved_root = allowed_root.resolve()
    if resolved_output == resolved_root or not resolved_output.is_relative_to(resolved_root):
        raise ValueError(f"refusing to reset output outside {resolved_root}: {resolved_output}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)


def _readme(file_count: int) -> bytes:
    text = f"""# Ryzen AI 1.8 raw evidence

This attachment accompanies the `{SNAPSHOT_ID}` publication snapshot. It contains
{file_count} sanitized text artifacts used to generate the report: independent-process
benchmark records, environment manifests, VitisAI assignment reports, full fidelity
metrics, dynamic-INT8 rejection evidence, and the Qwen3 compatibility matrix and logs.

`manifest.json` records both the SHA-256 of each original local source and the SHA-256
and byte count of its released representation. JSON is canonicalized, line endings are
normalized to LF, ANSI terminal escapes are removed, and local identities and absolute
paths are replaced with placeholders. Run `python tools/build_raw_release.py` from the
matching clean checkout to reproduce and validate the attachment.

Deliberate exclusions:

- ONNX models, external weight data, compiled NPU caches, and tokenizer copies;
- `embeddings.npz`, because the per-item drift, retrieval metrics, and worst cases are
  already present in `fidelity.json` and the embeddings are reproducible;
- the redistributable SciFact corpus, which is fetched from its pinned upstream archive;
- unrelated local experiments.

The archive is evidence for one laptop and the exact recorded software stack. It is not
a general hardware ranking and contains no calibrated power or energy measurement.
"""
    return text.encode("utf-8")


def build_release_directory(
    *,
    repo_root: Path = REPO_ROOT,
    output: Path = DEFAULT_OUTPUT,
    groups: Iterable[SourceGroup] | None = None,
    allow_dirty: bool = False,
) -> Path:
    """Build and validate the sanitized release directory."""

    repo_root = repo_root.resolve()
    output = output.resolve()
    if not allow_dirty and not tracked_worktree_is_clean(repo_root):
        raise RuntimeError("tracked worktree is dirty; commit or use --allow-dirty for review only")

    selected_groups = list(groups if groups is not None else default_source_groups(repo_root))
    environment = repo_root / "environment.local.json"
    compact_manifest = (
        repo_root / "benchmarks" / "results" / "published" / SNAPSHOT_ID / "manifest.json"
    )
    required = [environment, compact_manifest, *(group.source_root for group in selected_groups)]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing raw-release inputs: " + ", ".join(map(str, missing)))

    _safe_reset_directory(output, allowed_root=repo_root / "dist")
    entries: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}

    source_items: list[tuple[str, Path, Path]] = [
        ("environment", environment, Path("environment/environment.json"))
    ]
    for group in selected_groups:
        paths = sorted(
            path for path in group.source_root.rglob("*") if group.include(path, group.source_root)
        )
        if not paths:
            raise RuntimeError(f"source group selected no files: {group.name}")
        group_counts[group.name] = len(paths)
        source_items.extend(
            (group.name, path, group.release_root / path.relative_to(group.source_root))
            for path in paths
        )

    for group_name, source, release_path in source_items:
        if source.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise RuntimeError(f"forbidden binary selected: {source}")
        content = sanitized_bytes(source, repo_root=repo_root)
        destination = output / release_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        entries.append(
            {
                "bytes": len(content),
                "group": group_name,
                "path": release_path.as_posix(),
                "sha256": _sha256_bytes(content),
                "source_path": source.relative_to(repo_root).as_posix(),
                "source_sha256": sha256_file(source),
                "transformed": content != source.read_bytes(),
            }
        )

    readme = _readme(len(entries) + 1)
    (output / "README.md").write_bytes(readme)
    entries.append(
        {
            "bytes": len(readme),
            "group": "bundle-documentation",
            "path": "README.md",
            "sha256": _sha256_bytes(readme),
            "source_path": None,
            "source_sha256": None,
            "transformed": False,
        }
    )
    entries.sort(key=lambda item: item["path"])

    compact = json.loads(compact_manifest.read_text(encoding="utf-8"))
    manifest = {
        "archive_policy": {
            "canonical_json": True,
            "line_endings": "LF",
            "sanitized": True,
            "source_binaries_included": False,
        },
        "bundle_id": BUNDLE_ID,
        "deliberate_exclusions": [
            "model binaries and external weight data",
            "compiled NPU caches",
            "tokenizer copies",
            "embedding arrays",
            "pinned public corpus payload",
            "unrelated local experiments",
        ],
        "files": entries,
        "group_counts": group_counts,
        "recorded_through_utc": compact["recorded_through_utc"],
        "release_candidate": {
            "commit": _git_value(repo_root, "rev-parse", "HEAD"),
            "tracked_worktree_clean": tracked_worktree_is_clean(repo_root),
        },
        "schema_version": "1.0.0",
        "snapshot_manifest_sha256": sha256_file(compact_manifest),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    failures = validate_release_directory(output)
    if failures:
        raise RuntimeError("raw release validation failed:\n- " + "\n- ".join(failures))
    return output


def validate_release_directory(root: Path) -> list[str]:
    """Validate manifest integrity, selection policy, and sanitization."""

    failures: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return [f"invalid manifest: {error}"]

    expected_paths = {"manifest.json"}
    for entry in manifest.get("files", []):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"unsafe manifest path: {entry['path']}")
            continue
        expected_paths.add(relative.as_posix())
        path = root / relative
        if not path.is_file():
            failures.append(f"missing file: {entry['path']}")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden binary: {entry['path']}")
        content = path.read_bytes()
        if len(content) != entry["bytes"]:
            failures.append(f"byte-count mismatch: {entry['path']}")
        if _sha256_bytes(content) != entry["sha256"]:
            failures.append(f"SHA-256 mismatch: {entry['path']}")
        text = content.decode("utf-8", errors="replace")
        if "\r" in text:
            failures.append(f"non-LF line ending: {entry['path']}")
        if ANSI_ESCAPE.search(text):
            failures.append(f"ANSI escape remains: {entry['path']}")
        if (
            WINDOWS_BACKSLASH_PATH.search(text)
            or WINDOWS_ABSOLUTE_PATH.search(text)
            or POSIX_PRIVATE_PATH.search(text)
        ):
            failures.append(f"absolute path remains: {entry['path']}")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{name} pattern in {entry['path']}")

    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    for path in sorted(actual_paths - expected_paths):
        failures.append(f"unlisted file: {path}")
    for path in sorted(expected_paths - actual_paths):
        failures.append(f"manifest lists absent file: {path}")
    return failures


def write_deterministic_archive(root: Path, archive: Path) -> Path:
    """Write a byte-stable ZIP and SHA-256 sidecar from a validated bundle."""

    failures = validate_release_directory(root)
    if failures:
        raise RuntimeError("refusing to archive invalid release:\n- " + "\n- ".join(failures))
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as handle:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = Path(root.name) / path.relative_to(root)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            handle.writestr(info, path.read_bytes(), compresslevel=9)
    digest = sha256_file(archive)
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {archive.name}\n", encoding="utf-8", newline="\n")
    return archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow a review build while tracked files differ from HEAD",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate --output without rebuilding it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if args.validate_only:
        failures = validate_release_directory(output)
        if failures:
            raise SystemExit("raw release validation failed:\n- " + "\n- ".join(failures))
        print(f"validated raw release: {output}")
        return
    root = build_release_directory(output=output, allow_dirty=args.allow_dirty)
    archive = write_deterministic_archive(root, args.archive.resolve())
    print(f"raw release directory: {root}")
    print(f"raw release archive: {archive}")
    print(f"archive SHA-256: {sha256_file(archive)}")


if __name__ == "__main__":
    main()
