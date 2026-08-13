"""Reproducibility metadata and artifact helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "datasets",
    "einops",
    "numpy",
    "onnx",
    "onnxruntime",
    "onnxruntime-providers-ryzenai",
    "onnxruntime-vitisai",
    "torch",
    "transformers",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def package_versions(names: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str]:
    """Collect package versions, marking unavailable packages explicitly."""

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def git_commit(repo_root: Path) -> str | None:
    """Return the current commit without failing outside a Git checkout."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_tracked_dirty(repo_root: Path) -> bool | None:
    """Report tracked worktree changes while ignoring local untracked files."""

    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def environment_manifest(
    *,
    repo_root: Path,
    available_providers: Iterable[str] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the portable portion of a run environment manifest.

    Driver, power-profile, and detailed hardware fields must be supplied in
    ``extra`` because they are platform- and permission-dependent. Unknown
    values remain visible instead of being guessed.
    """

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "captured_at_utc": utc_now(),
        "git_commit": git_commit(repo_root),
        "git_tracked_dirty": git_tracked_dirty(repo_root),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": package_versions(),
        "onnxruntime_available_providers": list(available_providers),
        "hardware": {"cpu": None, "npu": None, "ram": None},
        "software": {
            "ryzen_ai_sdk": None,
            "npu_driver": None,
            "windows_build": platform.version() if platform.system() == "Windows" else None,
        },
        "controls": {"power_profile": None, "thread_settings": None},
    }
    if extra:
        _deep_update(manifest, extra)
    return manifest


def write_json(path: Path, data: Any) -> None:
    """Write stable, human-reviewable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
