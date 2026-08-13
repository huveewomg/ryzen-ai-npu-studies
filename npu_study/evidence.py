"""Fail-closed verification for VitisAI NPU benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VITIS_PROVIDER = "VitisAIExecutionProvider"
REPORT_ENV_VAR = "XLNX_ONNX_EP_REPORT_FILE"
DEFAULT_REPORT_NAME = "vitisai_ep_report.json"
NPU_DEVICE_NAMES = {"DPU", "IPU", "NPU", "VAIML"}
CPU_DEVICE_NAMES = {"CPU", "VITIS_EP_CPU"}


class NPUVerificationError(RuntimeError):
    """Raised when an NPU-labelled run cannot prove NPU execution."""


@dataclass(frozen=True)
class AssignmentEvidence:
    report_file: str
    report_sha256: str
    total_nodes: int
    npu_nodes: int
    cpu_nodes: int
    npu_node_coverage: float
    npu_devices: tuple[str, ...]
    npu_subgraphs: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_available_provider(available_providers: Iterable[str]) -> None:
    providers = list(available_providers)
    if VITIS_PROVIDER not in providers:
        raise NPUVerificationError(f"{VITIS_PROVIDER} is unavailable; found providers: {providers}")


def require_session_provider(session_providers: Iterable[str]) -> None:
    providers = list(session_providers)
    if VITIS_PROVIDER not in providers:
        raise NPUVerificationError(
            f"session did not activate {VITIS_PROVIDER}; active providers: {providers}"
        )


def configure_assignment_report(report_name: str = DEFAULT_REPORT_NAME) -> None:
    """Ask VitisAI EP 1.5+ to emit an operator-assignment report."""

    if Path(report_name).name != report_name:
        raise ValueError("report_name must be a filename, not a path")
    os.environ[REPORT_ENV_VAR] = report_name


def provider_options(model_path: Path, cache_dir: Path) -> dict[str, str]:
    """Build VitisAI EP cache options tied to the exact model bytes."""

    resolved_cache = cache_dir.expanduser().resolve()
    if " " in str(resolved_cache):
        raise ValueError(
            "VitisAI cache paths containing spaces are unsupported on the tested SDK; "
            "pass --cache-dir with a space-free path"
        )
    resolved_cache.mkdir(parents=True, exist_ok=True)
    resolved_model = model_path.expanduser().resolve()
    if not resolved_model.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {resolved_model}")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", resolved_model.stem)
    cache_key = f"{stem}-{_sha256(resolved_model)[:16]}"
    return {
        "cache_dir": str(resolved_cache),
        "cache_key": cache_key,
        "enable_cache_file_io_in_mem": "0",
    }


def locate_assignment_report(cache_dir: Path, report_name: str = DEFAULT_REPORT_NAME) -> Path:
    matches = sorted(
        cache_dir.expanduser().resolve().rglob(report_name),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not matches:
        raise NPUVerificationError(
            f"NPU assignment report {report_name!r} was not generated under {cache_dir}"
        )
    return matches[0]


def parse_assignment_report(path: Path) -> AssignmentEvidence:
    """Validate VitisAI's JSON operator-assignment report.

    Ryzen AI releases have used NPU, DPU, IPU, and VAIML labels. All are
    accepted, but at least one assigned NPU node is mandatory.
    """

    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise NPUVerificationError(f"cannot read assignment report {path}: {exc}") from exc

    stats = report.get("deviceStat")
    if not isinstance(stats, list):
        raise NPUVerificationError("assignment report has no deviceStat list")

    total_nodes = 0
    cpu_nodes = 0
    npu_nodes = 0
    npu_devices: list[str] = []
    for entry in stats:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip().upper()
        node_count = entry.get("nodeNum", 0)
        if not isinstance(node_count, int) or node_count < 0:
            raise NPUVerificationError(f"invalid nodeNum for device {name!r}")
        if name == "ALL":
            total_nodes = node_count
        elif name in CPU_DEVICE_NAMES:
            cpu_nodes += node_count
        elif name in NPU_DEVICE_NAMES:
            npu_nodes += node_count
            npu_devices.append(name)

    if npu_nodes <= 0:
        raise NPUVerificationError("assignment report contains zero NPU-assigned nodes")
    if total_nodes <= 0:
        total_nodes = cpu_nodes + npu_nodes
    if total_nodes <= 0 or npu_nodes + cpu_nodes > total_nodes:
        raise NPUVerificationError("assignment report contains inconsistent node counts")

    subgraph_stats = report.get("subgraphStat")
    if not isinstance(subgraph_stats, list):
        raise NPUVerificationError("assignment report has no subgraphStat list")
    npu_subgraphs = 0
    for entry in subgraph_stats:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("device", entry.get("name", ""))).strip().upper()
        count = entry.get("count", entry.get("subgraphNum", 0))
        if not isinstance(count, int) or count < 0:
            raise NPUVerificationError(f"invalid subgraph count for device {name!r}")
        if name in NPU_DEVICE_NAMES:
            npu_subgraphs += count
    if npu_subgraphs <= 0:
        raise NPUVerificationError("assignment report contains zero NPU subgraphs")

    return AssignmentEvidence(
        report_file=path.name,
        report_sha256=_sha256(path),
        total_nodes=total_nodes,
        npu_nodes=npu_nodes,
        cpu_nodes=cpu_nodes,
        npu_node_coverage=npu_nodes / total_nodes,
        npu_devices=tuple(sorted(set(npu_devices))),
        npu_subgraphs=npu_subgraphs,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
