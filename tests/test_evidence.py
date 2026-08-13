from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.run_benchmark import create_session, parse_args, validate_model_metadata
from npu_study.evidence import (
    NPUVerificationError,
    parse_assignment_report,
    provider_options,
    require_available_provider,
)


class EvidenceTests(unittest.TestCase):
    def test_missing_provider_fails(self):
        with self.assertRaises(NPUVerificationError):
            require_available_provider(["CPUExecutionProvider"])

    def test_require_npu_session_creation_fails_before_fallback(self):
        fake_ort = types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            with self.assertRaises(NPUVerificationError):
                create_session(
                    Path("missing.onnx"),
                    "npu",
                    Path("cache"),
                    True,
                )

    def test_valid_assignment_report(self):
        report = {
            "deviceStat": [
                {"name": "all", "nodeNum": 400},
                {"name": "VITIS_EP_CPU", "nodeNum": 2},
                {"name": "NPU", "nodeNum": 398},
            ],
            "subgraphStat": [{"device": "DPU", "count": 1}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            evidence = parse_assignment_report(path)
        self.assertEqual(evidence.npu_nodes, 398)
        self.assertEqual(evidence.cpu_nodes, 2)
        self.assertEqual(evidence.npu_subgraphs, 1)
        self.assertAlmostEqual(evidence.npu_node_coverage, 0.995)
        self.assertEqual(len(evidence.report_sha256), 64)

    def test_npu_cli_cannot_disable_evidence_gate(self):
        argv = [
            "run_benchmark.py",
            "--model",
            "fixture.onnx",
            "--provider",
            "npu",
            "--precision",
            "fp32",
            "--synthetic-input",
        ]
        with (
            patch.object(sys, "argv", argv),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_zero_npu_nodes_fails(self):
        report = {
            "deviceStat": [
                {"name": "all", "nodeNum": 4},
                {"name": "CPU", "nodeNum": 4},
            ],
            "subgraphStat": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(NPUVerificationError):
                parse_assignment_report(path)

    def test_zero_npu_subgraphs_fails(self):
        report = {
            "deviceStat": [
                {"name": "all", "nodeNum": 4},
                {"name": "NPU", "nodeNum": 4},
            ],
            "subgraphStat": [{"device": "CPU", "count": 1}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(NPUVerificationError):
                parse_assignment_report(path)

    def test_vitis_cache_key_changes_with_model_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            model.write_bytes(b"first")
            first = provider_options(model, root / "cache")["cache_key"]
            model.write_bytes(b"second")
            second = provider_options(model, root / "cache")["cache_key"]
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("model-"))

    def test_failed_quantized_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"model")
            metadata = {
                "output_model": {
                    "sha256": "9372c470eeadd5ecb9a0c8f35a2421e9aedea3f0cd742d46b6f8a3742d817b37"
                },
                "source_model": {"huggingface_revision": "revision"},
                "per_channel": True,
                "acceptance": {"passed": False},
            }
            model.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed smoke-fidelity"):
                validate_model_metadata(
                    model,
                    precision="dynamic-int8",
                    model_revision="revision",
                    quantization_granularity="per-channel",
                    required=True,
                )


if __name__ == "__main__":
    unittest.main()
