from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_raw_release import (
    sanitize_text,
    sanitized_bytes,
    validate_release_directory,
    write_deterministic_archive,
)


class RawReleaseTests(unittest.TestCase):
    def test_sanitizer_removes_identity_paths_and_ansi(self):
        repo = Path.home() / "Desktop" / "private project"
        text = (
            f"repo={repo} host-path=C:\\Users\\builder\\src\\runtime.cc:42 "
            "linker=-LC:\\embedded\\llvm_aie\\lib unix=/home/runner/work/model.onnx "
            "\x1b[31mfailed\x1b[0m"
        )
        sanitized = sanitize_text(text, repo_root=repo)
        self.assertIn("repo=<repo-root>", sanitized)
        self.assertNotIn("C:\\Users", sanitized)
        self.assertNotIn("C:\\embedded", sanitized)
        self.assertNotIn("/home/runner", sanitized)
        self.assertNotIn("\x1b", sanitized)
        self.assertIn("failed", sanitized)

    def test_json_is_canonicalized_and_sanitized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "record.json"
            source.write_text(json.dumps({"z": f"{root}\\model.onnx", "a": 1}), encoding="utf-8")
            content = sanitized_bytes(source, repo_root=root).decode("utf-8")
            self.assertEqual(json.loads(content), {"a": 1, "z": "<repo-root>\\model.onnx"})
            self.assertLess(content.index('"a"'), content.index('"z"'))
            self.assertTrue(content.endswith("\n"))

    def test_validator_detects_tampering_and_archive_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "bundle"
            root.mkdir()
            artifact = root / "record.json"
            artifact.write_text('{"passed": true}\n', encoding="utf-8", newline="\n")
            content = artifact.read_bytes()
            manifest = {
                "files": [
                    {
                        "bytes": len(content),
                        "path": "record.json",
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ]
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validate_release_directory(root), [])

            first = write_deterministic_archive(root, parent / "first.zip")
            second = write_deterministic_archive(root, parent / "second.zip")
            self.assertEqual(first.read_bytes(), second.read_bytes())

            artifact.write_text("tampered\n", encoding="utf-8", newline="\n")
            failures = validate_release_directory(root)
            self.assertTrue(any("mismatch" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
