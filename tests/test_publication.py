from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.validate_results import validate_publication_manifest


class PublicationManifestTests(unittest.TestCase):
    def test_manifest_detects_tampered_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "table.csv"
            artifact.write_text("a,b\n1,2\n", encoding="utf-8")
            content = artifact.read_bytes()
            manifest = {
                "published_files": [
                    {
                        "path": artifact.name,
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validate_publication_manifest(manifest_path), [])
            artifact.write_text("tampered\n", encoding="utf-8")
            failures = validate_publication_manifest(manifest_path)
            self.assertTrue(any("mismatch" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
