from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.run_fidelity import load_dataset, pool_hidden, verify_dataset
from npu_study.artifacts import sha256_file


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class FidelityDatasetTests(unittest.TestCase):
    def test_pool_hidden_supports_cls_and_mean(self):
        hidden = np.asarray([[[3.0, 4.0], [0.0, 2.0]]], dtype=np.float32)
        mask = np.asarray([[1, 1]], dtype=np.int64)
        np.testing.assert_allclose(pool_hidden(hidden, mask, "cls"), [[0.6, 0.8]])
        np.testing.assert_allclose(
            pool_hidden(hidden, mask, "mean"),
            np.asarray([[1.5, 3.0]]) / np.linalg.norm([1.5, 3.0]),
        )

    def test_manifest_and_canonical_files_are_bound_to_spec(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            dataset.mkdir()
            rows = {
                "corpus": {"id": "d1", "title": "A title", "text": "body"},
                "queries": {"id": "q1", "text": "claim"},
                "qrels": {"query_id": "q1", "document_id": "d1", "score": 1.0},
            }
            outputs = {}
            for name, row in rows.items():
                path = dataset / f"{name}.jsonl"
                write_json(path, row)
                outputs[name] = {
                    "filename": path.name,
                    "rows": 1,
                    "sha256": sha256_file(path),
                }
            spec = {
                "id": "test/retrieval",
                "archive_sha256": "a" * 64,
                "split": "test",
                "licenses": {"dataset": "test-only"},
                "canonical_sha256": {name: output["sha256"] for name, output in outputs.items()},
                "expected_rows": {name: 1 for name in rows},
            }
            spec_path = root / "spec.json"
            write_json(spec_path, spec)
            write_json(dataset / "manifest.json", {"source": spec, "canonical_outputs": outputs})

            identity = verify_dataset(dataset, spec_path)
            self.assertEqual(identity["id"], "test/retrieval")
            self.assertEqual(identity["revision"], "a" * 64)

            query_ids, document_ids, queries, documents, qrels = load_dataset(dataset)
            self.assertEqual(query_ids, ["q1"])
            self.assertEqual(document_ids, ["d1"])
            self.assertEqual(queries, ["search_query: claim"])
            self.assertEqual(documents, ["search_document: A title body"])
            self.assertEqual(qrels, {"q1": {"d1"}})

            changed_spec = dict(spec)
            changed_spec["split"] = "train"
            write_json(spec_path, changed_spec)
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_dataset(dataset, spec_path)


if __name__ == "__main__":
    unittest.main()
