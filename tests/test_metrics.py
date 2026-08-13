from __future__ import annotations

import unittest

import numpy as np

from npu_study.metrics import (
    average_precision_at_k,
    embedding_drift,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    retrieval_metrics,
    spearman_correlation,
)


class MetricsTests(unittest.TestCase):
    def test_identical_embedding_drift(self):
        values = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = embedding_drift(values, values.copy())
        self.assertAlmostEqual(result["min_cosine"], 1.0)
        self.assertEqual(result["max_absolute_error"], 0.0)

    def test_retrieval_metrics(self):
        queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        documents = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = retrieval_metrics(
            query_embeddings=queries,
            document_embeddings=documents,
            query_ids=["q1", "q2"],
            document_ids=["d1", "d2"],
            qrels={"q1": {"d1"}, "q2": {"d2"}},
        )
        self.assertEqual(result["recall_at_1"], 1.0)
        self.assertEqual(result["ndcg_at_10"], 1.0)
        self.assertEqual(result["map_at_10"], 1.0)
        self.assertEqual(result["mrr_at_10"], 1.0)

    def test_rank_metrics(self):
        self.assertEqual(recall_at_k(["a", "b"], {"b"}, 2), 1.0)
        self.assertGreater(ndcg_at_k(["a", "b"], {"b"}, 2), 0.0)
        correlation = spearman_correlation(np.asarray([1.0, 2.0, 3.0]), np.asarray([3.0, 2.0, 1.0]))
        self.assertAlmostEqual(correlation, -1.0)

    def test_average_precision_and_reciprocal_rank(self):
        ranked = ["x", "a", "y", "b"]
        relevant = {"a", "b"}
        self.assertAlmostEqual(average_precision_at_k(ranked, relevant, 4), 0.5)
        self.assertAlmostEqual(reciprocal_rank_at_k(ranked, relevant, 4), 0.5)


if __name__ == "__main__":
    unittest.main()
