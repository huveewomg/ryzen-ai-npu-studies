from __future__ import annotations

import unittest

try:
    import torch

    from models.export_bert_embed import pool_hidden, safe_stem
except ModuleNotFoundError as error:
    if error.name not in {"torch", "transformers"}:
        raise
    torch = None
    pool_hidden = None
    safe_stem = None


@unittest.skipIf(torch is None, "requires torch and transformers")
class BertExportTests(unittest.TestCase):
    def test_cls_and_mean_pooling(self):
        hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]])
        mask = torch.tensor([[1, 1, 0]])
        self.assertTrue(torch.equal(pool_hidden(hidden, mask, "cls"), torch.tensor([[1.0, 2.0]])))
        self.assertTrue(torch.equal(pool_hidden(hidden, mask, "mean"), torch.tensor([[2.0, 3.0]])))

    def test_model_id_becomes_safe_filename(self):
        self.assertEqual(safe_stem("BAAI/bge-small-en-v1.5"), "bge-small-en-v1.5")


if __name__ == "__main__":
    unittest.main()
