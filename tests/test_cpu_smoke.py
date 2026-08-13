from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.run_fidelity import OnnxEmbedder
from npu_study.metrics import embedding_drift


class _TinyTokenizer:
    def __call__(self, texts, **_kwargs):
        count = len(texts)
        return {
            "input_ids": np.tile(np.asarray([[1, 2, 3, 4]], dtype=np.int64), (count, 1)),
            "attention_mask": np.ones((count, 4), dtype=np.int64),
        }


class CpuSmokeTests(unittest.TestCase):
    def test_tiny_onnx_export_and_fidelity(self):
        try:
            import onnx
            import onnxruntime as ort
            from onnx import TensorProto, helper
        except ImportError as exc:
            self.skipTest(f"CPU ONNX dependencies unavailable: {exc}")

        input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 4])
        attention_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, 4])
        output = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, 4, 1])
        shape = helper.make_tensor("shape", TensorProto.INT64, [3], [1, 4, 1])
        nodes = [
            helper.make_node("Cast", ["input_ids"], ["float_ids"], to=TensorProto.FLOAT),
            helper.make_node("Reshape", ["float_ids", "shape"], ["hidden"]),
            helper.make_node("Cast", ["attention_mask"], ["float_mask"], to=TensorProto.FLOAT),
            helper.make_node("Reshape", ["float_mask", "shape"], ["mask_3d"]),
            helper.make_node("Mul", ["hidden", "mask_3d"], ["last_hidden_state"]),
        ]
        graph = helper.make_graph(
            nodes,
            "smoke",
            [input_ids, attention_mask],
            [output],
            initializer=[shape],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = min(model.ir_version, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.onnx"
            onnx.save(model, path)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.assertEqual(session.get_providers()[0], "CPUExecutionProvider")
            embedder = OnnxEmbedder(
                model_path=path,
                provider="cpu",
                cache_dir=None,
                require_npu=False,
            )
            embeddings = embedder.encode(_TinyTokenizer(), ["one", "two"])
        np.testing.assert_allclose(embeddings, np.ones((2, 1), dtype=np.float32))
        self.assertAlmostEqual(embedding_drift(embeddings, embeddings)["min_cosine"], 1.0)


if __name__ == "__main__":
    unittest.main()
