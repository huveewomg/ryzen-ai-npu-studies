"""Apply ONNX Runtime dynamic weight-only INT8 quantization to Nomic.

This script does not use AMD Quark and does not calibrate activations. It keeps
activations in floating point and quantizes supported weights dynamically.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from contextlib import chdir
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npu_study.artifacts import package_versions, sha256_file, utc_now, write_json  # noqa: E402
from npu_study.metrics import embedding_drift, normalize_rows  # noqa: E402

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
DEFAULT_MODEL = Path(__file__).parent / "nomic-embed-v1.5_b1_seq128.onnx"


def fixed_input_shape(session) -> tuple[int, int]:
    input_item = next(item for item in session.get_inputs() if item.name == "input_ids")
    if len(input_item.shape) != 2 or not all(isinstance(value, int) for value in input_item.shape):
        raise ValueError("quantization smoke validation requires a fixed [batch, sequence] model")
    return int(input_item.shape[0]), int(input_item.shape[1])


def pool(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    expanded = attention_mask.astype(np.float32)[..., None]
    pooled = np.sum(hidden * expanded, axis=1) / np.maximum(np.sum(expanded, axis=1), 1e-9)
    return normalize_rows(pooled)


def verify_quality(
    *,
    fp32_path: Path,
    int8_path: Path,
    model_revision: str,
) -> dict:
    import onnxruntime as ort

    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    batch_size, seq_len = fixed_input_shape(fp32_session)
    if fixed_input_shape(int8_session) != (batch_size, seq_len):
        raise ValueError("FP32 and INT8 model input shapes differ")

    base_texts = [
        "search_query: What is a neural processing unit?",
        "search_document: A neural processing unit accelerates machine-learning inference.",
        "search_query: How does dynamic quantization work?",
        "search_document: Dynamic quantization stores weights as integers.",
        "search_document: The weather forecast predicts rain tomorrow.",
    ]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=model_revision)
    fp32_embeddings: list[np.ndarray] = []
    int8_embeddings: list[np.ndarray] = []
    item_ids: list[str] = []

    for start in range(0, len(base_texts), batch_size):
        real_texts = base_texts[start : start + batch_size]
        padded = real_texts + [real_texts[-1]] * (batch_size - len(real_texts))
        encoded = tokenizer(
            padded,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="np",
        )
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        fp32_batch = pool(fp32_session.run(None, inputs)[0], inputs["attention_mask"])
        int8_batch = pool(int8_session.run(None, inputs)[0], inputs["attention_mask"])
        fp32_embeddings.extend(fp32_batch[: len(real_texts)])
        int8_embeddings.extend(int8_batch[: len(real_texts)])
        item_ids.extend(f"smoke-{index}" for index in range(start, start + len(real_texts)))

    drift = embedding_drift(np.asarray(fp32_embeddings), np.asarray(int8_embeddings))
    drift["item_ids"] = item_ids
    drift["scope"] = "five-item smoke check; not retrieval-quality evidence"
    return drift


def quantize(args: argparse.Namespace) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    input_path = args.model.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_int8.onnx")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nomic-quant-") as directory:
        quantization_input = input_path
        if not args.skip_preprocess:
            quantization_input = Path(directory) / "preprocessed.onnx"
            with chdir(directory):
                quant_pre_process(
                    input_model=input_path,
                    output_model_path=quantization_input,
                )
        quantization_input_sha256 = sha256_file(quantization_input)
        quantize_dynamic(
            model_input=str(quantization_input),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            per_channel=args.per_channel,
        )
    drift = verify_quality(
        fp32_path=input_path,
        int8_path=output_path,
        model_revision=args.model_revision,
    )
    cosine_passed = drift["min_cosine"] >= args.min_cosine
    error_passed = drift["max_absolute_error"] <= args.max_absolute_error
    passed = cosine_passed and error_passed

    metadata = {
        "schema_version": "1.0.0",
        "created_at_utc": utc_now(),
        "method": "onnxruntime.quantization.quantize_dynamic",
        "weight_type": "QInt8",
        "per_channel": args.per_channel,
        "activation_quantization": "none",
        "calibration": "none",
        "preprocessing": {
            "applied": not args.skip_preprocess,
            "method": "onnxruntime.quantization.shape_inference.quant_pre_process",
            "quantization_input_sha256": quantization_input_sha256,
        },
        "source_model": {
            "filename": input_path.name,
            "sha256": sha256_file(input_path),
            "huggingface_id": MODEL_NAME,
            "huggingface_revision": args.model_revision,
        },
        "output_model": {
            "filename": output_path.name,
            "sha256": sha256_file(output_path),
        },
        "smoke_validation": drift,
        "acceptance": {
            "minimum_per_item_cosine": args.min_cosine,
            "maximum_absolute_error": args.max_absolute_error,
            "minimum_per_item_cosine_passed": cosine_passed,
            "maximum_absolute_error_passed": error_passed,
            "passed": passed,
        },
        "packages": package_versions(),
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    print(f"quantized: {output_path}")
    print(f"minimum per-item cosine: {drift['min_cosine']:.6f}")
    if not cosine_passed:
        raise RuntimeError(
            f"INT8 minimum per-item cosine {drift['min_cosine']:.6f} is below {args.min_cosine}"
        )
    if not error_passed:
        raise RuntimeError(
            "INT8 maximum absolute error "
            f"{drift['max_absolute_error']:.6f} exceeds {args.max_absolute_error}"
        )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic weight-only INT8 quantization")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--min-cosine", type=float, default=0.98)
    parser.add_argument("--max-absolute-error", type=float, default=0.10)
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    quantize(parse_args())


if __name__ == "__main__":
    main()
