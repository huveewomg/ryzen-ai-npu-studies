"""Export a pinned Nomic model revision to fixed-shape ONNX artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npu_study.artifacts import package_versions, sha256_file, utc_now, write_json  # noqa: E402

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
DEFAULT_SEQ_LEN = 128
DEFAULT_BATCH_SIZE = 1
OPSET = 17


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * expanded_mask, 1) / torch.clamp(
        expanded_mask.sum(1), min=1e-9
    )


def export(
    *,
    seq_len: int,
    batch_size: int,
    output_dir: Path,
    model_revision: str,
    min_cosine: float,
) -> Path:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=model_revision)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        revision=model_revision,
        trust_remote_code=True,
    )
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"nomic-embed-v1.5_b{batch_size}_seq{seq_len}.onnx"
    dummy_input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    with torch.no_grad():
        test_output = model(dummy_input_ids, dummy_attention_mask)
    print(f"last_hidden_state shape: {test_output.last_hidden_state.shape}")

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        opset_version=OPSET,
        do_constant_folding=True,
    )

    verification = verify_onnx(
        output_path=output_path,
        tokenizer=tokenizer,
        torch_model=model,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    if verification["min_cosine"] < min_cosine:
        raise RuntimeError(
            f"ONNX verification cosine {verification['min_cosine']:.8f} is below {min_cosine}"
        )

    metadata = {
        "schema_version": "1.0.0",
        "created_at_utc": utc_now(),
        "source_model": MODEL_NAME,
        "source_revision": model_revision,
        "onnx": {
            "filename": output_path.name,
            "sha256": sha256_file(output_path),
            "opset": OPSET,
            "batch_size": batch_size,
            "sequence_length": seq_len,
        },
        "verification": verification,
        "packages": package_versions(),
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    print(f"exported: {output_path} ({output_path.stat().st_size / 1024**2:.1f} MiB)")
    return output_path


def verify_onnx(
    *,
    output_path: Path,
    tokenizer,
    torch_model,
    seq_len: int,
    batch_size: int,
) -> dict[str, float]:
    import numpy as np
    import onnxruntime as ort

    examples = [
        "search_query: What is a neural processing unit?",
        "search_document: An NPU accelerates machine-learning inference.",
        "search_document: A weather forecast can predict rain.",
    ]
    texts = [examples[index % len(examples)] for index in range(batch_size)]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    with torch.no_grad():
        pt_output = torch_model(encoded["input_ids"], encoded["attention_mask"])
        pt_embedding = functional.normalize(
            mean_pooling(pt_output, encoded["attention_mask"]), p=2, dim=1
        )

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": encoded["input_ids"].numpy(),
        "attention_mask": encoded["attention_mask"].numpy(),
    }
    ort_hidden = torch.from_numpy(session.run(None, ort_inputs)[0])
    ort_embedding = functional.normalize(
        mean_pooling((ort_hidden,), encoded["attention_mask"]), p=2, dim=1
    )
    cosines = functional.cosine_similarity(pt_embedding, ort_embedding).numpy()
    max_difference = float(torch.max(torch.abs(pt_embedding - ort_embedding)).item())
    result = {
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
        "max_absolute_difference": max_difference,
    }
    print(
        f"verification: min cosine={result['min_cosine']:.8f}, "
        f"max abs diff={result['max_absolute_difference']:.3e}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pinned Nomic ONNX artifacts")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    args = parser.parse_args()
    if args.seq_len < 1 or args.batch_size < 1:
        parser.error("sequence length and batch size must be positive")
    return args


def main() -> None:
    args = parse_args()
    export(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        model_revision=args.model_revision,
        min_cosine=args.min_cosine,
    )


if __name__ == "__main__":
    main()
