"""Export a pinned BERT-family embedding model to a fixed-shape ONNX artifact."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npu_study.artifacts import package_versions, sha256_file, utc_now, write_json  # noqa: E402

BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
OPSET = 17


class HiddenStateWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "mean":
        expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        return torch.sum(hidden * expanded, 1) / torch.clamp(expanded.sum(1), min=1e-9)
    raise ValueError(f"unsupported pooling: {pooling}")


def safe_stem(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_id.split("/")[-1]).strip("-")


def verify(
    *,
    output_path: Path,
    model,
    tokenizer,
    batch_size: int,
    seq_len: int,
    pooling: str,
) -> dict[str, float]:
    import numpy as np
    import onnxruntime as ort

    examples = [
        "Represent this sentence for searching relevant passages: What is an NPU?",
        "An NPU accelerates machine-learning inference.",
        "A weather forecast can predict rain.",
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
        torch_hidden = model(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        ).last_hidden_state
    torch_embeddings = functional.normalize(
        pool_hidden(torch_hidden, encoded["attention_mask"], pooling), p=2, dim=1
    )
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    inputs = {
        "input_ids": encoded["input_ids"].numpy(),
        "attention_mask": encoded["attention_mask"].numpy(),
    }
    onnx_hidden = torch.from_numpy(session.run(None, inputs)[0])
    onnx_embeddings = functional.normalize(
        pool_hidden(onnx_hidden, encoded["attention_mask"], pooling), p=2, dim=1
    )
    cosines = functional.cosine_similarity(torch_embeddings, onnx_embeddings).numpy()
    return {
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
        "max_absolute_difference": float(
            torch.max(torch.abs(torch_embeddings - onnx_embeddings)).item()
        ),
    }


def export(args: argparse.Namespace) -> Path:
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    model = AutoModel.from_pretrained(args.model_id, revision=args.model_revision)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem or safe_stem(args.model_id)
    output_path = args.output_dir / f"{stem}_b{args.batch_size}_seq{args.seq_len}.onnx"
    dummy_input_ids = torch.ones(args.batch_size, args.seq_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(args.batch_size, args.seq_len, dtype=torch.long)
    wrapper = HiddenStateWrapper(model).eval()
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        opset_version=OPSET,
        do_constant_folding=True,
    )
    verification = verify(
        output_path=output_path,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pooling=args.pooling,
    )
    if verification["min_cosine"] < args.min_cosine:
        raise RuntimeError(
            f"ONNX verification cosine {verification['min_cosine']:.8f} is below {args.min_cosine}"
        )
    write_json(
        output_path.with_suffix(".metadata.json"),
        {
            "schema_version": "1.0.0",
            "created_at_utc": utc_now(),
            "source_model": args.model_id,
            "source_revision": args.model_revision,
            "pooling": args.pooling,
            "onnx": {
                "filename": output_path.name,
                "sha256": sha256_file(output_path),
                "opset": OPSET,
                "batch_size": args.batch_size,
                "sequence_length": args.seq_len,
            },
            "verification": verification,
            "packages": package_versions(),
        },
    )
    print(
        f"exported {output_path} ({output_path.stat().st_size / 1024**2:.1f} MiB); "
        f"minimum cosine={verification['min_cosine']:.8f}"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-shape BERT embeddings")
    parser.add_argument("--model-id", default=BGE_MODEL_ID)
    parser.add_argument("--model-revision", default=BGE_REVISION)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem")
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    args = parser.parse_args()
    if args.batch_size < 1 or args.seq_len < 1:
        parser.error("batch size and sequence length must be positive")
    return args


if __name__ == "__main__":
    export(parse_args())
