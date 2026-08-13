"""Compare PyTorch, ONNX CPU, and verified VitisAI NPU embeddings."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npu_study.artifacts import (  # noqa: E402
    environment_manifest,
    sha256_file,
    utc_now,
    write_json,
)
from npu_study.evidence import (  # noqa: E402
    DEFAULT_REPORT_NAME,
    VITIS_PROVIDER,
    configure_assignment_report,
    locate_assignment_report,
    parse_assignment_report,
    provider_options,
    require_available_provider,
    require_session_provider,
)
from npu_study.metrics import embedding_drift, normalize_rows, retrieval_metrics  # noqa: E402

MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
DEFAULT_DATASET = Path(__file__).parent / "data" / "nanoscifact"
DEFAULT_DATASET_SPEC = Path(__file__).parent / "datasets" / "nanoscifact.json"
DEFAULT_ACCEPTANCE = Path(__file__).parent / "acceptance.json"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "local"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def verify_dataset(path: Path, spec_path: Path = DEFAULT_DATASET_SPEC) -> dict[str, Any]:
    with spec_path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    with (path / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("source") != spec:
        raise ValueError("dataset manifest source does not match the committed specification")

    canonical_outputs: dict[str, dict[str, Any]] = {}
    for name in ("corpus", "queries", "qrels"):
        source_path = path / f"{name}.jsonl"
        checksum = sha256_file(source_path)
        expected_checksum = spec["canonical_sha256"][name]
        row_count = len(read_jsonl(source_path))
        expected_rows = int(spec["expected_rows"][name])
        recorded = manifest.get("canonical_outputs", {}).get(name, {})
        if checksum != expected_checksum or recorded.get("sha256") != checksum:
            raise ValueError(f"{name} canonical SHA-256 verification failed")
        if row_count != expected_rows or recorded.get("rows") != row_count:
            raise ValueError(f"{name} row-count verification failed")
        canonical_outputs[name] = {
            "filename": source_path.name,
            "rows": row_count,
            "sha256": checksum,
        }

    return {
        "id": spec.get("id", spec.get("dataset")),
        "revision": spec.get("revision", spec.get("archive_sha256")),
        "licenses": spec.get("licenses", {"dataset": spec.get("license")}),
        "split": spec["split"],
        "source": spec.get("source", spec.get("url")),
        "spec_filename": spec_path.name,
        "spec_sha256": sha256_file(spec_path),
        "canonical_outputs": canonical_outputs,
    }


def pool_hidden(
    hidden: np.ndarray, attention_mask: np.ndarray, pooling: str = "mean"
) -> np.ndarray:
    if pooling == "cls":
        pooled = hidden[:, 0]
    elif pooling == "mean":
        expanded = attention_mask.astype(np.float32)[..., None]
        pooled = np.sum(hidden * expanded, axis=1) / np.maximum(np.sum(expanded, axis=1), 1e-9)
    else:
        raise ValueError(f"unsupported pooling: {pooling}")
    return normalize_rows(pooled)


def batches(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        real = items[start : start + batch_size]
        yield real + [real[-1]] * (batch_size - len(real)), len(real)


def encode_pytorch(
    model,
    tokenizer,
    items: list[str],
    batch_size: int,
    seq_len: int,
    pooling: str = "mean",
) -> np.ndarray:
    import torch

    output: list[np.ndarray] = []
    for text_batch, real_count in batches(items, batch_size):
        encoded = tokenizer(
            text_batch,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = model(encoded["input_ids"], encoded["attention_mask"])[0].cpu().numpy()
        pooled = pool_hidden(hidden, encoded["attention_mask"].numpy(), pooling)
        output.extend(pooled[:real_count])
    return np.asarray(output, dtype=np.float32)


class OnnxEmbedder:
    def __init__(
        self,
        *,
        model_path: Path,
        provider: str,
        cache_dir: Path | None,
        require_npu: bool,
        pooling: str = "mean",
    ) -> None:
        import onnxruntime as ort

        self.model_path = model_path.resolve()
        self.available_providers = ort.get_available_providers()
        self.report_root: Path | None = None
        self.report_path: Path | None = None
        self.assignment = None
        options = None
        if provider == "npu":
            if cache_dir is None:
                raise ValueError("cache_dir is required for NPU fidelity runs")
            if require_npu:
                require_available_provider(self.available_providers)
                configure_assignment_report(DEFAULT_REPORT_NAME)
            values = provider_options(self.model_path, cache_dir)
            options = [values]
            self.report_root = cache_dir / values["cache_key"]
        requested = VITIS_PROVIDER if provider == "npu" else "CPUExecutionProvider"
        started = time.perf_counter()
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=[requested],
            provider_options=options,
        )
        self.session_creation_ms = (time.perf_counter() - started) * 1000.0
        self.session_providers = self.session.get_providers()
        self.provider = provider
        self.require_npu = require_npu
        self.pooling = pooling
        if provider == "npu" and require_npu:
            require_session_provider(self.session_providers)

        input_item = next(item for item in self.session.get_inputs() if item.name == "input_ids")
        if len(input_item.shape) != 2 or not all(
            isinstance(value, int) for value in input_item.shape
        ):
            raise ValueError("fidelity evaluation requires fixed batch and sequence dimensions")
        self.batch_size = int(input_item.shape[0])
        self.seq_len = int(input_item.shape[1])

    def encode(self, tokenizer, items: list[str]) -> np.ndarray:
        output: list[np.ndarray] = []
        for text_batch, real_count in batches(items, self.batch_size):
            encoded = tokenizer(
                text_batch,
                padding="max_length",
                truncation=True,
                max_length=self.seq_len,
                return_tensors="np",
            )
            inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            pooled = pool_hidden(
                self.session.run(None, inputs)[0], inputs["attention_mask"], self.pooling
            )
            output.extend(pooled[:real_count])

        if self.provider == "npu" and self.require_npu:
            root = self.report_root if self.report_root and self.report_root.exists() else None
            if root is None:
                raise RuntimeError("VitisAI report cache directory was not created")
            self.report_path = locate_assignment_report(root)
            self.assignment = parse_assignment_report(self.report_path)
        return np.asarray(output, dtype=np.float32)

    def evidence(self) -> dict[str, Any]:
        return {
            "model_filename": self.model_path.name,
            "model_sha256": sha256_file(self.model_path),
            "requested_provider": self.provider,
            "available_providers": self.available_providers,
            "session_providers": self.session_providers,
            "session_creation_ms": self.session_creation_ms,
            "npu_verified": self.assignment is not None,
            "assignment": self.assignment.to_dict() if self.assignment else None,
        }


def load_dataset(
    path: Path,
    query_prefix: str = "search_query: ",
    document_prefix: str = "search_document: ",
):
    corpus = read_jsonl(path / "corpus.jsonl")
    queries = read_jsonl(path / "queries.jsonl")
    qrel_rows = read_jsonl(path / "qrels.jsonl")
    query_ids = [str(row["id"]) for row in queries]
    document_ids = [str(row["id"]) for row in corpus]
    query_texts = [f"{query_prefix}{row['text']}" for row in queries]
    document_texts = []
    for row in corpus:
        body = " ".join(
            part.strip() for part in (str(row.get("title", "")), str(row["text"])) if part.strip()
        )
        document_texts.append(f"{document_prefix}{body}")
    qrels: dict[str, set[str]] = {}
    for row in qrel_rows:
        if float(row.get("score", 1.0)) > 0:
            qrels.setdefault(str(row["query_id"]), set()).add(str(row["document_id"]))
    return query_ids, document_ids, query_texts, document_texts, qrels


def evaluate_acceptance(
    acceptance: dict[str, Any],
    drifts: dict[str, dict[str, Any]],
    retrieval: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    reference = retrieval["pytorch_fp32"]
    checks: list[dict[str, Any]] = []
    for name, drift in drifts.items():
        checks.append(
            {
                "name": f"{name}.minimum_per_item_cosine",
                "actual": drift["min_cosine"],
                "threshold": acceptance["minimum_per_item_cosine"][name],
                "passed": drift["min_cosine"] >= acceptance["minimum_per_item_cosine"][name],
            }
        )
        checks.append(
            {
                "name": f"{name}.maximum_absolute_error",
                "actual": drift["max_absolute_error"],
                "threshold": acceptance["maximum_absolute_error"][name],
                "passed": drift["max_absolute_error"] <= acceptance["maximum_absolute_error"][name],
            }
        )
        recall_drop = reference["recall_at_10"] - retrieval[name]["recall_at_10"]
        ndcg_drop = reference["ndcg_at_10"] - retrieval[name]["ndcg_at_10"]
        checks.extend(
            [
                {
                    "name": f"{name}.recall_at_10_drop",
                    "actual": recall_drop,
                    "threshold": acceptance["maximum_absolute_recall_at_10_drop"],
                    "passed": recall_drop <= acceptance["maximum_absolute_recall_at_10_drop"],
                },
                {
                    "name": f"{name}.ndcg_at_10_drop",
                    "actual": ndcg_drop,
                    "threshold": acceptance["maximum_absolute_ndcg_at_10_drop"],
                    "passed": ndcg_drop <= acceptance["maximum_absolute_ndcg_at_10_drop"],
                },
                {
                    "name": f"{name}.mean_ranking_spearman",
                    "actual": retrieval[name]["mean_ranking_spearman"],
                    "threshold": acceptance["minimum_mean_ranking_spearman"],
                    "passed": retrieval[name]["mean_ranking_spearman"]
                    >= acceptance["minimum_mean_ranking_spearman"],
                },
            ]
        )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def run(args: argparse.Namespace) -> None:
    from transformers import AutoModel, AutoTokenizer

    with args.acceptance.open("r", encoding="utf-8") as handle:
        acceptance = json.load(handle)
    dataset_identity = verify_dataset(args.dataset, args.dataset_spec)
    query_ids, document_ids, query_texts, document_texts, qrels = load_dataset(
        args.dataset, args.query_prefix, args.document_prefix
    )
    all_items = query_texts + document_texts
    item_ids = [f"query:{item}" for item in query_ids] + [
        f"document:{item}" for item in document_ids
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    torch_model = AutoModel.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        trust_remote_code=True,
    ).cpu()
    torch_model.eval()

    fp32_cpu = OnnxEmbedder(
        model_path=args.fp32_model,
        provider="cpu",
        cache_dir=None,
        require_npu=False,
        pooling=args.pooling,
    )
    batch_size, seq_len = fp32_cpu.batch_size, fp32_cpu.seq_len
    stage_timings_seconds: dict[str, float] = {}

    def timed_encode(name: str, callback) -> np.ndarray:
        print(f"encoding {name}: {len(all_items)} items", flush=True)
        started = time.perf_counter()
        values = callback()
        elapsed = time.perf_counter() - started
        stage_timings_seconds[name] = elapsed
        print(f"encoded {name}: {elapsed:.1f} seconds", flush=True)
        return values

    embeddings: dict[str, np.ndarray] = {
        "pytorch_fp32": timed_encode(
            "pytorch_fp32",
            lambda: encode_pytorch(
                torch_model,
                tokenizer,
                all_items,
                batch_size,
                seq_len,
                args.pooling,
            ),
        ),
        "onnx_fp32_cpu": timed_encode(
            "onnx_fp32_cpu", lambda: fp32_cpu.encode(tokenizer, all_items)
        ),
    }
    embedders: dict[str, OnnxEmbedder] = {"onnx_fp32_cpu": fp32_cpu}

    if args.int8_model:
        int8_cpu = OnnxEmbedder(
            model_path=args.int8_model,
            provider="cpu",
            cache_dir=None,
            require_npu=False,
            pooling=args.pooling,
        )
        if (int8_cpu.batch_size, int8_cpu.seq_len) != (batch_size, seq_len):
            raise ValueError("FP32 and dynamic INT8 model shapes must match")
        embeddings["onnx_dynamic_int8_cpu"] = timed_encode(
            "onnx_dynamic_int8_cpu", lambda: int8_cpu.encode(tokenizer, all_items)
        )
        embedders["onnx_dynamic_int8_cpu"] = int8_cpu

    if not args.cpu_only:
        npu_candidates = []
        if args.npu_precisions in {"fp32", "both"}:
            npu_candidates.append(("onnx_fp32_npu", args.fp32_model))
        if args.npu_precisions in {"dynamic-int8", "both"}:
            npu_candidates.append(("onnx_dynamic_int8_npu", args.int8_model))
        for name, model_path in npu_candidates:
            embedder = OnnxEmbedder(
                model_path=model_path,
                provider="npu",
                cache_dir=args.cache_dir,
                require_npu=True,
                pooling=args.pooling,
            )
            embeddings[name] = timed_encode(
                name, lambda item=embedder: item.encode(tokenizer, all_items)
            )
            embedders[name] = embedder

    query_count = len(query_ids)
    reference_queries = embeddings["pytorch_fp32"][:query_count]
    reference_documents = embeddings["pytorch_fp32"][query_count:]
    reference_scores = normalize_rows(reference_queries) @ normalize_rows(reference_documents).T
    retrieval = {
        "pytorch_fp32": retrieval_metrics(
            query_embeddings=reference_queries,
            document_embeddings=reference_documents,
            query_ids=query_ids,
            document_ids=document_ids,
            qrels=qrels,
        )
    }
    drifts: dict[str, dict[str, Any]] = {}
    for name, values in embeddings.items():
        if name == "pytorch_fp32":
            continue
        drift = embedding_drift(embeddings["pytorch_fp32"], values)
        drift["worst_cosine_item"] = item_ids[drift["worst_cosine_index"]]
        drift["worst_absolute_error_item"] = item_ids[drift["worst_absolute_error_index"]]
        drifts[name] = drift
        retrieval[name] = retrieval_metrics(
            query_embeddings=values[:query_count],
            document_embeddings=values[query_count:],
            query_ids=query_ids,
            document_ids=document_ids,
            qrels=qrels,
            reference_scores=reference_scores,
        )

    applicable_acceptance = json.loads(json.dumps(acceptance))
    for name in list(applicable_acceptance["minimum_per_item_cosine"]):
        if name not in drifts:
            del applicable_acceptance["minimum_per_item_cosine"][name]
            del applicable_acceptance["maximum_absolute_error"][name]
    decision = evaluate_acceptance(applicable_acceptance, drifts, retrieval)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "1.0.0",
        "recorded_at_utc": utc_now(),
        "model": {
            "id": args.model_id,
            "revision": args.model_revision,
            "pooling": args.pooling,
            "query_prefix": args.query_prefix,
            "document_prefix": args.document_prefix,
        },
        "dataset": dataset_identity,
        "scope": {
            "cpu_only": args.cpu_only,
            "npu_precisions": "none" if args.cpu_only else args.npu_precisions,
            "evaluated_embeddings": sorted(embeddings),
        },
        "stage_timings_seconds": stage_timings_seconds,
        "acceptance_sha256": sha256_file(args.acceptance),
        "acceptance": applicable_acceptance,
        "decision": decision,
        "drift": drifts,
        "retrieval": retrieval,
        "execution_evidence": {name: value.evidence() for name, value in embedders.items()},
    }
    write_json(args.output_dir / "fidelity.json", result)
    providers = fp32_cpu.available_providers
    extra = {}
    if args.manifest_input:
        with args.manifest_input.open("r", encoding="utf-8") as handle:
            extra = json.load(handle)
    write_json(
        args.output_dir / "environment.json",
        environment_manifest(
            repo_root=REPO_ROOT,
            available_providers=providers,
            extra=extra,
        ),
    )
    np.savez_compressed(args.output_dir / "embeddings.npz", **embeddings)
    for name, embedder in embedders.items():
        if embedder.report_path:
            shutil.copy2(embedder.report_path, args.output_dir / f"{name}-{DEFAULT_REPORT_NAME}")
    if not decision["passed"]:
        raise RuntimeError("one or more predeclared fidelity thresholds failed")
    print(f"fidelity gates passed: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run numerical and retrieval fidelity gates")
    parser.add_argument("--fp32-model", type=Path, required=True)
    parser.add_argument(
        "--int8-model",
        type=Path,
        help="Optional rejected-candidate comparison; omit for the valid FP32-only protocol",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-spec", type=Path, default=DEFAULT_DATASET_SPEC)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--manifest-input", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--query-prefix", default="search_query: ")
    parser.add_argument("--document-prefix", default="search_document: ")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--npu-precisions",
        choices=["fp32", "dynamic-int8", "both"],
        default="fp32",
        help="Select NPU candidates; CPU evaluates every supplied artifact",
    )
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    args = parser.parse_args()
    if not args.cpu_only and args.cache_dir is None:
        parser.error("--cache-dir is required unless --cpu-only is selected")
    if args.npu_precisions in {"dynamic-int8", "both"} and args.int8_model is None:
        parser.error("--int8-model is required for the selected NPU precision scope")
    return args


if __name__ == "__main__":
    run(parse_args())
