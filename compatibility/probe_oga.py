"""Probe one OGA artifact for generation or multi-token hidden-state extraction."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "onnxruntime-genai-directml-ryzenai",
                "onnxruntime-providers-ryzenai",
                "onnxruntime-vitisai",
                "numpy",
            )
        },
    }


def probe_model(model_path: Path, expect_hidden: bool, text: str) -> dict[str, object]:
    import onnxruntime_genai as og

    started = time.perf_counter()
    model = og.Model(str(model_path))
    load_ms = (time.perf_counter() - started) * 1000.0
    tokenizer = og.Tokenizer(model)
    tokens = tokenizer.encode(text)
    params = og.GeneratorParams(model)
    generator = og.Generator(model, params)
    generator.append_tokens(tokens)
    started = time.perf_counter()
    generator.generate_next_token()
    prefill_ms = (time.perf_counter() - started) * 1000.0
    result: dict[str, object] = {
        "model_load_ms": load_ms,
        "prefill_ms": prefill_ms,
        "input_token_count": len(tokens),
        "generated_token_count": len(generator.get_sequence(0)),
        "hidden_states_expected": expect_hidden,
    }
    if expect_hidden:
        hidden = np.asarray(generator.get_output("hidden_states"))
        embedding = hidden[0, len(tokens) - 1].astype(np.float64)
        norm = float(np.linalg.norm(embedding))
        result.update(
            {
                "hidden_states_shape": list(hidden.shape),
                "embedding_dimension": int(embedding.shape[0]),
                "embedding_finite": bool(np.all(np.isfinite(embedding))),
                "embedding_norm_before_normalization": norm,
                "passed": bool(hidden.shape[1] >= len(tokens) and norm > 0),
            }
        )
    else:
        result["passed"] = len(generator.get_sequence(0)) > len(tokens)
    del generator
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe an OGA model artifact")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--expect-hidden", action="store_true")
    parser.add_argument(
        "--text",
        default="What is a neural processing unit and how does it differ from a CPU?",
    )
    parser.add_argument("--environment-only", action="store_true")
    args = parser.parse_args()
    if not args.environment_only and args.model is None:
        parser.error("--model is required unless --environment-only is selected")
    return args


def main() -> None:
    args = parse_args()
    result: dict[str, object] = {"environment": environment()}
    if not args.environment_only:
        result["probe"] = probe_model(args.model, args.expect_hidden, args.text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
