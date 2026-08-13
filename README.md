# Ryzen AI NPU studies

Reproducible experiments on AMD Ryzen AI NPUs.

The first study measures BERT-family text embeddings on an XDNA2 NPU with
Ryzen AI 1.8. It also records a small Qwen3 compatibility matrix across Ryzen
AI 1.7.1 and 1.8.0.

## Results

- Nomic NPU inference was 1.77x to 6.48x faster than CPU across eight tested
  shapes. Each confirmatory result uses five independent CPU and NPU
  processes.
- Full BEIR SciFact fidelity passed for Nomic and BGE with verified VAIML node
  assignment.
- The tested dynamic INT8 candidates failed the predeclared fidelity or NPU
  assignment gates, so they are kept as negative results.
- All four Qwen3 probes built and reached hybrid partitioning, then failed at
  generation. This is a compatibility comparison, not a 1.7 versus 1.8
  performance benchmark.

CPU batch 32, sequence 512 is censored and excluded from the speedup range.
See [the full results](docs/results.md) for confidence intervals, fidelity
scores, failure signatures, and limits.

The reviewed tables, figures, and manifest are in
[`benchmarks/results/published/rai180-20260813`](benchmarks/results/published/rai180-20260813).

## Test machine

| Component | Configuration |
|---|---|
| CPU | AMD Ryzen AI 7 PRO 350 (8C/16T) |
| NPU | AMD XDNA2 |
| iGPU | Radeon 860M |
| RAM | 64 GB |
| OS | Windows 11 Pro 10.0.26200 |
| SDK | Ryzen AI 1.8.0 |
| NPU driver | 32.0.20101.3760 |

## Reproduce the study

Use a Ryzen AI environment that matches the installed NPU driver.

```powershell
python -m pip install -r requirements.txt
python evaluation/fetch_nanoscifact.py
python models/build_nomic_matrix.py
```

Run fidelity checks with the exported batch-1, sequence-128 models:

```powershell
python evaluation/run_fidelity.py `
  --fp32-model models/nomic-embed-v1.5_b1_seq128.onnx `
  --int8-model models/nomic-embed-v1.5_b1_seq128_int8_per_channel.onnx `
  --npu-precisions fp32 `
  --cache-dir <SPACE_FREE_NPU_CACHE_DIR> `
  --manifest-input environment.local.json
```

Run the CPU and NPU matrix:

```powershell
python benchmarks/run_matrix.py `
  --texts-file evaluation/data/nanoscifact/corpus.jsonl `
  --cache-dir <SPACE_FREE_NPU_CACHE_DIR> `
  --manifest-input environment.local.json
```

Each model shape is exported separately. NPU runs fail unless VitisAI reports
real NPU-assigned nodes and inference completes. Local models, compiled caches,
raw embeddings, and machine-specific files stay out of Git.

## Validate the repository

```powershell
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
python tools/validate_results.py
git diff --check
```

Public CI covers linting, unit tests, CPU inference and fidelity smoke tests,
result schemas, and Python compilation. It does not require NPU hardware.

## Limits

- Results come from one laptop and one software configuration.
- CPU timings include some background-activity noise from normal light use.
- Static shapes, driver versions, power settings, and runtime builds can change
  performance.
- NanoSciFact is small and domain-specific.
- This study makes no power, energy, or thermal claims.

## More detail

- [Results and interpretation](docs/results.md)
- [Qwen3 compatibility matrix](docs/qwen3-compatibility.md)
- [Setup notes](docs/setup.md)
- [Experiment history](docs/experiments.md)
- [Publication review](docs/publication-status.md)

## License

Licensed under the [Apache License 2.0](LICENSE).
