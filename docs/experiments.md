# Historical Experiment Log

## Provenance and interpretation

The measurements below were recorded on **2026-06-14** on one Ryzen AI 7
PRO 350 laptop using Ryzen AI 1.7.1. The commits were assembled into a curated
release branch later. This document preserves the observations as historical
baseline data; it does not backdate a public release, claim first discovery, or
present the values as a current/general benchmark.

The original raw per-iteration files and complete environment manifest were not
committed. Consequently, the latency table cannot pass the publication gates.
The current harness exists to reproduce or reject these observations with raw
artifacts, numerical validation, and fail-closed NPU evidence.

## Experiment 1: `nomic-embed-text-v1.5`

### Historical model and environment

| Property | Recorded value |
|---|---|
| Model | `nomic-ai/nomic-embed-text-v1.5` |
| Architecture | NomicBERT |
| Parameters | 137M |
| Embedding dimension | 768 |
| Exported shape | batch 1, sequence 128 |
| ONNX opset | 17 |
| CPU | AMD Ryzen AI 7 PRO 350 (8C/16T) |
| NPU | AMD XDNA2 |
| RAM | 64 GB |
| NPU driver | 32.0.20101.3760 |
| SDK | Ryzen AI 1.7.1 |

The model revision and model-file hash were not recorded in the historical run.
The rerun pins both and stores them with each artifact.

### Historical ONNX export

The model was exported at a fixed `[1, 128]` input shape. A single sample
comparison reported PyTorch/ONNX cosine similarity `1.000000` and maximum
absolute difference `2.38e-07`. This confirmed that one export example matched;
it did not establish corpus-wide fidelity.

### Historical VitisAI assignment

The compiler report saved under `<NPU_CACHE_DIR>` recorded:

| Metric | Recorded value |
|---|---:|
| Operators total | 656 |
| Operators assigned to NPU | 652 (99.4%) |
| Operators assigned to CPU | 4 (0.6%) |
| Reported GOPs | 30.2 |
| NPU subgraphs | 1 |
| Reported device precision | BF16 |
| First compilation time | approximately 5 minutes |

These values are useful historical evidence, but they must be regenerated for
each publication SDK/driver combination. Merely seeing
`VitisAIExecutionProvider` in a session does not establish operator assignment.

### Historical quantization attempts

#### AMD Quark attempt

The then-installed Quark API did not accept the attempted configuration, and
custom-op compilation also required a working MSVC toolchain. No Quark artifact
was used in the successful latency matrix.

#### Static INT8 attempts

Percentile calibration failed during histogram processing. MinMax static
quantization produced large changes in query-document similarity on the three
hand-written examples:

| Static configuration | Largest recorded similarity change |
|---|---:|
| MinMax, per-channel | 0.2070 |
| MinMax, per-tensor with preprocessing | 0.2095 |

This showed that those particular artifacts failed the small smoke check. It
does not establish that static quantization is unsuitable for all embedding
models or identify a proven causal mechanism.

#### ONNX Runtime dynamic INT8 attempt

The successful INT8 artifact used
`onnxruntime.quantization.quantize_dynamic(..., weight_type=QInt8)`. It was not
calibrated and did not use AMD Quark. Weights were quantized while activations
remained floating point at runtime.

| Pair | FP32 similarity | Dynamic INT8 similarity | Absolute change |
|---|---:|---:|---:|
| NPU query / related document | 0.5689 | 0.5963 | 0.0274 |
| Quantization query / related document | 0.7950 | 0.8009 | 0.0059 |
| NPU query / unrelated weather document | 0.3881 | 0.3870 | 0.0011 |

This three-pair result was a smoke check only. “Rank order preserved” on these
examples is not sufficient deployment or retrieval-quality evidence. The new
fidelity study compares per-item embeddings and NanoSciFact retrieval metrics.

### Historical latency table

The historical script used one warmup followed by 20 iterations with synthetic
all-ones token IDs at fixed batch 1 and sequence 128.

| Configuration | Mean latency | Derived throughput | Versus FP32 CPU |
|---|---:|---:|---:|
| FP32 CPU | 68.8 ms | 14.5 docs/s | 1.00x |
| FP32 VitisAI | 53.2 ms | 18.8 docs/s | 1.29x |
| Dynamic INT8 CPU | 24.4 ms | 41.0 docs/s | 2.82x |
| Dynamic INT8 VitisAI | 47.5 ms | 21.1 docs/s | 1.45x |

### What the historical run supports

- On this recorded setup and shape, FP32 VitisAI had a lower reported mean
  latency than the tested FP32 CPU configuration.
- On the same setup and shape, dynamic INT8 CPU had the lowest reported mean
  latency of the four configurations.
- The retained VitisAI report assigned most FP32 graph operators to the NPU.
- The tested dynamic INT8 artifact changed the three smoke-pair similarities
  less than the unsuccessful static artifacts.

### What it does not support

- a current Ryzen AI SDK performance claim;
- a model-size break-even point between CPU and NPU;
- a general claim about static or dynamic quantization quality;
- production retrieval quality;
- power, energy, or thermal behavior;
- results for other machines, models, batches, sequence lengths, or CPU thread
  configurations.

## Rerun protocol

The Ryzen AI 1.8 study applied the following protocol. Its current results and
limitations are in [`docs/results.md`](results.md).

1. Pin the model revision and hash every ONNX artifact.
2. Use a fresh VitisAI assignment report for every NPU-labelled configuration.
3. Pass PyTorch/ONNX/NPU/INT8 numerical and retrieval-fidelity thresholds.
4. Test batch sizes 1/8/32 and sequence lengths 32/128/512.
5. Use at least five independent processes per cell, with 20 warmups and 100
   timed iterations in each process.
6. Separate inference-only latency from tokenizer-inclusive latency.
7. Record raw samples, P50/P95/P99, mean, standard deviation, throughput,
   environment controls, and worst-case fidelity examples.
8. Repeat selected corners with one additional supported BERT-family model.

No historical result should be silently replaced. New tables must link to their
raw artifact directory and state the exact SDK/driver version.

## Ryzen AI 1.8 pilot outcome (2026-08-13)

The fresh fixed-shape batch-1/sequence-128 FP32 model reproduced the NPU path
under Ryzen AI 1.8. Its assignment report contained 656 nodes: 652 assigned to
VAIML, four to CPU, and one NPU subgraph. The cold content-addressed compilation
took 393.2 seconds. In the single exploratory process, warmed inference averaged
33.14 ms on NPU and 94.48 ms on CPU. These are pilot values, not the final
multi-process estimates.

The complete NanoSciFact gate passed for FP32 CPU and FP32 NPU. FP32 NPU had a
worst per-item cosine of 0.997559 against PyTorch and preserved Recall@10 at
0.86. Per-channel dynamic INT8 CPU failed: worst cosine was 0.970483 and
Recall@10 fell to 0.81. The per-channel INT8 VitisAI report assigned zero nodes
to the NPU. The failed artifacts and thresholds remain preserved locally;
dynamic INT8 will not be presented as an NPU performance configuration.
