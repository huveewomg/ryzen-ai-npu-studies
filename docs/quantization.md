# Quantization Notes for the Nomic Study

## Scope

The release-candidate Nomic pipeline uses **ONNX Runtime dynamic weight-only
INT8 quantization**. It does not use AMD Quark, activation calibration, or a
static QDQ transform. Other quantization methods remain valid research options,
but their results must be named and validated separately.

## Core mapping

Integer quantization represents a floating-point value using a scale and zero
point:

```text
quantized = round(real / scale) + zero_point
reconstructed = (quantized - zero_point) * scale
```

The reconstructed value is approximate. Whether the error is acceptable depends
on the model, transform, data, runtime, and downstream metric; it cannot be
decided from model size or a few similarity examples alone.

## Dynamic weight-only INT8 used here

`models/quantize_nomic.py` calls:

```python
quantize_dynamic(
    model_input=preprocessed_source,
    model_output=destination,
    weight_type=QuantType.QInt8,
    per_channel=True,
)
```

This reduces supported model weights to INT8 while runtime activations remain
floating point. No calibration dataset is read. The generated artifact records:

- source and output SHA-256 hashes;
- the pinned Hugging Face model revision;
- quantization method and weight type;
- ONNX Runtime preprocessing and the per-channel setting;
- installed package versions;
- a five-item direct embedding-drift smoke check.

That smoke check can catch a badly corrupted artifact, but it cannot approve the
model for retrieval use. Publication requires the separate NanoSciFact fidelity
run.

## Ryzen AI 1.8 pilot decision

The pinned batch-1/sequence-128 pilot kept the acceptance thresholds fixed and
tested two dynamic QInt8 candidates:

| Candidate | Smoke minimum cosine | NanoSciFact | VitisAI assignment |
|---|---:|---|---|
| Per-tensor | 0.967809 | Not advanced | Not advanced |
| Per-channel | 0.983720 | Failed: minimum cosine 0.970483 and Recall@10 0.81 versus 0.86 | Failed: zero NPU nodes |

The per-tensor candidate failed the predeclared 0.98 smoke threshold. The
per-channel candidate passed the smoke threshold, but it failed the stronger
corpus gate and the VitisAI report assigned 884 nodes to CPU, one node to
`VITIS_EP_CPU`, and none to the NPU. Consequently, dynamic INT8 is excluded
from the publication performance matrix and reported as a negative result.

## Static QDQ quantization

Static quantization estimates activation ranges from representative calibration
data and normally inserts QuantizeLinear/DequantizeLinear boundaries around
supported operations. Relevant choices include:

| Choice | Description | Validation concern |
|---|---|---|
| MinMax | Covers observed extrema | Sensitive to outliers |
| Percentile | Clips a chosen tail | Percentile and corpus must be recorded |
| Entropy | Chooses ranges by distribution criterion | More computation and configuration |
| Per-tensor | One scale for a tensor | May lose channel-specific resolution |
| Per-channel | Separate weight scale by channel | Runtime/operator support varies |

There is no project-wide rule that percentile, per-channel, or static QDQ is
always best for an embedding model or for VitisAI. Each artifact needs a pinned
configuration, a representative licensed corpus, numerical comparison, operator
assignment evidence, and retrieval metrics.

## Historical attempts

The June 2026 experiments tried several static ONNX Runtime transforms. One
percentile attempt crashed in histogram processing. Two MinMax artifacts changed
query-document similarity by approximately 0.21 on one of three hand-written
pairs. Those artifacts failed the smoke check and were discarded.

The observed failures do not prove that attention accumulation was the cause or
that all static quantization will fail. They only describe the tested artifacts.
See `docs/experiments.md` for the preserved numbers and limitations.

## Required validation ladder

Every quantized publication artifact must pass the following comparisons using
the same tokenizer, prefixes, padding, truncation, pooling, and normalization:

1. PyTorch FP32 to ONNX FP32 CPU.
2. ONNX FP32 CPU to verified VitisAI NPU.
3. PyTorch FP32 to dynamic INT8 CPU.
4. PyTorch FP32 to verified dynamic INT8 NPU.

For each candidate, record:

- per-item embedding cosine;
- mean and minimum cosine;
- maximum absolute error;
- the worst item IDs;
- Recall@1/5/10;
- nDCG@10;
- mean ranking Spearman correlation against PyTorch FP32.

The project-specific pre-result thresholds are committed in
`evaluation/acceptance.json`. They are guardrails for this study, not general
quality standards. A failed threshold must remain visible and be investigated;
it must not be relaxed after looking at results without a documented review.

## NPU execution is a separate question

A numerically valid ONNX artifact may still execute mostly or entirely on CPU.
Conversely, successful NPU assignment does not prove numerical correctness.
Publication therefore requires both:

- VitisAI provider and operator-assignment evidence with non-zero NPU nodes;
- numerical and retrieval fidelity against the defined FP32 reference.

The benchmark harness enables AMD's `vitisai_ep_report.json`, saves it alongside
the run, and fails non-zero when `--require-npu` cannot validate it.

## Artifact naming

Fixed-shape files encode their shape and method:

```text
nomic-embed-v1.5_b1_seq128.onnx
nomic-embed-v1.5_b1_seq128_int8_per_channel.onnx
```

Do not reuse a compiled VitisAI cache across model hashes, SDK versions, or NPU
driver versions. The final report must cite the raw artifact directory rather
than copying numbers by hand.
