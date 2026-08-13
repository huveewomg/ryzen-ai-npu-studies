# Qwen3 decoder-embedding compatibility

This note updates the historical Ryzen AI 1.7.1 laboratory finding with a
controlled four-cell retest on the same Ryzen AI 7 PRO 350 laptop. It is a
version-bound compatibility result, not a claim that all decoder embedding
models or AMD NPU paths fail.

## Controlled matrix

The pinned model was `Qwen/Qwen3-Embedding-0.6B` revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Each cell built an FP16 OGA
DirectML model, passed it through AMD's hybrid partitioner, and then ran the same
multi-token generation probe. `include_hidden_states` was the only setting
changed within each SDK pair.

| Ryzen AI | OGA | VitisAI ORT | Hidden states | OGA build | Hybrid partition | Probe |
|---|---|---|---:|---|---|---|
| 1.7.1 | 0.11.2 | 1.23.3 | off | pass | pass | fail: fixed-shape logits mismatch |
| 1.7.1 | 0.11.2 | 1.23.3 | on | pass | pass | fail: fixed-shape logits mismatch |
| 1.8.0 | 0.14.0 | 1.27.0 | off | pass | pass | fail: hybrid-model dtype rejection |
| 1.8.0 | 0.14.0 | 1.27.0 | on | pass | pass | fail: hybrid-model dtype rejection |

The package version is
`onnxruntime-genai-directml-ryzenai==0.11.2` on 1.7.1 and
`onnxruntime-genai-directml-ryzenai==0.14.0` on 1.8.0. The Ryzen AI 1.8.0
provider package is `onnxruntime-providers-ryzenai==1.8.0`.

## Minimal failure signatures

Ryzen AI 1.7.1 generated logits with shape `{1,1,151669}` where the runtime
requested `{1,15,151669}`:

```text
OrtValue shape verification failed.
Current shape:{1,1,151669} Requested shape:{1,15,151669}
```

Ryzen AI 1.8.0 completed hybrid partitioning but rejected the hybrid graph while
loading it:

```text
Type Error: Type 'tensor(float16)' of input parameter
(/model/embed_tokens/Gather/output_0) of operator
(SkipSimplifiedLayerNormalizationBf) in node
(/model/layers.0/post_attention_layernorm/SkipLayerNorm) is invalid.
```

Turning hidden states on did not change the stage reached or the failure class in
either SDK. The 1.7.1 failure moved past the original partitioner assertion, and
1.8.0 changed the terminal error from a shape mismatch to a dtype rejection, but
neither tested path produced a usable multi-token embedding.

## Reproduction boundary

This was a controlled FP16 OGA DirectML-to-hybrid test because it was executable
on the Windows-only machine. It was not AMD's canonical AWQ flow, did not retest
Jina v5 specifically, and did not establish a result for every raw decoder ONNX
route. Those untested paths must not be described as failures.

The compact matrix is
[`qwen-compatibility.json`](../benchmarks/results/published/rai180-20260813/qwen-compatibility.json).
The sanitized raw release attachment contains each cell's environment, build,
partition, and probe records and minimal logs; generated model binaries and
external weight data are deliberately excluded.
