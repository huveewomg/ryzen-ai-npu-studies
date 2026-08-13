# Publication gate status

Snapshot: 2026-08-13. This file records the implementation of the internal
publication-readiness plan.

| Gate | State | Evidence |
|---|---|---|
| 0: curated snapshot | Complete | `huveewomg/ryzen-ai-npu-studies` contains a reviewed one-commit snapshot rather than the laboratory history. It remains private until the owner approves publication. |
| 1: fail-closed NPU | Complete | Nomic placed 652/656 nodes on VAIML. BGE placed 450/455. The harness rejects missing or zero-node assignment reports. |
| 2: fidelity | Complete | Full official BEIR SciFact passed for Nomic and BGE FP32 CPU/NPU. Dynamic INT8 failed its predeclared gate and is retained as a negative result. |
| 3: staged ablation | Complete with one censored corner | Seventeen Nomic provider/shape groups have five independent processes. CPU batch 32, sequence 512 has one 45-minute timeout and two long-ceiling samples, so it is not a confirmatory estimate. |
| 4: Qwen3 compatibility | Complete negative result; tracker update pending | OGA build and hybrid partition pass on 1.7.1 and 1.8.0 with hidden states off/on. All probes fail with recorded shape or dtype signatures. The reviewed update is in `docs/qwen3-compatibility.md`; posting it back to the laboratory tracker is deferred until public links exist. |
| 5: CI, docs, raw evidence, license | Complete | Results, limitations, CI, schemas, a compact reviewed snapshot, and a deterministic sanitized raw-evidence attachment builder are present. Clean-checkout and secret scans passed, and Apache-2.0 was selected. |

## Publication decision

The embedding study is worth publishing as a one-machine reproducibility note,
not as a general hardware ranking. The strongest claims are the fail-closed NPU
assignment evidence, the full SciFact fidelity results, and the Nomic n=5 shape
matrix. The censored CPU corner, single-process BGE timings, and failed Qwen3
paths must remain visible.

## Repository split

This repository is the reviewed, curated snapshot. The separate private
laboratory repository keeps the full experiment history and failed local ideas
out of the published record.

## Remaining public-release actions

1. Review the private `huveewomg/ryzen-ai-npu-studies` repository.
2. Change its visibility only when the owner explicitly approves publication.
3. Attach the ZIP produced by `python tools/build_raw_release.py` and its SHA-256
   sidecar to the release.
4. Post the reviewed compatibility update back to the laboratory tracker with
   links to the public note and raw attachment.
