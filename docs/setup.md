# Reproducible Ryzen AI Environment Setup

## Version rule

Use an NPU driver and Ryzen AI SDK combination supported by AMD for the target
machine. Record both exact versions for every run. A registered execution
provider is necessary but not sufficient evidence that a model graph executed
on the NPU.

The current embedding study used Ryzen AI 1.8.0 and NPU driver
`32.0.20101.3760`. Ryzen AI 1.7.1 remains installed only for the controlled
decoder compatibility comparison. Compiled caches must not be shared between
SDK or driver versions.

## 1. Verify the hardware

On Windows, confirm that the NPU is present in Device Manager or Task Manager.
Where permissions allow, record the signed driver version:

```powershell
Get-PnpDevice -FriendlyName "*NPU*" |
  Get-PnpDeviceProperty -KeyName DEVPKEY_Device_DriverVersion |
  Select-Object Data
```

Copy the observed value into a local manifest input. Do not guess or reuse the
historical value.

## 2. Activate the SDK environment

Activate the environment installed by the selected Ryzen AI SDK, then verify its
runtime and provider list:

```powershell
python --version
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

For NPU work, the list must include `VitisAIExecutionProvider`. The benchmark
will still reject the run unless VitisAI emits a non-empty operator-assignment
report.

Install only the additional project dependencies into that environment:

```powershell
python -m pip install -r requirements.txt
```

The AMD-provided builds of PyTorch, ONNX Runtime, Transformers, and related
components remain controlled by the SDK. Their installed versions are captured
in each `environment.json` artifact.

## 3. Prepare a local manifest input

Create an untracked `environment.local.json` containing details that Python
cannot obtain reliably without elevated or vendor-specific APIs:

```json
{
  "hardware": {
    "cpu": "AMD Ryzen AI 7 PRO 350",
    "npu": "AMD XDNA2",
    "ram": "64 GB"
  },
  "software": {
    "ryzen_ai_sdk": "<EXACT_VERSION>",
    "npu_driver": "<EXACT_VERSION>",
    "windows_build": "<EXACT_BUILD>"
  },
  "controls": {
    "power_profile": "<PROFILE>",
    "thread_settings": "<ORT_AND_CPU_THREAD_SETTINGS>"
  }
}
```

Do not commit a machine-specific manifest containing usernames, absolute paths,
serial numbers, tokens, or other private identifiers. Reviewed run artifacts
should retain only the fields required to interpret the measurement.

## 4. Choose an NPU cache directory

The tested toolchain rejects some cache paths containing spaces. Pass a local,
space-free directory explicitly with `--cache-dir`. The path itself is not saved
in publication JSON; only sanitized reports copied into the run directory are
retained.

Never reuse a compiled cache across:

- different ONNX hashes;
- different Ryzen AI SDK or VitisAI EP versions;
- different NPU driver versions.

## 5. Confirm fail-closed behavior

Before collecting results, run the unit regression proving that unavailable NPU
providers are rejected:

```powershell
python -m unittest tests.test_evidence -v
```

For an actual NPU run, use `--require-npu`. The harness enables AMD's operator
assignment report, requests disk cache/report output, runs inference, parses the
report, and rejects zero NPU-assigned nodes.

## Benchmark controls

Before every process group:

- record the active power profile and whether the machine is connected to AC;
- record CPU/ORT thread settings;
- close or list material background workloads;
- keep the device on the same surface and in the same thermal conditions;
- use the prescribed 20 warmups and 100 timed iterations;
- do not infer power or thermal behavior from latency alone.

The study currently makes no thermal-throttling, power, or energy claim because
no calibrated measurement method has been selected.

## Troubleshooting classifications

Classify failures precisely:

| Observation | Classification |
|---|---|
| VitisAI not in `get_available_providers()` | provider unavailable |
| Session excludes VitisAI | provider activation failure/fallback |
| Assignment report missing | unverifiable NPU run |
| Report has zero NPU nodes or subgraphs | CPU-only or unverifiable graph assignment |
| Session creation fails | compile/partition failure |
| Session runs but fidelity fails | numerical failure |
| NPU nodes exist but performance regresses | valid but slower tested configuration |

Do not relabel any of these as successful NPU execution without the missing
evidence.
