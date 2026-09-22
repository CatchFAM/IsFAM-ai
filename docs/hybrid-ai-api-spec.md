# IsFAM Hybrid AI Contract

## Target architecture

```text
Android
  - ONNX speaker embedding inference (implemented in isfam-app; device validation pending)
  - encrypted family voiceprint storage (implemented with Android Keystore)
  - local cosine-similarity decision (implemented; threshold 0.65)
  - uploads only the call-audio segment needed for deepvoice detection

FastAPI
  - 196-feature spectral MLP deepvoice detection (implemented)
  - audio normalization and quality measurement (implemented)
  - CPU inference, startup warm-up, and Transformers fallback (implemented)
```

For a deepvoice-only deployment, disable speaker-model preloading so the server does not spend
memory on the model moved to Android:

```bash
ISFAM_PRELOAD_MODELS=true
ISFAM_PRELOAD_SPEAKER_MODEL=false
```

The Java product server is outside this repository. At commit `669af54`, it contains the
`isfam.fastapi.base-url` setting but no controller/client that proxies this request. Until that
proxy is implemented, an Android client configured with the Java server URL cannot reach the
deepvoice endpoint. The proxy should preserve the audio field and response body below.

## Deepvoice detection API

### Model information

```http
GET /api/v1/anti-spoofing/model-info
```

Returns readiness, model name/version, device, threshold, window settings, and batch size. The
deployment version is configured with `ISFAM_ANTI_SPOOFING_MODEL_VERSION`.

The default backend is the packaged `isfam/spectral-mlp-dfadd-v1` model. Set
`ISFAM_ANTI_SPOOFING_BACKEND=transformers` to use the previous Hugging Face model.

### Request

```http
POST /api/v1/anti-spoofing/detect
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|---|---|---:|---|
| `audio_file` | file | yes | `wav`, `mp3`, or `m4a`; maximum 25 MB |

The server converts the upload to mono 16 kHz PCM WAV before inference. A 3–5 second speech
segment is recommended. The spectral backend analyzes at most the center 5 seconds; callers
should send consecutive call chunks rather than one long recording.

### Success response (`200`)

```json
{
  "analysis_status": "complete",
  "processing_time_ms": 32.85,
  "is_spoofed": false,
  "spoof_score": 0.2817,
  "threshold": 0.5107,
  "predicted_label": "real",
  "predicted_score": 0.7183,
  "message": "bonafide",
  "model_name": "isfam/spectral-mlp-dfadd-v1",
  "analyzed_segments": 1,
  "max_spoof_segment_index": 0,
  "segment_seconds": 5.0,
  "label_scores": [
    {"label": "real", "score": 0.7183},
    {"label": "fake", "score": 0.2817}
  ],
  "audio_quality": {
    "is_analyzable": true,
    "message": "analyzable",
    "duration_seconds": 5.0,
    "rms_energy": 0.08,
    "peak_amplitude": 0.72,
    "speech_ratio": 0.84
  }
}
```

`analysis_status=more_voice_required` means the model was run but the audio was too short,
quiet, or speech-sparse for a reliable final decision. The client should collect another
segment; it must not treat the raw `is_spoofed` value as a confirmed warning in that case.

### Error responses

| Status | Meaning |
|---:|---|
| `400` | Missing or too-short file |
| `413` | File exceeds the configured size limit |
| `415` | Unsupported extension |
| `422` | File cannot be decoded or inspected |
| `500` | Model loading or inference failed |

## Android contract

The on-device family verifier should expose this logical result to the app UI:

```json
{
  "is_registered_family": true,
  "best_family_id": "local-family-id",
  "similarity": 0.81,
  "threshold": 0.65,
  "model_version": "ecapa-onnx-v1"
}
```

The Kotlin implementation now lives in the separate `isfam-app` repository. It includes the
INT8 ECAPA asset, ONNX Runtime, SpeechBrain-compatible FBank preprocessing, local cosine
matching, Keystore-backed AES-GCM storage, and a hybrid result combiner. JVM golden tests and
debug APK packaging pass. Real-device ONNX execution, latency, memory, battery, and threshold
calibration are still required before release.

## Speaker ONNX proof of concept

The ECAPA neural core is now reproducibly exportable:

```bash
.venv/bin/pip install -r requirements-onnx.txt
HF_HUB_OFFLINE=1 .venv/bin/python scripts/export_speaker_onnx.py
HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate_speaker_onnx.py
```

Generated model (git-ignored):

```text
models/onnx/ecapa_tdnn_voiceprint.onnx
input  = float32[batch, frames, 80]
output = float32[batch, 192]
size   = 79.61 MB
```

Validation on the current 10 files and 45 speaker pairs produced 100% PyTorch/ONNX
decision agreement. See `reports/speaker_onnx_validation.md` for numerical differences and
timings.

Raw-waveform export is not complete. SpeechBrain's STFT frontend currently fails the ONNX
export path because it uses complex tensor operations. Android must reproduce this frontend:

```text
mono PCM float32, 16 kHz
-> 25 ms window / 10 ms hop / 400-point FFT
-> 80-bin FBank
-> per-utterance mean normalization (no standard-deviation normalization)
-> ECAPA ONNX model
-> 192-value voiceprint
```

The macOS CPU PoC measured the ONNX neural core slower than the current PyTorch full pipeline,
so no mobile speed claim is made yet. Android real-device latency, memory, battery use, and
frontend equivalence must be measured before shipping.

The optional INT8 export is 20.52 MB (74.2% smaller than FP32) and retained 45/45 decisions on
the current dataset. Its maximum pair-similarity delta was 0.01781, so threshold calibration
and Android real-device measurements are required before choosing it over FP32.
