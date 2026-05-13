# Supported Segmentation Models

`sherox.segment` performs VAD-based speech segmentation. It currently supports two VAD families:

- Silero VAD
- Ten-VAD

Both expect 16 kHz audio by default and are used to split audio into timestamped speech regions.

## Available Models

| `--vad-model` | Variant | Local Path | Source | Notes |
|--------------|---------|------------|--------|-------|
| `silero` | `silero_vad.onnx` | `models/silero_vad.onnx` | `asr-models/silero_vad.onnx` | Default |
| `ten-vad` | `ten-vad.int8.onnx` | `models/ten-vad.int8.onnx` | `asr-models/ten-vad.int8.onnx` | Default Ten-VAD variant |
| `ten-vad` | `ten-vad.onnx` | `models/ten-vad.onnx` | `asr-models/ten-vad.onnx` | Full-precision Ten-VAD variant |

## Examples

Default Silero:

```bash
sherox.segment --mic
```

Ten-VAD:

```bash
sherox.segment --mic --vad-model ten-vad
```

Explicit Ten-VAD variant:

```bash
sherox.segment --wav audio.wav --vad-model ten-vad --ten-vad-model ten-vad.onnx
```

## Related Options

- `--vad-model {silero,ten-vad}`
- `--ten-vad-model {ten-vad.onnx,ten-vad.int8.onnx}`
- `--threshold`
- `--min-silence`
- `--min-speech`
- `--sample-rate`
- `--capture-rate`
