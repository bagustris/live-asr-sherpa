# Supported Speaker Identification Models

`sherox.sid` identifies known speakers using a speaker embedding extractor. It uses one embedding model plus Silero VAD for microphone mode.

## Default Built-In Model

| Purpose | Default Path | Source | Notes |
|--------|--------------|--------|-------|
| Speaker embedding extractor | `models/nemo_en_titanet_large.onnx` | `speaker-recongition-models/nemo_en_titanet_large.onnx` | Auto-downloaded on first use; default for `--model` |
| VAD for `--mic` mode | `models/silero_vad.onnx` | `asr-models/silero_vad.onnx` | Auto-downloaded for microphone segmentation |

## Alternative Embedding Models

Any compatible model from the Sherpa-ONNX speaker recognition release can be used with `--model`.

| Model | Size | Lang | Notes |
|-------|------|------|-------|
| `nemo_en_titanet_large.onnx` | 96 MB | en | Default; highest accuracy |
| `wespeaker_en_voxceleb_resnet293_LM.onnx` | 109 MB | en | WeSpeaker large |
| `wespeaker_en_voxceleb_CAM++_LM.onnx` | 27 MB | en | Good accuracy and speed balance |
| `nemo_en_speakerverification_speakernet.onnx` | 22 MB | en | Lightest option |
| `wespeaker_zh_cnceleb_resnet34_LM.onnx` | 25 MB | zh | Chinese |
| `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | 26 MB | zh/en | Bilingual |

## Examples

Default model:

```bash
sherox.sid --wav audio.wav --speaker-file speakers.txt
```

Custom embedding model:

```bash
sherox.sid --mic \
  --speaker-file speakers.txt \
  --model models/wespeaker_en_voxceleb_CAM++_LM.onnx
```

## Related Options

- `--speaker-file PATH`
- `--model PATH`
- `--threshold FLOAT`
- `--sample-rate`
- `--capture-rate`
- `--chunk-size`

Download alternative models from the [speaker recognition models release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models).
