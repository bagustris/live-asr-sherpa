# Supported Diarization Models

`sherox.asr --diarization` uses a two-model diarization pipeline:

- a speaker segmentation model
- a speaker embedding extractor

If you do not pass custom paths, both are auto-downloaded on first use.

## Default Models

| Purpose | Default Path | Source Archive / File | Notes |
|--------|--------------|------------------------|-------|
| Segmentation | `models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx` | `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` | Auto-downloaded and extracted |
| Embedding | `models/nemo_en_speakerverification_speakernet.onnx` | `nemo_en_speakerverification_speakernet.onnx` | Auto-downloaded standalone ONNX file |

Approximate sizes from the README:

- segmentation: about 7 MB
- embedding: about 23 MB

## CLI Options

You can override either model explicitly:

```bash
sherox.asr --mic --offline --diarization \
  --diarization-seg-model /path/to/model.onnx \
  --diarization-emb-model /path/to/embedding.onnx
```

Relevant flags:

- `--diarization`
- `--diarization-seg-model PATH`
- `--diarization-emb-model PATH`
- `--num-speakers N`
- `--speaker-tag`

## Notes

- Diarization works with both online and offline ASR pipelines in this repo.
- The diarization pipeline is built from pyannote segmentation plus a speaker embedding extractor.
- When `--num-speakers` is omitted, clustering uses auto-detection.
