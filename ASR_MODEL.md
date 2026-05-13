# Supported ASR Models

All models from the [Sherpa-ONNX model zoo](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/) can be used with `sherox.asr`. Download and extract a model into the `models/` directory, then pass the directory via `--model-dir` and the architecture via `--model-type`.

> [!TIP]
> Models marked **auto** are downloaded automatically on first run. All others must be downloaded manually from the [Sherpa-ONNX releases page](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models).

## Online (Streaming) Models

Use these with the default pipeline and do not pass `--offline`. They support real-time partial hypotheses.

| Model | `--model-dir` | `--model-type` | Lang | Notes |
|-------|--------------|----------------|------|-------|
| Zipformer En 2023 | `models/zipformer-en-2023` | *(blank)* | en | Online streaming; **auto-downloaded when selected** |
| Zipformer En 2024 | `models/sherpa-onnx-streaming-zipformer-en-2024-02-13` | `zipformer2` | en | Newer, slightly higher accuracy |
| Conformer En | `models/sherpa-onnx-streaming-conformer-en-2023-05-09` | `conformer` | en | Conformer transducer |
| Zipformer ZH/EN | `models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` | `zipformer` | zh/en | Bilingual |
| Paraformer ZH/EN | `models/sherpa-onnx-streaming-paraformer-bilingual-zh-en` | `paraformer` | zh/en | Streaming paraformer |
| WeNet CTC En | `models/sherpa-onnx-streaming-wenet-librispeech` | `wenet_ctc` | en | WeNet CTC |
| Zipformer2 CTC En | `models/sherpa-onnx-streaming-zipformer2-ctc-2024-09-18` | `zipformer2_ctc` | en | CTC variant |
| Multilingual Streaming Zipformer | `models/zipformer-multilingual-2025-02-10` | `multilingual_streaming` | ar/en/id/ja/ru/th/vi/zh | **Auto-downloaded when selected**; built-in alias for `sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10` |

Example:

```bash
sherox.asr --mic \
  --model-dir models/sherpa-onnx-streaming-zipformer-en-2024-02-13 \
  --model-type zipformer2
```

Built-in multilingual example:

```bash
sherox.asr --mic --model-type multilingual_streaming
```

## Offline Models

Use these with `--offline`. Audio is VAD-segmented before recognition, which usually improves accuracy at the cost of latency. A [Silero VAD](https://github.com/snakers4/silero-vad) model (`silero_vad.onnx`) is auto-downloaded when needed.

| Model | `--model-dir` | `--model-type` | Lang | Notes |
|-------|--------------|----------------|------|-------|
| Parakeet TDT 0.6B FP16 | `models/parakeet-tdt-0.6b-v2` | `nemo_transducer` | en | **Auto-downloaded** when selected; larger, more accurate |
| Parakeet TDT 0.6B INT8 | `models/parakeet-tdt-0.6b-v2-int8` | `nemo_transducer` | en | Default English; **auto-downloaded**; smaller and faster |
| Whisper tiny.en | `models/sherpa-onnx-whisper-tiny.en` | `whisper` | en | Smallest Whisper |
| Whisper base.en | `models/sherpa-onnx-whisper-base.en` | `whisper` | en | |
| Whisper small.en | `models/sherpa-onnx-whisper-small.en` | `whisper` | en | Good accuracy and speed balance |
| Whisper medium.en | `models/sherpa-onnx-whisper-medium.en` | `whisper` | en | Higher accuracy |
| Whisper large-v3 | `models/sherpa-onnx-whisper-large-v3` | `whisper` | multi | Multilingual; use `--language` |
| Paraformer ZH | `models/sherpa-onnx-paraformer-zh-2023-09-14` | `paraformer` | zh | |
| NeMo CTC En | `models/sherpa-onnx-nemo-ctc-en-conformer-medium` | `nemo_ctc` | en | NeMo Conformer CTC |
| SenseVoice | `models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` | `sense_voice` | multi | 5 languages; use `--language` |
| Moonshine tiny | `models/sherpa-onnx-moonshine-tiny-en-int8` | `moonshine` | en | Very fast, English only |
| Moonshine base | `models/sherpa-onnx-moonshine-base-en-int8` | `moonshine` | en | Better accuracy than tiny |
| FireRedASR | `models/sherpa-onnx-fire-red-asr-large-zh-2025-02-16` | `fire_red_asr` | zh | |
| ReazonSpeech JA | `models/reazonspeech-ja` | `ja` | ja | **Auto-downloaded**; Japanese |
| ReazonSpeech JA-EN | `models/reazonspeech-ja-en` | `ja-en` | ja/en | **Auto-downloaded**; bilingual |
| ReazonSpeech JA-EN-MLS | `models/reazonspeech-ja-en-mls-5k` | `ja-en-mls-5k` | ja/en | **Auto-downloaded**; bilingual plus MLS 5k |
| NeMo Parakeet CTC JA | `models/parakeet-ctc-ja-int8` | `parakeet-ctc-ja` | ja | **Auto-downloaded**; Japanese 0.6B int8 CTC |
| Cohere Transcribe 14-Lang | `models/cohere-transcribe-14-lang-int8` | `cohere_transcribe` | multi | 14 languages; multilingual ASR |

Examples:

```bash
# Parakeet TDT (auto-downloaded offline default)
sherox.asr --mic --offline --model-type nemo_transducer

# Parakeet TDT INT8 (smaller, auto-downloaded)
sherox.asr --mic --offline \
  --model-dir models/parakeet-tdt-0.6b-v2-int8 \
  --model-type nemo_transducer

# Whisper small (English)
sherox.asr --mic --offline \
  --model-dir models/sherpa-onnx-whisper-small.en \
  --model-type whisper

# Whisper large-v3 (multilingual)
sherox.asr --mic --offline \
  --model-dir models/sherpa-onnx-whisper-large-v3 \
  --model-type whisper --language zh

# SenseVoice (5 languages)
sherox.asr --mic --offline \
  --model-dir models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
  --model-type sense_voice --language ja

# Moonshine tiny
sherox.asr --mic --offline \
  --model-dir models/sherpa-onnx-moonshine-tiny-en-int8 \
  --model-type moonshine

# ReazonSpeech Japanese (auto-downloaded)
sherox.asr --mic --model-type ja

# ReazonSpeech bilingual Japanese-English
sherox.asr --mic --model-type ja-en

# ReazonSpeech bilingual + MLS 5k English
sherox.asr --wav audio.wav --model-type ja-en-mls-5k

# NeMo Parakeet CTC Japanese
sherox.asr --wav audio.wav --model-type parakeet-ctc-ja

# Cohere Transcribe multilingual (14 languages)
sherox.asr --mic --offline --model-type cohere_transcribe --language en

# Cohere Transcribe with different language (e.g. Chinese)
sherox.asr --wav audio.wav --offline --model-type cohere_transcribe --language zh
```

## Supported `--model-type` Values

Online:

- `(blank)`
- `transducer`
- `zipformer`
- `zipformer2`
- `conformer`
- `lstm`
- `paraformer`
- `ctc`
- `wenet_ctc`
- `zipformer2_ctc`
- `multilingual_streaming`

Offline:

- `(blank)`
- `transducer`
- `nemo_transducer`
- `paraformer`
- `whisper`
- `ctc`
- `nemo_ctc`
- `sense_voice`
- `moonshine`
- `fire_red_asr`
- `cohere_transcribe`
- `ja`
- `ja-en`
- `ja-en-mls-5k`
- `parakeet-ctc-ja`
