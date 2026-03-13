# Live ASR with Sherpa-ONNX

*Minimal, production-quality streaming speech recognition on CPU*

A terminal-based Automatic Speech Recognition (ASR) application built with [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx). Transcribe speech in real-time from your microphone or offline from WAV files — no GPU required.

## Features

- **Real-time microphone transcription** — partial hypotheses update live with <500 ms latency
- **Offline WAV transcription** — process audio files through the same pipeline
- **Speaker diarization** — colour-coded per-speaker output; ASR and diarization run concurrently to keep latency low
- **Unified model loading** — all sherpa-onnx model families supported via a single `--model-type` flag
- **CPU-optimized** — runs efficiently on any modern CPU using ONNX Runtime
- **Auto model download** — fetches the default Zipformer model on first run; diarization models also auto-downloaded
- **Endpoint detection** — intelligently segments speech with configurable silence rules
- **Rich terminal output** — colour-coded speaker labels and styled status messages via the `rich` library

## Prerequisites

- Python 3.8+
- A working microphone (for `--mic` mode)
- Linux recommended (Ubuntu compatible)

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `rich` is included as a dependency for styled terminal output.

### 2. Run

**Microphone mode** — stream and transcribe live audio:

```bash
python3 src/main.py --mic
```

**WAV file mode** — transcribe a pre-recorded file:

```bash
python3 src/main.py --wav path/to/audio.wav
```

> [!NOTE]
> WAV files must be **mono, 16-bit, 16 kHz**. Convert with:
> ```bash
> ffmpeg -i input.wav -ar 16000 -ac 1 output.wav
> ```

The default Zipformer model (~300 MB) is downloaded automatically on first run.

## Speaker Diarization

Add `--diarization` to any command to colour-code the transcript by speaker. Two lightweight models are downloaded automatically on first use (~7 MB segmentation + ~23 MB embedding):

```bash
# Offline with diarization (auto-downloads all models)
python3 src/main.py --mic --offline --diarization

# If you know how many speakers will be present:
python3 src/main.py --mic --offline --diarization --num-speakers 2

# WAV file with diarization
python3 src/main.py --wav meeting.wav --offline --diarization --num-speakers 3
```

Each speaker's transcript is printed in a distinct colour:

```
  [Speaker 00] Good morning everyone.
  [Speaker 01] Thanks for joining the call.
  [Speaker 00] Let's get started.
```

Diarization and ASR run **concurrently** (using a background thread pool), so the combined latency is approximately `max(ASR_time, diarization_time)` rather than the sum.

## Supported Models

All models from the [Sherpa-ONNX model zoo](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/) can be used. Download and extract a model into the `models/` directory, then pass the directory name via `--model-dir` and the architecture via `--model-type`.

> [!TIP]
> Models marked **auto** are downloaded automatically on first run. All others must be downloaded manually from the [Sherpa-ONNX releases page](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models).

### Online (Streaming) Models

Use these with the default pipeline (no `--offline` flag). They support real-time partial hypotheses.

| Model | `--model-dir` | `--model-type` | Lang | Notes |
|-------|--------------|----------------|------|-------|
| Zipformer En 2023 | `models/zipformer-en-2023` | *(blank)* | en | Default; **auto-downloaded** |
| Zipformer En 2024 | `models/sherpa-onnx-streaming-zipformer-en-2024-02-13` | `zipformer2` | en | Newer, slightly higher accuracy |
| Conformer En | `models/sherpa-onnx-streaming-conformer-en-2023-05-09` | `conformer` | en | Conformer transducer |
| Zipformer ZH/EN | `models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` | `zipformer` | zh/en | Bilingual |
| Paraformer ZH/EN | `models/sherpa-onnx-streaming-paraformer-bilingual-zh-en` | `paraformer` | zh/en | Streaming paraformer |
| WeNet CTC En | `models/sherpa-onnx-streaming-wenet-librispeech` | `wenet_ctc` | en | WeNet CTC |
| Zipformer2 CTC En | `models/sherpa-onnx-streaming-zipformer2-ctc-2024-09-18` | `zipformer2_ctc` | en | CTC variant |

Example:
```bash
python3 src/main.py --mic \
  --model-dir models/sherpa-onnx-streaming-zipformer-en-2024-02-13 \
  --model-type zipformer2
```

### Offline Models

Use these with `--offline`. Audio is VAD-segmented before recognition (higher accuracy, higher latency). A [Silero VAD](https://github.com/snakers4/silero-vad) model (`silero_vad.onnx`) is auto-downloaded when needed.

| Model | `--model-dir` | `--model-type` | Lang | Notes |
|-------|--------------|----------------|------|-------|
| Parakeet TDT 0.6B FP16 | `models/parakeet-tdt-0.6b-v2` | `nemo_transducer` | en | **Auto-downloaded** (`--offline` default) |
| Parakeet TDT 0.6B INT8 | `models/parakeet-tdt-0.6b-v2-int8` | `nemo_transducer` | en | **Auto-downloaded**; smaller & faster |
| Whisper tiny.en | `models/sherpa-onnx-whisper-tiny.en` | `whisper` | en | Smallest Whisper |
| Whisper base.en | `models/sherpa-onnx-whisper-base.en` | `whisper` | en | |
| Whisper small.en | `models/sherpa-onnx-whisper-small.en` | `whisper` | en | Good accuracy/speed balance |
| Whisper medium.en | `models/sherpa-onnx-whisper-medium.en` | `whisper` | en | Higher accuracy |
| Whisper large-v3 | `models/sherpa-onnx-whisper-large-v3` | `whisper` | multi | Multilingual; use `--language` |
| Paraformer ZH | `models/sherpa-onnx-paraformer-zh-2023-09-14` | `paraformer` | zh | |
| NeMo CTC En | `models/sherpa-onnx-nemo-ctc-en-conformer-medium` | `nemo_ctc` | en | NeMo Conformer CTC |
| SenseVoice | `models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` | `sense_voice` | multi | 5 languages; use `--language` |
| Moonshine tiny | `models/sherpa-onnx-moonshine-tiny-en-int8` | `moonshine` | en | Very fast, English only |
| Moonshine base | `models/sherpa-onnx-moonshine-base-en-int8` | `moonshine` | en | Better accuracy than tiny |
| FireRedASR | `models/sherpa-onnx-fire-red-asr-large-zh-2025-02-16` | `fire_red_asr` | zh | |

Examples:
```bash
# Parakeet TDT (auto-downloaded offline default)
python3 src/main.py --mic --offline --model-type nemo_transducer

# Parakeet TDT INT8 (smaller, auto-downloaded)
python3 src/main.py --mic --offline \
  --model-dir models/parakeet-tdt-0.6b-v2-int8 \
  --model-type nemo_transducer

# Whisper small (English)
python3 src/main.py --mic --offline \
  --model-dir models/sherpa-onnx-whisper-small.en \
  --model-type whisper

# Whisper large-v3 (multilingual)
python3 src/main.py --mic --offline \
  --model-dir models/sherpa-onnx-whisper-large-v3 \
  --model-type whisper --language zh

# SenseVoice (5 languages)
python3 src/main.py --mic --offline \
  --model-dir models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
  --model-type sense_voice --language ja

# Moonshine tiny
python3 src/main.py --mic --offline \
  --model-dir models/sherpa-onnx-moonshine-tiny-en-int8 \
  --model-type moonshine
```

## CLI Options

```
--mic                   Stream from microphone
--wav PATH              Transcribe a WAV file
--model-dir PATH        Sherpa-ONNX model directory
                          Default (online):  models/zipformer-en-2023
                          Default (offline): models/parakeet-tdt-0.6b-v2
--model-type TYPE       Model architecture hint (leave blank for auto-detect)
                          Online:  transducer, zipformer, zipformer2, conformer, lstm,
                                   paraformer, ctc, wenet_ctc, zipformer2_ctc
                          Offline: transducer, nemo_transducer, paraformer, whisper,
                                   ctc, nemo_ctc, sense_voice, moonshine, fire_red_asr
--offline               Use VAD-segmented offline pipeline instead of streaming
--language LANG         Language code for Whisper / SenseVoice (default: en)
--sample-rate INT       Audio sample rate in Hz (default: 16000)
--chunk-size FLOAT      Chunk size in seconds (default: 0.16)
--threads INT           CPU thread count for ONNX runtime (default: 4)
--capture-rate HZ       Microphone capture rate — use 48000 for device compatibility
--vad-model PATH        Path to silero_vad.onnx (auto-downloaded if not provided)
--listening             Show a live RMS energy bar for mic level calibration
--diarization           Enable speaker diarization with colour-coded output
--num-speakers N        Number of speakers (-1 = auto-detect, default: -1)
--diarization-seg-model PATH
                        Pyannote segmentation model.onnx (auto-downloaded if absent)
--diarization-emb-model PATH
                        Speaker embedding extractor .onnx (auto-downloaded if absent)
```

## Architecture

```
src/
├── main.py            # CLI entry point, model download, validation
├── asr_engine.py      # Unified model loading for all sherpa-onnx model types + diarization
├── streaming.py       # Streaming decode loop & VAD-segmented offline loop; rich output
├── audio.py           # Microphone capture & WAV file reading
├── config.py          # Configuration dataclass
└── requirements.txt   # Python dependencies
```

| Module | Responsibility |
|--------|----------------|
| `main.py` | Parses arguments, validates inputs, auto-downloads model/VAD/diarization models, dispatches to streaming |
| `asr_engine.py` | Builds `OnlineRecognizer`, `OfflineRecognizer`, `VoiceActivityDetector`, or `OfflineSpeakerDiarization` |
| `streaming.py` | Feeds audio chunks to the recognizer; runs ASR and diarization concurrently; renders colour-coded output via `rich` |
| `audio.py` | Provides two generators: `mic_stream()` for live capture, `read_wav()` for file input |
| `config.py` | Holds runtime parameters (sample rate, chunk size, thread count, model path, language, diarization settings) |

## Endpoint Detection (Online Mode)

| Rule | Behavior |
|------|----------|
| Rule 1 | 2.4 s trailing silence → hard endpoint |
| Rule 2 | 1.2 s silence after sufficient speech → early endpoint |
| Rule 3 | 300 s max utterance → forced endpoint (effectively disabled) |
