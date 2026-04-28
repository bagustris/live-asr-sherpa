# Live ASR with Sherpa-ONNX

*Minimal, light, live speech recognition (and others) on Laptop's CPU*

A terminal-based toolkit built with [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx). Transcribe speech in real-time from your microphone or offline from WAV files — no GPU required. Also supports speaker diarization, speaker identification, TTS, and speech segmentation.

Documentation:  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/bagustris/live-asr-sherpa).

## Features

- **Real-time microphone transcription** — partial hypotheses update live with <500 ms latency
- **Offline WAV transcription** — process audio files through the same pipeline
- **Speaker diarization** — colour-coded per-speaker output; ASR and diarization run concurrently to keep latency low
- **Speaker identification** — identify known speakers in real-time or from a WAV file using neural embeddings
- **Unified model loading** — all sherpa-onnx model families supported via a single `--model-type` flag
- **CPU-optimized** — runs efficiently on any modern CPU using ONNX Runtime
- **Auto model download** — fetches default models on first run
- **Endpoint detection** — intelligently segments speech with configurable silence rules
- **Rich terminal output** — colour-coded speaker labels and styled status messages via the `rich` library

## Prerequisites

- Python 3.8+
- A working microphone (for `--mic` mode)
- Linux recommended (Ubuntu compatible)

## Getting Started

### 1. Install

```bash
pip install -e .
```

> **Note:** This installs all dependencies and registers the `sherox.asr`, `sherox.sid`, `sherox.segment`, and `sherox.tts` CLI commands.
>
> Japanese TTS uses the `piper-plus` backend rather than `sherpa-onnx`. Install it with:
> ```bash
> pip install -e '.[tts-ja]'
> ```

### 2. Run ASR

**Microphone mode** — stream and transcribe live audio:

```bash
sherox.asr --mic
```

**WAV file mode** — transcribe a pre-recorded file:

```bash
sherox.asr --wav path/to/audio.wav
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
# Microphone with diarization (auto-downloads all models)
sherox.asr --mic --offline --diarization

# With a known speaker count for better accuracy
sherox.asr --mic --offline --diarization --num-speakers 2

# Show [Speaker N] prefix in addition to colour
sherox.asr --mic --offline --diarization --speaker-tag

# WAV file with diarization
sherox.asr --wav meeting.wav --offline --diarization --num-speakers 3
```

Each speaker's transcript is printed in a distinct colour:

```
  [Speaker 00] Good morning everyone.
  [Speaker 01] Thanks for joining the call.
  [Speaker 00] Let's get started.
```

Diarization and ASR run **concurrently** (using a background thread pool), so the combined latency is approximately `max(ASR_time, diarization_time)` rather than the sum.

## Speaker Identification

`sherox.sid` identifies known speakers from a reference database using neural speaker embeddings. The default model (`nemo_en_titanet_large.onnx`, ~96 MB) is downloaded automatically on first run.

### 1. Prepare a speaker file

Create a text file with one `name /path/to/ref.wav` entry per line. Multiple files for the same speaker are averaged:

```
alice /path/to/alice1.wav
alice /path/to/alice2.wav
bob   /path/to/bob1.wav
```

### 2. Run

**WAV file mode** — identify speaker in a recording:

```bash
sherox.sid --wav audio.wav --speaker-file speakers.txt
```

**Microphone mode** — real-time identification (VAD-segmented):

```bash
sherox.sid --mic --speaker-file speakers.txt
```

Each identified speaker is printed in a distinct colour; audio that does not match any registered speaker prints as `unknown`.

**Options:**

```
--speaker-file PATH   Text file with 'name /path/wav' entries (required)
--model PATH          Speaker embedding ONNX model (default: models/nemo_en_titanet_large.onnx)
--threshold FLOAT     Cosine similarity threshold (0–1, higher = stricter; default: 0.6)
--capture-rate HZ     Mic capture rate (default: 16000; use 48000 for device compatibility)
--listening           Show RMS energy bar for mic level calibration
```

### Available embedding models

| Model | Size | Lang | Notes |
|-------|------|------|-------|
| `nemo_en_titanet_large.onnx` | 96 MB | en | **Default**; highest accuracy |
| `wespeaker_en_voxceleb_resnet293_LM.onnx` | 109 MB | en | WeSpeaker large |
| `wespeaker_en_voxceleb_CAM++_LM.onnx` | 27 MB | en | Good accuracy/speed balance |
| `nemo_en_speakerverification_speakernet.onnx` | 22 MB | en | Lightest option |
| `wespeaker_zh_cnceleb_resnet34_LM.onnx` | 25 MB | zh | Chinese |
| `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | 26 MB | zh/en | Bilingual |

Download any model from the [speaker recognition models release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) and pass it via `--model`.

## Supported ASR Models

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
sherox.asr --mic \
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
| ReazonSpeech JA | `models/reazonspeech-ja` | `ja` | ja | **Auto-downloaded**; Japanese |
| ReazonSpeech JA-EN | `models/reazonspeech-ja-en` | `ja-en` | ja/en | **Auto-downloaded**; bilingual |
| ReazonSpeech JA-EN-MLS | `models/reazonspeech-ja-en-mls-5k` | `ja-en-mls-5k` | ja/en | **Auto-downloaded**; bilingual + MLS 5k |
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

# Cohere Transcribe multilingual (14 languages)
sherox.asr --mic --offline --model-type cohere_transcribe --language en

# Cohere Transcribe with different language (e.g., Chinese)
sherox.asr --wav audio.wav --offline --model-type cohere_transcribe --language zh
```

## CLI Reference

### `sherox.asr`

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
                                   ctc, nemo_ctc, sense_voice, moonshine, fire_red_asr,
                                   cohere_transcribe
                          ReazonSpeech (offline): ja, ja-en, ja-en-mls-5k
--offline               Use VAD-segmented offline pipeline instead of streaming
--language LANG         Language code for Whisper / SenseVoice / Cohere Transcribe (default: en)
--sample-rate INT       Audio sample rate in Hz (default: 16000)
--chunk-size FLOAT      Chunk size in seconds (default: 0.16)
--threads INT           CPU thread count for ONNX runtime (default: 4)
--capture-rate HZ       Microphone capture rate — use 48000 for device compatibility
--vad-model {silero,ten-vad}
                        VAD type for offline segmentation (default: silero)
--ten-vad-model {ten-vad.onnx,ten-vad.int8.onnx}
                        Ten-VAD model variant (default: ten-vad.int8.onnx)
--listening             Show a live RMS energy bar for mic level calibration
--diarization           Enable speaker diarization with colour-coded output
--speaker-tag           Prefix each diarized line with [Speaker N] (requires --diarization)
--num-speakers N        Number of speakers (-1 = auto-detect, default: -1)
--diarization-seg-model PATH
                        Pyannote segmentation model.onnx (auto-downloaded if absent)
--diarization-emb-model PATH
                        Speaker embedding extractor .onnx (auto-downloaded if absent)
```

### `sherox.sid`

```
--mic                   Stream from microphone (VAD-segmented)
--wav PATH              Identify speaker in a WAV file
--speaker-file PATH     Text file with 'name /path/to/ref.wav' entries (required)
--model PATH            Speaker embedding ONNX model
                          (default: models/nemo_en_titanet_large.onnx; auto-downloaded)
--threshold FLOAT       Cosine similarity threshold for a match (default: 0.6)
--sample-rate INT       Expected sample rate for WAV input (default: 16000)
--capture-rate HZ       Microphone capture rate (default: 16000)
--chunk-size FLOAT      Mic audio chunk size in seconds (default: 0.1)
--threads INT           CPU thread count for ONNX runtime (default: 4)
--listening             Show a live RMS energy bar for mic level calibration
```

### `sherox.tts`

```
--text TEXT             Text to synthesise
--file PATH             Read text from a file
--lang LANG             ISO 639-3 language code (default: ind)
                        ind = Indonesian (Sherpa-ONNX Piper VITS)
                        jpn = Japanese (Piper Plus Tsukuyomi)
--model-dir PATH        Custom Sherpa-ONNX TTS model directory
                        (not used for the built-in Japanese Piper Plus backend)
--speaker-id N          Speaker identity index for multi-speaker models
--speed F               Speech rate multiplier (default: 1.0)
--output PATH           Output WAV file path
--play                  Play audio after synthesis
--threads INT           CPU thread count for ONNX runtime (default: 4)
```

Examples:

```bash
# Default Indonesian model via sherpa-onnx
sherox.tts --text "Halo dunia"

# Japanese via Piper Plus Tsukuyomi
sherox.tts --text "こんにちは、今日は良い天気ですね。" --lang jpn
```

## Architecture

```
sherox/
├── asr.py         # sherox.asr — CLI, model download, validation, dispatch
├── sid.py         # sherox.sid — Speaker identification CLI
├── segment.py     # sherox.segment — VAD-based audio segmentation CLI
├── tts.py         # sherox.tts — Text-to-speech CLI
├── asr_engine.py  # Unified model loading: ASR, VAD, diarization, embeddings
├── streaming.py   # Streaming & offline decode loops; rich terminal output
├── audio.py       # Microphone capture and WAV file reading generators
└── config.py      # Configuration dataclasses for all modules
```

| Module | Responsibility |
|--------|----------------|
| `asr.py` | Parses arguments, validates inputs, auto-downloads ASR/VAD/diarization models, dispatches to streaming |
| `sid.py` | Parses arguments, builds speaker database from reference WAVs, runs mic/WAV identification |
| `segment.py` | VAD-based segmentation of audio into timestamped speech clips |
| `tts.py` | Text-to-speech synthesis and playback |
| `asr_engine.py` | Builds `OnlineRecognizer`, `OfflineRecognizer`, `VoiceActivityDetector`, `OfflineSpeakerDiarization`, and `SpeakerEmbeddingExtractor` |
| `streaming.py` | Feeds audio chunks to the recognizer; runs ASR and diarization concurrently; renders colour-coded output via `rich` |
| `audio.py` | Provides two generators: `mic_stream()` for live capture, `read_wav()` for file input |
| `config.py` | Holds runtime parameters (`Config`, `SidConfig`, `SegmentConfig`, `TtsConfig`) |

## Endpoint Detection (Online Mode)

| Rule | Behavior |
|------|----------|
| Rule 1 | 2.4 s trailing silence → hard endpoint |
| Rule 2 | 1.2 s silence after sufficient speech → early endpoint |
| Rule 3 | 300 s max utterance → forced endpoint (effectively disabled) |
