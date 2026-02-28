# asr-so

*Minimal, production-quality streaming speech recognition on CPU*

A terminal-based Automatic Speech Recognition (ASR) application built with [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx). Transcribe speech in real-time from your microphone or offline from WAV files — no GPU required.

## Features

- **Real-time microphone transcription** — partial hypotheses update live with <500ms latency
- **Offline WAV transcription** — process audio files through the same streaming pipeline
- **CPU-optimized** — runs efficiently on any modern CPU using ONNX Runtime
- **Auto model download** — fetches the Zipformer model on first run
- **Endpoint detection** — intelligently segments speech with configurable silence rules
- **Clean terminal output** — partial results overwrite in-place, finalized segments print on new lines

## Prerequisites

- Python 3.8+
- A working microphone (for `--mic` mode)
- Linux recommended (Ubuntu compatible)

## Getting Started

### 1. Install dependencies

```bash
cd src
pip install -r requirements.txt
```

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

The model is downloaded automatically on first run (~300 MB). To use a custom model directory:

```bash
python3 src/main.py --mic --model-dir /path/to/model
```

### CLI Options

```
--mic                 Stream from microphone
--wav PATH            Transcribe a WAV file
--model-dir PATH      Sherpa-ONNX model directory (default: model)
--sample-rate INT     Audio sample rate in Hz (default: 16000)
--chunk-size FLOAT    Chunk size in seconds (default: 0.16)
--threads INT         CPU thread count for ONNX runtime (default: 4)
```

## Architecture

```
src/
├── main.py            # CLI entry point, model download, validation
├── asr_engine.py      # Sherpa-ONNX recognizer setup & endpoint config
├── streaming.py       # Real-time decode loop & terminal rendering
├── audio.py           # Microphone capture & WAV file reading
├── config.py          # Configuration dataclass
└── requirements.txt   # Python dependencies
```

| Module | Responsibility |
|--------|----------------|
| `main.py` | Parses arguments, validates inputs, downloads model if missing, dispatches to streaming |
| `asr_engine.py` | Builds the `OnlineRecognizer` with Zipformer transducer and endpoint detection rules |
| `streaming.py` | Feeds audio chunks to the recognizer, renders partial/final hypotheses to the terminal |
| `audio.py` | Provides two generators: `mic_stream()` for live capture, `read_wav()` for file input |
| `config.py` | Holds runtime parameters (sample rate, chunk size, thread count, model path) |

## Model

Uses [sherpa-onnx-streaming-zipformer-en-2023-06-26](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) — a streaming transducer optimized for low-latency CPU inference with competitive word error rate for English.

Endpoint detection rules:

| Rule | Behavior |
|------|----------|
| Rule 1 | 2.4s trailing silence → hard endpoint |
| Rule 2 | 1.2s silence after speech → early endpoint |
| Rule 3 | 20s max utterance → forced endpoint |

## Possible Improvements

- **Quantized models** — INT8 quantization for faster inference on edge devices
- **VAD integration** — WebRTC VAD to skip silence and reduce compute
- **Word timestamps** — per-word timing for subtitle generation
- **Confidence scores** — decode-time confidence for downstream filtering
- **Multilingual support** — swap in non-English Sherpa-ONNX models
- **Downstream NLP** — pipe transcripts to summarization, translation, or entity extraction
