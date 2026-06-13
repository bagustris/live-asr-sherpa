[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/bagustris/live-asr-sherpa)

# Sherox: Speech AI Inference Toolkit

*Minimal, light, live speech recognition (and others) on Local PC*

A terminal-based toolkit originally built with [**SHER**rpa-**O**nn**X**](https://github.com/k2-fsa/sherpa-onnx). Transcribe speech in real-time from your microphone or offline from WAV files — no GPU required. Also supports speaker diarization, speaker identification, TTS, language identification, keyword spotting, wake-word detection, and speech segmentation.

Documentation: https://deepwiki.com/bagustris/live-asr-sherpa

Supported tasks:  
- Automatic Speech Recognition (ASR): `sherox.asr` (with optional diarization)
- Speaker Identification (SID): `sherox.sid`
- Text-to-Speech (TTS): `sherox.tts`
- Language Identification (LID): `sherox.lid`
- Keyword Spotting (KWS): `sherox.kws`
- Wake-word Detection: `sherox.wake`
- Speech Segmentation: `sherox.segment`

## Features

- **Real-time microphone transcription** — partial hypotheses update live with <500 ms latency
- **Offline WAV transcription** — process audio files through the same pipeline
- **Speaker diarization** — colour-coded per-speaker output; ASR and diarization run concurrently to keep latency low
- **Speaker identification** — enroll speakers via microphone or WAV files and identify them in real-time using neural embeddings
- **Unified model loading** — all sherpa-onnx model families supported via a single `--model-type` flag
- **CPU-optimized** — runs efficiently on any modern CPU using ONNX Runtime
- **Auto model download** — fetches default models on first run
- **Endpoint detection** — intelligently segments speech with configurable silence rules
- **Wake-word detection** — run custom `livekit-wakeword` ONNX triggers from a microphone or WAV file
- **Rich terminal output** — colour-coded speaker labels and styled status messages via the `rich` library

## Prerequisites

- Python 3.11+
- A working microphone (for `--mic` mode)
- Linux recommended (Ubuntu compatible)

## Getting Started

### 1. Install

```bash
pip install -e .
```

> **Note:** This installs all dependencies and registers the `sherox`, `sherox.asr`, `sherox.sid`, `sherox.segment`, `sherox.tts`, `sherox.kws`, `sherox.wake`, `sherox.lid`, `sherox.server`, and `sherox.models` CLI commands.
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

The default English Parakeet int8 model is downloaded automatically on first run.

## Speaker Diarization

Add `--diarization` to any command to colour-code the transcript by speaker. Two lightweight models are downloaded automatically on first use (~7 MB segmentation + ~23 MB embedding):

See [DIARIZATION_MODEL.md](DIARIZATION_MODEL.md) for the diarization model files, defaults, and manual override paths.

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

### 1. Enroll speakers

**From microphone** — speak for a few seconds, press Ctrl+C when done:

```bash
# Saves recordings as alice_mic_enroll_001.wav alongside speakers.txt
sherox.sid --enroll-mic alice
```

**From WAV files** — point to existing recordings:

```bash
sherox.sid --enroll alice ref1.wav ref2.wav
```

Both modes append entries to `speakers.txt` (default, created if absent). Duplicate `name+path` pairs are silently skipped. Multiple WAV files for the same speaker are averaged into a single embedding.

The speaker file format (one `name /path/wav` per line):

```
alice /path/to/alice_mic_enroll_001.wav
bob   /path/to/bob1.wav
```

### 2. Identify

**Microphone mode** — real-time identification (VAD-segmented):

```bash
sherox.sid --mic
```

**WAV file mode** — identify speaker in a recording:

```bash
sherox.sid --wav audio.wav
```

Each identified speaker is printed in a distinct colour; audio that does not match any registered speaker prints as `unknown`.

**Options:**

```
--enroll-mic NAME      Enroll a speaker by recording from microphone (Ctrl+C to stop)
--enroll NAME WAV...   Enroll a speaker from WAV file(s)
--mic                  Identify speakers from microphone (VAD-segmented)
--wav PATH             Identify speaker in a WAV file
--speaker-file PATH    Text file with 'name /path/wav' entries (default: speakers.txt)
--model PATH           Speaker embedding ONNX model (default: models/nemo_en_titanet_large.onnx)
--threshold FLOAT      Cosine similarity threshold (0–1, higher = stricter; default: 0.6)
--capture-rate HZ      Mic capture rate (default: 16000; use 48000 for device compatibility)
```

> **Tip:** A live RMS energy bar is shown by default in `--mic` and `--enroll-mic` modes. Use it to verify your microphone is picking up audio and to calibrate your speaking volume.

See [SID_MODEL.md](SID_MODEL.md) for the default speaker ID model, alternative embedding models, and related VAD dependency.

## Supported ASR Models

See [ASR_MODEL.md](ASR_MODEL.md) for the complete supported ASR model catalog, grouped by streaming and offline usage, including built-in aliases and auto-downloaded defaults.

## CLI Reference

### `sherox.asr`

```
--mic                   Stream from microphone
--wav PATH              Transcribe a WAV file
--model-dir PATH        Sherpa-ONNX model directory
                          Default English: models/parakeet-tdt-0.6b-v2-int8
                          Other languages: selected from --lang/--language when supported
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
--enroll-mic NAME        Enroll a speaker by recording from microphone
                         (Ctrl+C to stop; recordings saved alongside --speaker-file)
--enroll NAME WAV [WAV…] Enroll a speaker from one or more WAV files
--mic                    Stream from microphone (VAD-segmented)
--wav PATH               Identify speaker in a WAV file
--speaker-file PATH      Text file with 'name /path/to/ref.wav' entries
                           (default: speakers.txt)
--model PATH             Speaker embedding ONNX model
                           (default: models/nemo_en_titanet_large.onnx; auto-downloaded)
--threshold FLOAT        Cosine similarity threshold for a match (default: 0.6)
--sample-rate INT        Expected sample rate for WAV input (default: 16000)
--capture-rate HZ        Microphone capture rate (default: 16000)
--chunk-size FLOAT       Mic audio chunk size in seconds (default: 0.1)
--threads INT            CPU thread count for ONNX runtime (default: 4)
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
--output PATH           Output WAV file path (default: output.wav; use none or - with --play to skip saving)
--play                  Play audio after synthesis
--no-save               Do not write a WAV file; requires --play
--threads INT           CPU thread count for ONNX runtime (default: 4)
```

Examples:

```bash
# Default Indonesian model via sherpa-onnx
sherox.tts --text "Halo dunia"

# Play without saving a WAV file
sherox.tts --text "Halo dunia" --play --no-save

# Equivalent playback-only form
sherox.tts --text "Halo dunia" --play --output none

# Japanese via Piper Plus Tsukuyomi
sherox.tts --text "こんにちは、今日は良い天気ですね。" --lang jpn
```

See [TTS_MODEL.md](TTS_MODEL.md) for the supported built-in TTS languages, backends, and model details.

### `sherox.kws`

```
--mic                    Stream from microphone
--wav PATH               Spot keywords in a WAV file
--keywords LIST          Comma-separated keyword phrases (required, or --keywords-file)
--keywords-file PATH     Plain-text file with one keyword per line (required, or --keywords)
--model-dir PATH         Custom KWS model directory
                           (default: auto-download zipformer-gigaspeech-3.3M)
--sample-rate INT        Model input sample rate (default: 16000)
--capture-rate HZ        Microphone capture rate (default: 16000)
--chunk-size FLOAT       Audio chunk duration per decode (default: 0.1)
--threads INT            CPU thread count for ONNX runtime (default: 4)
--max-active-paths INT   Beam width for keyword search (default: 4)
--keywords-score FLOAT   Boost score for keyword tokens (default: 1.0)
--keywords-threshold FLOAT
                         Trigger threshold; higher = stricter (default: 0.25)
```

### `sherox.wake`

Wake-word detection powered by [`livekit-wakeword`](https://github.com/livekit/livekit-wakeword). Use any custom ONNX model trained via `livekit-wakeword run configs/foo.yaml` — works for any language, including non-English names like "Hey Bagus".

Examples:

```bash
sherox.wake --mic --model models/hey_bagus.onnx
sherox.wake --wav audio.wav --model models/hey_bagus.onnx --threshold 0.7
sherox.wake --mic --model hey_bagus --model models/hey_sherox.onnx
```

Model paths must point to `.onnx` files. Relative names are also searched under `models/`, so `--model hey_bagus` resolves to `models/hey_bagus.onnx` when present. Each detection prints `[HH:MM:SS] model_name: confidence`.

```
--mic                    Stream from microphone
--wav PATH               Scan a WAV file
--model PATH             Path to a wake-word ONNX model (may be repeated for multiple)
--threshold FLOAT        Detection threshold 0.0-1.0 (default: 0.5)
--debounce SECS          Minimum seconds between detections (default: 2.0)
--chunk-size SECS        Audio chunk duration per inference (default: 2.0)
```

### `sherox.lid`

```
--mic                    Stream from microphone
--wav PATH               Identify language in a WAV file
--size {tiny,base,small,medium}
                         Whisper variant (default: tiny)
--encoder PATH           Custom Whisper encoder .onnx (overrides --size)
--decoder PATH           Custom Whisper decoder .onnx (overrides --size)
--sample-rate INT        Expected WAV sample rate (default: 16000)
--capture-rate HZ        Microphone capture rate (default: 16000)
--chunk-size FLOAT       Mic audio chunk size in seconds (default: 0.1)
--threads INT            CPU thread count for ONNX runtime (default: 4)
--provider {cpu,cuda,coreml}
                         ONNX Runtime execution provider (default: cpu)
--listening              Show RMS energy bar for mic level calibration
```

### `sherox.segment`

```
--mic                    Segment microphone input
--wav PATH               Segment a WAV file
--vad-model {silero,ten-vad}
                         VAD model (default: silero)
--ten-vad-model {ten-vad.onnx,ten-vad.int8.onnx}
                         Ten-VAD model variant (default: ten-vad.int8.onnx)
--threshold F            VAD speech probability threshold 0.0–1.0 (default: 0.5)
--min-silence SEC        Silence to end a segment in seconds (default: 0.5)
--min-speech SEC         Speech to start a segment in seconds (default: 0.25)
--sample-rate HZ         Audio sample rate for VAD model (default: 16000)
--capture-rate HZ        Microphone capture rate (default: 16000)
--threads INT            CPU thread count for ONNX runtime (default: 4)
--output-dir DIR         Save segments as WAV files into this directory
--listening              Show RMS energy bar for mic level calibration
```

### `sherox.server`

```
--model-dir PATH         Path to model directory (auto-selected if omitted)
--model-type TYPE        Model architecture hint (blank = auto-detect)
--offline                Use VAD-segmented offline pipeline
--sample-rate HZ         Expected audio sample rate (default: 16000)
--threads INT            CPU thread count for ONNX runtime (default: 4)
--language LANG          Language code for Whisper/SenseVoice/Cohere (default: en)
--vad-model {silero,ten-vad}
                         VAD type for offline segmentation (default: silero)
--ten-vad-model {ten-vad.onnx,ten-vad.int8.onnx}
                         Ten-VAD model variant (default: ten-vad.int8.onnx)
--host HOST              Host address to listen on (default: 0.0.0.0)
--port PORT              Port to listen on (default: 8000)
--log-level {debug,info,warning,error}
                         Uvicorn log level (default: info)
```

### `sherox list-models`

```
--module {asr,tts,other,all}
                         Filter by module (default: all)
--no-color               Disable colour output
```

See [SEGMENT_MODEL.md](SEGMENT_MODEL.md) for the VAD models used by `sherox.segment`.

## Architecture

```
sherox/
├── asr.py         # sherox.asr — CLI, model download, validation, dispatch
├── sid.py         # sherox.sid — Speaker identification CLI
├── segment.py     # sherox.segment — VAD-based audio segmentation CLI
├── tts.py         # sherox.tts — Text-to-speech CLI
├── wake.py        # sherox.wake — Wake-word detection CLI
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
| `wake.py` | Wake-word detection from microphone or WAV using custom `livekit-wakeword` ONNX models |
| `asr_engine.py` | Builds `OnlineRecognizer`, `OfflineRecognizer`, `VoiceActivityDetector`, `OfflineSpeakerDiarization`, and `SpeakerEmbeddingExtractor` |
| `streaming.py` | Feeds audio chunks to the recognizer; runs ASR and diarization concurrently; renders colour-coded output via `rich` |
| `audio.py` | Provides two generators: `mic_stream()` for live capture, `read_wav()` for file input |
| `config.py` | Holds runtime parameters (`Config`, `SidConfig`, `SegmentConfig`, `TtsConfig`, `WakeConfig`) |

## Endpoint Detection (Online Mode)

| Rule | Behavior |
|------|----------|
| Rule 1 | 2.4 s trailing silence → hard endpoint |
| Rule 2 | 1.2 s silence after sufficient speech → early endpoint |
| Rule 3 | 300 s max utterance → forced endpoint (effectively disabled) |
