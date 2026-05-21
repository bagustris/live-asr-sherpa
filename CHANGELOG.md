# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Major.Minor.Patch] - YYYY-MM-DD

## [0.5.0] - 2026-05-21

### Added
- `sherox list-models`: new subcommand listing all auto-downloadable models
  across ASR, TTS, SID, KWS, VAD, punctuation and diarization modules.
  Output is a Rich table with columns: Module, Name, Language, Pipeline, Size,
  Notes. Supports `--module asr|tts|other|all` filter and `--no-color` for
  plain-text piping. Also accessible as `sherox.models`.
- `sherox.asr --no-color`: disable ANSI colour codes in transcript output,
  useful when redirecting to a file or piping to tools that do not interpret
  colour escapes (e.g. `sherox.asr --wav speech.wav --no-color > transcript.txt`).
- `sherox.asr --json`: emit each finalised segment as a newline-delimited JSON
  object `{"type":"segment","text":"...","start":0.0,"end":1.5}` (speaker key
  added when `--diarization` is active). Partial hypotheses are suppressed.
  Works with both online and offline pipelines, mic, WAV, and pipe modes.
  Example: `sherox.asr --mic --json | jq -r '.text'`
- `sherox.server` WebSocket endpoint `/ws` already streams both partial
  (`{"type":"partial"}`) and finalised (`{"type":"segment"}`) hypotheses in
  real-time (completed in v0.4.0, documented here for completeness).

### Changed
- `streaming.run_streaming()` and `streaming.run_offline_vad_streaming()` gain
  `json_output: bool` and `no_color: bool` keyword arguments.
- `streaming._rich_print()` gains an optional `console` parameter so callers
  can pass a `Console(no_color=True)` instance.
- New internal helper `streaming._emit_segment()` routes each finalised segment
  to either JSON stdout or Rich-formatted terminal output.

## [0.4.0] - 2026-05-21

### Added
- `sherox.asr --translate`: Whisper speech-to-English translation mode.
  Requires `--offline --model-type whisper`. Any non-Whisper model type
  (e.g. `sense_voice`) is rejected with a clear error.
- Chinese TTS support (`--lang zh` / `zho`): new `vits-icefall-zh-aishell3`
  model entry (8 kHz output, Simplified Chinese, lexicon-based G2P).
  Short aliases `zh`, `zh-cn`, `zh-tw`, `cmn`, `chi` all resolve to `zho`.
  The `build_tts()` path-resolution bug for lexicon-only models is fixed
  (empty `data_dir` is now passed as `""` rather than `str(model_dir / "")`).
- `sherox.sid --enroll NAME WAV [WAV…]`: enroll a new speaker without
  needing any model or VAD. Entries are appended to `--speaker-file` as
  absolute paths. Duplicate `name+path` pairs are silently skipped.
- `sherox.kws`: new keyword spotting subcommand backed by the Sherpa-ONNX
  Zipformer-GigaSpeech 3.3 M model (English, auto-downloaded).
  Supports `--mic` / `--wav`, `--keywords` (comma-separated string) or
  `--keywords-file` (plain-text file), configurable beam width and thread
  count. Each hit is printed with a wall-clock timestamp.

## [0.3.0] - 2026-05-21

### Added
- `--pipe` flag for `sherox.asr`: reads raw 16-bit LE mono PCM from stdin, enabling
  pipeline usage such as `arecord -f S16_LE -r 16000 -c 1 | sherox.asr --pipe` or
  `ffmpeg -i audio.mp4 -f s16le -ar 16000 -ac 1 - | sherox.asr --pipe`
- `sherox` bare entry point: `sherox --version` / `sherox -v` shows version, `sherox --help` lists subcommands

### Fixed
- `--model-type ja/ja-en/ja-en-mls-5k` short aliases now correctly resolve to their ReazonSpeech models
- Offline CTC recognizer builder now calls the correct `from_wenet_ctc` / `from_nemo_ctc` methods

## [0.2.0] - 2026-05-21

### Added
- Version number (sherox --version)
- CHANGELOG.md file

### Fixed
- German default model for ASR (--lang de)
