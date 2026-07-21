# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-07-15

### Added
- `sherox.wake`: new wake-word detection subcommand backed by
  `livekit-wakeword`, supporting custom ONNX models (any language) from a
  microphone or WAV file. Supports multiple models at once, configurable
  `--threshold`, `--debounce`, and `--chunk-size`. Models are resolved from
  an explicit path or looked up under `models/<name>.onnx`.
- `sherox.tts --lang eng-kitten`: Kitten TTS Nano v0.8 (quantized, 24 kHz,
  8 speakers), auto-downloaded to `models/kitten/`.
- `sherox.tts`: Supertonic-3 multilingual model (24 kHz, 10 speakers,
  ~120 MB shared download) adds 25 new `--lang` codes — `kor`, `ara`, `bul`,
  `ces`, `dan`, `ell`, `est`, `fin`, `hin`, `hrv`, `hun`, `ita`, `lit`, `lav`,
  `nld`, `pol`, `por`, `ron`, `rus`, `slk`, `slv`, `swe`, `tur`, `ukr`, `vie`
  — plus `ind-supertonic` as an alternate Indonesian voice. Existing
  dedicated-model languages (`eng`, `deu`, `fra`, `spa`, `ind`, `zho`, `jpn`)
  are unaffected.
- `sherox list-models`: registry now includes `eng-kitten`, the
  Supertonic-3 entry, and the `wake` module.
- `--model-type nemo_ctc --language en`: auto-downloads
  `sherpa-onnx-nemo-ctc-en-conformer-medium` by default (small variant
  available via explicit `--model-dir`), exposing per-word CTC confidence
  for English without a manual download step.
- `sherox.asr --vad-min-silence-duration`, `--vad-threshold`,
  `--vad-min-speech-duration`: tunable VAD sensitivity/timing for the
  offline pipeline (previously only settable through `Config`, not the
  CLI), trading segmentation safety for lower end-to-end latency.
- `sherox.asr --vad-max-speech-duration`: caps how long the offline VAD
  will accumulate audio without a pause before force-cutting a segment
  (sherpa-onnx's own default is 20s and was previously not exposed at
  all); lower it for more frequent feedback during long, uninterrupted
  dictation.
- `sherox.asr --debug-latency`: prints per-segment `queue` / `decode` /
  `endpoint→text` timing to stderr for the offline pipeline, to identify
  where latency is actually going before tuning VAD or thread settings.
- `sherox.asr --mic` now auto-saves its transcript when the session ends
  or is interrupted with Ctrl+C — `--output`/`--output-dir` (previously
  WAV/pipe-only) now also apply to mic mode, and when neither is given it
  falls back to a timestamped file
  (`sherox_asr_<model>_YYYYMMDD_HHMMSS.txt`) under the system temp
  directory so a live session is never silently lost. Disable with
  `--no-save-transcript`.

### Changed
- Offline recognizers (`sherox.asr`, `sherox.server`) now run a throwaway
  warm-up decode right after model load, moving ONNX Runtime's cold-start
  cost into "Loading model…" instead of the user's first spoken utterance.

### Fixed
- `sherox.tts` Supertonic backend now reports the model's actual sample
  rate (`tts.sample_rate`) instead of an unset `audio.sample_rate`
  attribute, which could write WAV files with the wrong rate.

## [0.8.0] - 2026-06-07

### Changed
- `--listening` replaced by `--no-mic-level` across `asr`, `segment`, and `lid`
  — mic level bar is now **on by default** when using `--mic`; pass
  `--no-mic-level` to suppress it.
- `kws` (`sherox.kws --mic`) now shows the live RMS energy bar by default,
  consistent with all other mic-mode commands. Suppress with `--no-mic-level`.
- `livekit-wakeword` moved from core dependencies to the optional `wake` extra
  (`pip install sherox[wake]`).

## [Major.Minor.Patch] - YYYY-MM-DD

## [0.7.0] - 2026-06-04

### Added
- `sherox.sid --enroll-mic NAME`: enroll a new speaker by recording from the
  microphone. Uses Silero VAD to segment speech, saves each segment as a WAV
  file alongside `--speaker-file`, then appends entries. Press Ctrl+C when
  done speaking.
- `SidConfig` now exposes `vad_threshold`, `vad_min_silence_duration`, and
  `vad_min_speech_duration` fields, allowing callers to tune VAD segmentation
  for speaker identification without touching ASR defaults.

### Changed
- `--speaker-file` is no longer required; it defaults to `speakers.txt` in the
  current directory. If the default file is missing during `--mic`/`--wav`,
  the error message now suggests enrolling a speaker or specifying a custom
  path with `--speaker-file`.
- VAD parameters in `run_mic()` and `enroll_speaker_mic()` are now tuned for
  speaker identification (threshold=0.3, min_speech=1.0s, min_silence=1.0s)
  instead of the ASR defaults (threshold=0.1, min_speech=0.25s,
  min_silence=0.5s). This prevents tiny fragments from breath pauses and
  non-speech noise, producing 2-6 second segments that yield stable Titanet
  embeddings.
- Mic level bar (`show_mic_level`) is now enabled by default for both
  `--mic` and `--enroll-mic` modes, so users can see live RMS energy and
  verify the microphone is working.
- `_load_speaker_file` error message now includes actionable hints when the
  default `speakers.txt` is missing.

## [0.6.0] - 2026-06-03

### Added
- `python -m sherox`: the top-level CLI can now be invoked as a module via the
  new `sherox/__main__.py` entry point.
- Exception hierarchy in `sherox/__init__.py`: `SherpaError` (base) with
  `ModelNotFoundError`, `AudioError`, and `ConfigError`. Library code now raises
  these instead of calling `sys.exit()`, so callers (tests, embedding apps) can
  handle failures gracefully.
- `sherox.asr` / `sherox.segment`: validate that `--capture-rate` is
  `>= --sample-rate`, with a hint to use `--capture-rate 48000` for device
  compatibility.

### Changed
- CLI entry points (`asr`, `kws`, `lid`, `segment`, `server`, `sid`, `tts`) now
  delegate to a shared `utils.run_cli()` helper that maps `SherpaError` to exit
  code 1 and `KeyboardInterrupt` to 130.
- Consolidated the per-module `_safe_tar_members` implementations into a single
  `utils.safe_tar_members()` used by `asr`, `kws`, `lid`, and `tts`.

### Fixed
- `sherox.server` `/ws` online handler: network sends are no longer performed
  while holding `online_lock`, so a slow WebSocket client can no longer stall
  decoding for other connections. The decode/endpoint/reset sequence remains
  atomic under the lock.
- `utils.safe_tar_members()` now also rejects symlink/hardlink members whose
  target resolves outside the extraction directory (path-traversal hardening on
  Python < 3.12, where `tarfile` `filter="data"` is unavailable).
- `audio.mic_stream`: the input stream is now stopped and closed via
  `try/finally` so device resources are released when the generator is closed.

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
