# Copilot Instructions

## Build & Test Commands

```bash
# Install for development (includes pytest, pytest-cov, fastapi, httpx)
pip install -e '.[dev]'

# Install Japanese TTS support
pip install -e '.[tts-ja]'

# Install server dependencies
pip install -e '.[server]'

# Run all tests
pytest

# Run a single test file
pytest tests/test_asr.py

# Run a single test class or function
pytest tests/test_asr.py::TestParseArgs::test_mic_mode

# Run with coverage
pytest --cov=sherox --cov=benchmark

# Run the CLI locally
python -m sherox.asr --mic
python -m sherox.asr --wav data/happy_50_gtts.wav
```

## Architecture

```
sherox/
├── asr.py         # CLI entry point: arg parsing, model download, input validation, dispatch
├── asr_engine.py  # Sherpa-ONNX object construction: recognizers, VAD, diarization, embeddings
├── streaming.py   # Audio decode loops (online streaming + offline VAD-segmented); rich terminal output
├── audio.py       # Generators: mic_stream() and read_wav() — the only sources of audio data
├── config.py      # Dataclasses: Config, SidConfig, SegmentConfig, TtsConfig
├── sid.py         # Speaker identification CLI
├── segment.py     # VAD-based audio segmentation CLI
├── tts.py         # Text-to-speech CLI
├── server.py      # FastAPI HTTP/WebSocket ASR server (optional; requires [server] extra)
└── utils.py       # Shared helpers: download_file(), _info(), _error()
```

The data flow for ASR is: `asr.py` (parse + validate + download) → `asr_engine.py` (build recognizer/VAD/diarization objects) → `streaming.py` (feed audio from `audio.py`, decode, render output).

**Online vs offline pipeline distinction:**  
- Online (streaming): `build_recognizer()` → `run_streaming()` — partial hypotheses updated live.  
- Offline (VAD-segmented): `build_offline_recognizer()` + `build_vad()` → `run_offline_vad_streaming()` — audio is accumulated until silence, then decoded.  
- Diarization runs concurrently in a `ThreadPoolExecutor` inside `streaming.py`; latency is `max(ASR_time, diarization_time)`.

## Key Conventions

**Lazy imports for heavy dependencies:** `sherpa_onnx`, `sounddevice`, and `soundfile` are imported lazily via `_require_*()` functions at call time. This keeps the module importable in test environments without hardware or the full runtime installed. Tests mock these with `MagicMock`.

**Proxy objects for missing sherpa-onnx:** `asr_engine.py` defines `_MissingSherpaOnnxProxy` so that `import sherpa_onnx` never raises at module level — errors are deferred to call time. Follow this pattern when adding new sherpa-onnx types.

**`cfg.vad_model` must be resolved before `build_vad()`:** The `Config.vad_model` field starts empty. `asr._validate_vad()` downloads the model if needed and sets the resolved path on the config object. Never call `build_vad()` without first running `_validate_vad()`.

**`_find(directory, glob_pattern)`** in `asr_engine.py` is the canonical way to locate ONNX files inside a model directory. It always sorts and returns the first match — rely on it rather than hard-coding file names.

**Rich output:** All terminal output uses `rich`. Use `_info()` / `_error()` from `utils.py` for status messages. Speaker-coloured transcript lines are built as `rich.text.Text` objects in `streaming.py`. The console instance is `Console(highlight=False, markup=False)` to avoid accidental markup interpretation in transcript text.

**WAV input requirements:** Input audio must be mono, 16-bit, 16 kHz. `audio.py` resamples from `capture_rate` to `sample_rate` using linear interpolation (`_resample()`). When adding new audio sources, follow the generator protocol of `mic_stream()` and `read_wav()`.

**Config dataclasses are the sole parameter-passing mechanism** between CLI modules and engine/streaming layers. Do not add keyword arguments to `build_*()` functions; put new fields in the relevant `*Config` dataclass instead.

**Test mocking pattern:** Tests patch `sys.argv` with `patch("sys.argv", [...])` for CLI arg tests. Sherpa-ONNX runtime objects are replaced with `MagicMock()`. Audio device calls (`sounddevice.InputStream`) are patched to avoid hardware access.

**Model directory layout:** Models live in `models/<model-dir-name>/`. The `_find()` helper uses glob patterns like `"encoder*.onnx"` to locate files — new model types should follow this convention.
