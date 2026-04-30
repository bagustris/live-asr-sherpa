# Repository Guidelines

## Project Structure & Module Organization
Core package code lives in `sherox/`. Use `sherox/asr.py` for the ASR CLI, `sherox/segment.py` for segmentation, and `sherox/tts.py` for TTS. Shared runtime pieces such as model configuration, audio I/O, and streaming logic sit alongside them in `sherox/config.py`, `sherox/audio.py`, and `sherox/streaming.py`.

Tests are in `tests/` and mirror the package by feature, for example `tests/test_asr.py` and `tests/test_streaming.py`. Benchmark utilities live in `benchmark/`. Sample audio and transcripts used for local checks are under `data/`.

## Build, Test, and Development Commands
Install runtime dependencies with:

```bash
pip install -r requirements.txt
```

Install a contributor setup with tests and coverage extras:

```bash
pip install -e '.[dev]'
```

Run the main CLI locally with the packaged entry points:

```bash
python -m sherox.asr --mic
python -m sherox.asr --wav data/happy_50_gtts.wav
```

Run the full test suite or a focused module:

```bash
pytest
pytest tests/test_asr.py
pytest --cov=sherox --cov=benchmark
```

Benchmark a model with:

```bash
python benchmark/benchmark.py --data-dir /path/to/LibriSpeech/dev-clean-2 --offline
```

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints where useful, module-level constants in `UPPER_SNAKE_CASE`, functions and variables in `snake_case`, and test classes named `Test...`. Keep CLI help text and error messages direct. Match nearby import ordering and avoid broad refactors in targeted patches.

## Testing Guidelines
This repo uses `pytest`; coverage is optional but supported through `pytest-cov`. Add tests next to the affected behavior and name files `test_<feature>.py`. Prefer small unit tests with mocks for downloads, audio devices, and Sherpa runtime objects so tests stay deterministic and CPU-only.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `add example data`, `fix broken link and update conf`, and `Address reviewer comments`. Keep commits focused and descriptive. PRs should explain the behavioral change, list test coverage, and call out any model, audio, or benchmark implications. Include terminal output or screenshots only when CLI presentation changed.
