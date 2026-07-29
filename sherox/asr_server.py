"""sherox.asr_server — HTTP ASR API server.

Usage:
    # Default host/port:
    sherox.asr_server

    # Custom host/port, more ONNX threads:
    sherox.asr_server --host 0.0.0.0 --port 8200 --threads 2

API endpoints:
    GET  /health       -> {"status": "ok", "languages_loaded": [...]}
    POST /transcribe   -> multipart form: file=<audio file>, lang="en" (optional)
                          -> JSON {"text": "...", "lang": "en"}

ASR recognizers are built lazily per language on first request and cached for
the lifetime of the process (mirrors sherox.tts_server's _TtsCache pattern), so
repeat calls skip the model-load cost that a fresh CLI/subprocess invocation
would otherwise pay every time. Each language gets sherox.asr's default offline
model (resolve_language_model), auto-downloaded on first use.

For real-time microphone streaming use sherox.server instead — that sibling
server exposes a WebSocket endpoint and a browser demo page on top of a single
eagerly-loaded model; this one trades those for multi-language lazy loading.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

try:
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.routing import APIRouter
    import uvicorn
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "Server dependencies not installed. "
        "Run: pip install 'sherox[server]'"
    ) from _e

from .asr import _normalize_language, ensure_model, resolve_language_model
from .asr_engine import build_offline_recognizer, warmup_offline_recognizer
from .config import Config
from .streaming import _run_asr
from .utils import run_cli as _run_cli

logger = logging.getLogger(__name__)


# ── App state ─────────────────────────────────────────────────────────────────

class _AsrCache:
    """Per-language ASR recognizer cache, built lazily on first request.

    A single lock serializes *builds* (first use of a language) so concurrent
    requests for a not-yet-loaded language don't race into downloading/
    building the same model twice; already-cached languages are read without
    contention.
    """

    def __init__(self, project_dir: Path, num_threads: int, sample_rate: int) -> None:
        self.project_dir = project_dir
        self.num_threads = num_threads
        self.sample_rate = sample_rate
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, lang: str):
        lang = _normalize_language(lang)
        if lang not in self._models:
            with self._lock:
                if lang not in self._models:
                    model_type, model_dir_name, offline = resolve_language_model(
                        lang, offline=True,
                    )
                    model_dir = self.project_dir / "models" / model_dir_name
                    ensure_model(str(model_dir), model_type)
                    cfg = Config(
                        model_dir=str(model_dir), sample_rate=self.sample_rate,
                        num_threads=self.num_threads, model_type=model_type,
                        offline=offline, language=lang,
                    )
                    recognizer = build_offline_recognizer(cfg)
                    warmup_offline_recognizer(recognizer, cfg.sample_rate)
                    self._models[lang] = recognizer
        return self._models[lang]

    def loaded_languages(self) -> list[str]:
        return sorted(self._models.keys())


# ── App factory ───────────────────────────────────────────────────────────────

def _create_app(project_dir: Path, num_threads: int, sample_rate: int = 16000) -> FastAPI:
    app = FastAPI(title="sherox ASR server", version="0.1.0")
    app.state.asr_cache = _AsrCache(project_dir, num_threads, sample_rate)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


# ── Routes ────────────────────────────────────────────────────────────────────

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    """Return server status and which languages have been loaded so far."""
    cache: _AsrCache = request.app.state.asr_cache
    return {"status": "ok", "languages_loaded": cache.loaded_languages()}


@router.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...), lang: str = Form("en")):
    """Transcribe an uploaded audio file (WAV, FLAC, OGG…).

    Form fields: `file` (audio) and `lang` (optional, default "en").
    The audio must be at the server's configured sample rate (default 16 kHz);
    stereo files are automatically downmixed to mono.
    Returns ``{"text": "<transcription>", "lang": "<language>"}`` on success.
    """
    cache: _AsrCache = request.app.state.asr_cache
    raw = await file.read()

    try:
        samples = _decode_audio(raw, cache.sample_rate)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    loop = asyncio.get_event_loop()
    try:
        recognizer = await loop.run_in_executor(None, cache.get, lang)
        text = await loop.run_in_executor(
            None, _run_asr, recognizer, samples, cache.sample_rate,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"text": text, "lang": _normalize_language(lang)}


# ── Inference helpers ─────────────────────────────────────────────────────────

def _decode_audio(raw: bytes, sample_rate: int) -> np.ndarray:
    """Decode uploaded audio bytes to mono float32 at the server's sample rate."""
    samples, sr = soundfile.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)  # stereo → mono
    if sr != sample_rate:
        raise ValueError(
            f"Audio must be {sample_rate} Hz, got {sr} Hz. "
            "Resample before uploading (e.g. ffmpeg -i input.wav -ar 16000 out.wav)."
        )
    return samples


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_server_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="sherox ASR HTTP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU thread count for ONNX runtime, per loaded language")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host address to listen on")
    parser.add_argument("--port", type=int, default=8200,
                        help="Port to listen on")
    parser.add_argument("--sample-rate", type=int, default=16000, metavar="HZ",
                        help="Expected audio sample rate for incoming audio")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Uvicorn log level")
    return parser.parse_args()


def main() -> None:
    _run_cli(_main_impl)


def _main_impl() -> None:
    args = parse_server_args()
    project_dir = Path(__file__).resolve().parent.parent
    app = _create_app(project_dir, args.threads, args.sample_rate)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":  # pragma: no cover
    main()
