"""sherox.tts_server — HTTP TTS API server.

Usage:
    # Default host/port:
    sherox.tts_server

    # Custom host/port, more ONNX threads:
    sherox.tts_server --host 0.0.0.0 --port 8100 --threads 2

API endpoints:
    GET  /health       -> {"status": "ok", "languages_loaded": [...]}
    POST /synthesize    -> JSON {"text": "...", "lang": "eng", "speed": 1.0, "speaker_id": 0}
                          -> WAV audio (audio/wav)

TTS backends are built lazily per language on first request and cached for
the lifetime of the process (mirrors sherox.tts's own CLI caching and
proscor.tts's _get_tts pattern), so repeat calls skip the model-load cost
that a fresh CLI/subprocess invocation would otherwise pay every time.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import threading
from pathlib import Path
from typing import Any

import soundfile

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from fastapi.routing import APIRouter
    import uvicorn
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "Server dependencies not installed. "
        "Run: pip install 'sherox[server]'"
    ) from _e

from .config import TtsConfig
from .tts import _normalize_language, build_tts, synthesise_to_file
from .utils import run_cli as _run_cli

logger = logging.getLogger(__name__)


# ── App state ─────────────────────────────────────────────────────────────────

class _TtsCache:
    """Per-language TTS backend cache, built lazily on first request.

    A single lock serializes *builds* (first use of a language) so concurrent
    requests for a not-yet-loaded language don't race into downloading/
    building the same model twice; already-cached languages are read without
    contention.
    """

    def __init__(self, project_dir: Path, num_threads: int) -> None:
        self.project_dir = project_dir
        self.num_threads = num_threads
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, lang: str):
        lang = _normalize_language(lang)
        if lang not in self._models:
            with self._lock:
                if lang not in self._models:
                    cfg = TtsConfig(
                        model_dir="", language=lang, num_threads=self.num_threads,
                    )
                    self._models[lang] = build_tts(cfg, self.project_dir)
        return self._models[lang]

    def loaded_languages(self) -> list[str]:
        return sorted(self._models.keys())


# ── App factory ───────────────────────────────────────────────────────────────

def _create_app(project_dir: Path, num_threads: int) -> FastAPI:
    app = FastAPI(title="sherox TTS server", version="0.1.0")
    app.state.tts_cache = _TtsCache(project_dir, num_threads)

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
    cache: _TtsCache = request.app.state.tts_cache
    return {"status": "ok", "languages_loaded": cache.loaded_languages()}


@router.post("/synthesize")
async def synthesize(request: Request):
    """Synthesize `text` and return WAV audio.

    Body: {"text": str, "lang": str = "eng", "speed": float = 1.0, "speaker_id": int = 0}
    """
    body = await request.json()
    text = body.get("text", "")
    lang = body.get("lang", "eng")
    speed = float(body.get("speed", 1.0))
    speaker_id = int(body.get("speaker_id", 0))

    if not text:
        raise HTTPException(status_code=422, detail="'text' is required")

    cache: _TtsCache = request.app.state.tts_cache
    loop = asyncio.get_event_loop()

    try:
        tts = await loop.run_in_executor(None, cache.get, lang)
        cfg = TtsConfig(
            model_dir="", language=lang, speaker_id=speaker_id, speed=speed,
            output="", play=False, no_save=True, num_threads=cache.num_threads,
        )
        samples, sample_rate = await loop.run_in_executor(
            None, synthesise_to_file, tts, text, cfg,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    buf = io.BytesIO()
    soundfile.write(buf, samples, samplerate=sample_rate, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_server_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="sherox TTS HTTP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU thread count for ONNX runtime, per loaded language")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host address to listen on")
    parser.add_argument("--port", type=int, default=8100,
                        help="Port to listen on")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Uvicorn log level")
    return parser.parse_args()


def main() -> None:
    _run_cli(_main_impl)


def _main_impl() -> None:
    args = parse_server_args()
    project_dir = Path(__file__).resolve().parent.parent
    app = _create_app(project_dir, args.threads)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":  # pragma: no cover
    main()
