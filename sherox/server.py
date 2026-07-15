"""sherox.server — HTTP/WebSocket ASR API server.

Usage:
    # Offline (VAD-segmented), Cohere Transcribe multilingual:
    sherox.server --offline --model-type cohere_transcribe --language ja

    # Default English Parakeet int8:
    sherox.server --host 0.0.0.0 --port 8000

    # Custom host/port with Whisper:
    sherox.server --offline --model-type whisper --language en --host 0.0.0.0 --port 8000

API endpoints:
    GET  /health       → {"status":"ok","model":"...","mode":"offline|online"}
    POST /transcribe   → multipart WAV upload → {"text":"..."}
    WS   /ws           → binary int16 PCM chunks in, JSON text frames out
                         {"type":"segment","text":"..."}  (finalized utterance)
                         {"type":"partial","text":"..."}  (live hypothesis, online only)
    GET  /static/      → browser demo page (mic streaming + file upload)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

try:
    from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.routing import APIRouter
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "Server dependencies not installed. "
        "Run: pip install 'sherox[server]'"
    ) from _e

from .asr import _validate_model as _asr_validate_model
from .asr import _validate_vad as _asr_validate_vad
from .asr_engine import (
    build_offline_recognizer,
    build_recognizer,
    build_vad,
    warmup_offline_recognizer,
)
from .config import Config
from .streaming import _run_asr
from .utils import run_cli as _run_cli

logger = logging.getLogger(__name__)


# ── App state ─────────────────────────────────────────────────────────────────

@dataclass
class _AppState:
    recognizer: Any
    cfg: Config
    project_dir: Path
    mode: str                            # "offline" | "online"
    model_name: str
    online_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    online_thread_lock: threading.Lock = field(default_factory=threading.Lock)


# ── Validation wrappers ───────────────────────────────────────────────────────

def _startup_validate_model(model_dir: str, model_type: str) -> None:
    """Validate / download model; re-raise sys.exit as RuntimeError."""
    try:
        _asr_validate_model(model_dir, model_type)
    except SystemExit as exc:
        raise RuntimeError(f"Model not found or download failed: {exc}") from exc


def _startup_validate_vad(
    vad_type: str, ten_vad_model: str, offline: bool, project_dir: Path
) -> str:
    """Validate / download VAD model; re-raise sys.exit as RuntimeError."""
    try:
        return _asr_validate_vad(vad_type, ten_vad_model, offline, project_dir)
    except SystemExit as exc:
        raise RuntimeError(f"VAD validation failed: {exc}") from exc


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg: Config = app.state.cfg
    project_dir: Path = app.state.project_dir
    loop = asyncio.get_event_loop()

    await loop.run_in_executor(None, _startup_validate_model, cfg.model_dir, cfg.model_type)

    if cfg.offline:
        cfg.vad_model = await loop.run_in_executor(
            None, _startup_validate_vad,
            cfg.vad_type, cfg.ten_vad_model, True, project_dir,
        )
        recognizer = await loop.run_in_executor(None, build_offline_recognizer, cfg)
        await loop.run_in_executor(None, warmup_offline_recognizer, recognizer, cfg.sample_rate)
        mode = "offline"
    else:
        recognizer = await loop.run_in_executor(None, build_recognizer, cfg)
        mode = "online"

    model_name = Path(cfg.model_dir).name
    app.state.asr = _AppState(
        recognizer=recognizer,
        cfg=cfg,
        project_dir=project_dir,
        mode=mode,
        model_name=model_name,
    )
    logger.info("Model '%s' ready (%s mode).", model_name, mode)
    yield
    app.state.asr = None


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter()


# ── App factory ───────────────────────────────────────────────────────────────

def _create_app(cfg: Config, project_dir: Path) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="sherox ASR server", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.project_dir = project_dir

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(static_dir), html=True),
            name="static",
        )

    return app


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health(request: Request):
    """Return server status and loaded model information."""
    asr: _AppState = request.app.state.asr
    return {"status": "ok", "model": asr.model_name, "mode": asr.mode}


@router.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...)):
    """Transcribe an uploaded audio file (WAV, FLAC, OGG…).

    The audio must be at the server's configured sample rate (default 16 kHz).
    Stereo files are automatically downmixed to mono.
    Returns ``{"text": "<transcription>"}`` on success.
    """
    asr: _AppState = request.app.state.asr
    raw = await file.read()
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(None, lambda: _transcribe_bytes(raw, asr))
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    return {"text": text}


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Real-time ASR over WebSocket.

    **Audio input (client → server):** binary frames of raw int16 PCM,
    mono, at the server's configured sample rate (default 16 kHz).
    Recommended chunk size: 160–2560 samples (10–160 ms).

    **Text output (server → client):** JSON text frames:

    - ``{"type": "segment", "text": "..."}``  — finalized utterance
    - ``{"type": "partial", "text": "..."}``  — live hypothesis (online mode only)
    """
    await ws.accept()
    asr: _AppState = ws.app.state.asr
    if asr.mode == "offline":
        await _handle_ws_offline(ws, asr)
    else:
        await _handle_ws_online(ws, asr)


# ── Inference helpers ─────────────────────────────────────────────────────────

def _int16_to_float32(data: bytes) -> np.ndarray:
    """Convert raw int16 PCM bytes to float32 in [-1.0, 1.0]."""
    if len(data) % 2 != 0:
        data = data[:-1]  # drop stray trailing byte
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribe_bytes(raw: bytes, asr: _AppState) -> str:
    """Decode uploaded audio bytes and run ASR. Runs in a thread-pool executor."""
    import soundfile as sf  # noqa: PLC0415
    samples, sr = sf.read(BytesIO(raw), dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)  # stereo → mono
    if sr != asr.cfg.sample_rate:
        raise ValueError(
            f"Audio must be {asr.cfg.sample_rate} Hz, got {sr} Hz. "
            "Resample before uploading (e.g. ffmpeg -i input.wav -ar 16000 out.wav)."
        )
    if asr.mode == "offline":
        return _run_asr(asr.recognizer, samples, sr)
    return _online_transcribe_sync(samples, sr, asr)


def _online_transcribe_sync(samples: np.ndarray, sr: int, asr: _AppState) -> str:
    """Thread-safe single-pass transcription with the online recognizer."""
    with asr.online_thread_lock:
        stream = asr.recognizer.create_stream()
        stream.accept_waveform(sr, samples)
        tail = np.zeros(int(sr * 0.5), dtype=np.float32)  # half-second silence to force endpoint
        stream.accept_waveform(sr, tail)
        while asr.recognizer.is_ready(stream):
            asr.recognizer.decode_stream(stream)
        return asr.recognizer.get_result(stream).strip()


# ── WebSocket handlers ────────────────────────────────────────────────────────

async def _handle_ws_offline(ws: WebSocket, asr: _AppState) -> None:
    """Per-connection handler for offline (VAD-segmented) mode.

    Each connection owns its own VAD instance (VAD is stateful).
    The shared recognizer is thread-safe for offline inference because
    _run_asr() creates a fresh stream per call.
    """
    loop = asyncio.get_event_loop()
    vad = await loop.run_in_executor(None, build_vad, asr.cfg)

    async def _decode_and_send(samples: np.ndarray) -> None:
        text = await loop.run_in_executor(
            None, _run_asr, asr.recognizer, samples, asr.cfg.sample_rate
        )
        if text:
            try:
                await ws.send_text(json.dumps({"type": "segment", "text": text}))
            except Exception:
                pass  # client already disconnected

    async def _drain_vad() -> None:
        while not vad.empty():
            segment = vad.front
            samples = np.array(segment.samples, dtype=np.float32)
            vad.pop()
            await _decode_and_send(samples)

    try:
        while True:
            data = await ws.receive_bytes()
            chunk = _int16_to_float32(data)
            vad.accept_waveform(chunk)
            await _drain_vad()
    except WebSocketDisconnect:
        pass
    finally:
        vad.flush()
        await _drain_vad()


async def _handle_ws_online(ws: WebSocket, asr: _AppState) -> None:
    """Per-connection handler for online (streaming) mode.

    Each connection owns its own stream object. The shared recognizer is
    guarded by asr.online_lock so concurrent connections don't race on
    decode_stream. Partial hypotheses are streamed as they change;
    endpoint detection triggers a final segment message.
    """
    loop = asyncio.get_event_loop()
    stream = asr.recognizer.create_stream()
    last_partial = ""

    try:
        while True:
            data = await ws.receive_bytes()
            chunk = _int16_to_float32(data)

            # Hold the lock only for the decode + endpoint check + reset
            # sequence, which must be atomic w.r.t. the shared recognizer.
            # Network sends happen outside the lock so a slow client cannot
            # stall decoding for other connections.
            async with asr.online_lock:
                stream.accept_waveform(asr.cfg.sample_rate, chunk)
                while asr.recognizer.is_ready(stream):
                    await loop.run_in_executor(None, asr.recognizer.decode_stream, stream)
                text = asr.recognizer.get_result(stream).strip()
                is_ep = asr.recognizer.is_endpoint(stream)
                if is_ep and text:
                    asr.recognizer.reset(stream)

            if is_ep and text:
                await ws.send_text(json.dumps({"type": "segment", "text": text}))
                last_partial = ""
            elif text and text != last_partial:
                await ws.send_text(json.dumps({"type": "partial", "text": text}))
                last_partial = text

    except WebSocketDisconnect:
        pass


# ── Model directory constants (mirrors asr.py) ────────────────────────────────

_OFFLINE_ONLY_TYPES = frozenset({
    "nemo_transducer", "whisper", "nemo_ctc", "sense_voice", "moonshine",
    "fire_red_asr", "cohere_transcribe", "ja", "ja-en", "ja-en-mls-5k",
})
_OFFLINE_ONLY_NAME_PATTERNS = (
    "parakeet", "nemo", "whisper", "sense_voice", "moonshine",
    "fire_red_asr", "cohere", "reazonspeech",
)
_MODEL_TARGET = "zipformer-en-2023"
_PARAKEET_TARGET = "parakeet-tdt-0.6b-v2-int8"
_COHERE_TARGET = "cohere-transcribe-14-lang-int8"
_REAZON_JA_TARGET = "reazonspeech-ja"
_REAZON_JA_EN_TARGET = "reazonspeech-ja-en"
_REAZON_JA_EN_MLS_TARGET = "reazonspeech-ja-en-mls-5k"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_server_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="sherox ASR HTTP/WebSocket server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model args — mirrors sherox.asr
    parser.add_argument("--model-dir", default=None, metavar="PATH",
                        help="Path to model directory (auto-selected if omitted)")
    parser.add_argument("--model-type", default="", metavar="TYPE",
                        help="Model architecture hint (blank = auto-detect)")
    parser.add_argument("--offline", action="store_true",
                        help="Use VAD-segmented offline pipeline "
                             "(required for Whisper, NeMo, Cohere, etc.)")
    parser.add_argument("--sample-rate", type=int, default=16000, metavar="HZ",
                        help="Expected audio sample rate for incoming audio")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU thread count for ONNX runtime")
    parser.add_argument("--language", default="en", metavar="LANG",
                        help="Language code for Whisper/SenseVoice/Cohere (e.g. en, zh, ja)")
    parser.add_argument("--vad-model", dest="vad_type", default="silero",
                        choices=["silero", "ten-vad"],
                        help="VAD type for offline segmentation")
    parser.add_argument("--ten-vad-model", default="ten-vad.int8.onnx",
                        choices=["ten-vad.onnx", "ten-vad.int8.onnx"],
                        help="Ten-VAD model variant")
    # Server args
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host address to listen on")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Uvicorn log level")
    return parser.parse_args()


def main() -> None:
    _run_cli(_main_impl)


def _main_impl() -> None:
    args = parse_server_args()
    args.model_type = args.model_type.lower()

    project_dir = Path(__file__).resolve().parent.parent

    # Resolve model directory (mirrors asr.py logic)
    if args.model_dir is None:
        if args.model_type == "ja":
            raw_model_dir = f"models/{_REAZON_JA_TARGET}"
        elif args.model_type == "ja-en":
            raw_model_dir = f"models/{_REAZON_JA_EN_TARGET}"
        elif args.model_type == "ja-en-mls-5k":
            raw_model_dir = f"models/{_REAZON_JA_EN_MLS_TARGET}"
        elif args.model_type == "cohere_transcribe":
            raw_model_dir = f"models/{_COHERE_TARGET}"
        elif (
            args.offline
            or args.model_type == "nemo_transducer"
            or args.language.lower() in {"en", "eng"}
        ):
            raw_model_dir = f"models/{_PARAKEET_TARGET}"
        else:
            raw_model_dir = f"models/{_MODEL_TARGET}"
    else:
        raw_model_dir = args.model_dir
    model_dir = Path(raw_model_dir)
    if not model_dir.is_absolute():
        model_dir = project_dir / model_dir

    # Auto-detect offline-only models
    offline = args.offline
    model_name_lower = model_dir.name.lower()
    if not offline and (
        args.model_type in _OFFLINE_ONLY_TYPES
        or any(pat in model_name_lower for pat in _OFFLINE_ONLY_NAME_PATTERNS)
    ):
        offline = True

    # Remap ReazonSpeech aliases so sherpa-onnx uses auto-detect
    model_type = args.model_type
    if model_type in {"ja", "ja-en", "ja-en-mls-5k"}:
        model_type = ""

    cfg = Config(
        model_dir=str(model_dir),
        sample_rate=args.sample_rate,
        num_threads=args.threads,
        model_type=model_type,
        offline=offline,
        vad_type=args.vad_type,
        ten_vad_model=args.ten_vad_model,
        language=args.language,
    )

    app = _create_app(cfg, project_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":  # pragma: no cover
    main()
