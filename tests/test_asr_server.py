"""Tests for sherox.asr_server — FastAPI HTTP ASR API."""
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Attempt to import FastAPI test deps; skip entire module if unavailable ────
pytest.importorskip("fastapi", reason="fastapi not installed (pip install 'sherox[server]')")
pytest.importorskip("httpx", reason="httpx not installed (pip install httpx)")

from fastapi.testclient import TestClient  # noqa: E402

import sherox.asr_server as asr_srv  # noqa: E402


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_recognizer(text: str = "hello world") -> MagicMock:
    """Return a mock offline recognizer whose streams return *text*."""
    rec = MagicMock()
    stream = MagicMock()
    stream.result.text = text
    rec.create_stream.return_value = stream
    return rec


def _make_wav_bytes(sr: int = 16000, duration: float = 0.3) -> bytes:
    """Create an in-memory WAV file for upload tests."""
    import soundfile as sf  # noqa: PLC0415
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    buf.seek(0)
    return buf.read()


@pytest.fixture()
def client():
    """TestClient backed by a mocked recognizer build chain (no model I/O)."""
    mock_rec = _make_recognizer()

    with (
        patch.object(
            asr_srv, "resolve_language_model", return_value=("", "fake-model", True),
        ),
        patch.object(asr_srv, "ensure_model"),
        patch.object(
            asr_srv, "build_offline_recognizer", return_value=mock_rec,
        ) as mock_build,
        patch.object(asr_srv, "warmup_offline_recognizer"),
    ):
        app = asr_srv._create_app(Path("/fake/project"), num_threads=1, sample_rate=16000)
        with TestClient(app) as c:
            yield c, mock_build


def _post_wav(c, wav: bytes, lang: str | None = None):
    data = {} if lang is None else {"lang": lang}
    return c.post(
        "/transcribe", files={"file": ("audio.wav", wav, "audio/wav")}, data=data,
    )


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200

    def test_no_languages_loaded_before_first_request(self, client):
        c, _ = client
        assert c.get("/health").json() == {"status": "ok", "languages_loaded": []}

    def test_lists_language_after_transcribe(self, client):
        c, _ = client
        _post_wav(c, _make_wav_bytes(), lang="en")
        assert c.get("/health").json()["languages_loaded"] == ["en"]


# ── /transcribe ───────────────────────────────────────────────────────────────

class TestTranscribe:
    def test_returns_text(self, client):
        c, _ = client
        resp = _post_wav(c, _make_wav_bytes())
        assert resp.status_code == 200
        assert resp.json() == {"text": "hello world", "lang": "en"}

    def test_defaults_lang_to_en(self, client):
        c, _ = client
        resp = _post_wav(c, _make_wav_bytes())
        assert resp.json()["lang"] == "en"

    def test_missing_file_is_422(self, client):
        c, _ = client
        resp = c.post("/transcribe", data={"lang": "en"})
        assert resp.status_code == 422

    def test_wrong_sample_rate_is_422(self, client):
        c, _ = client
        resp = _post_wav(c, _make_wav_bytes(sr=8000))
        assert resp.status_code == 422
        assert "16000 Hz" in resp.json()["detail"]

    def test_unreadable_audio_is_422(self, client):
        c, _ = client
        resp = c.post(
            "/transcribe", files={"file": ("audio.wav", b"not audio", "audio/wav")},
        )
        assert resp.status_code == 422

    def test_backend_failure_is_500(self, client):
        c, mock_build = client
        mock_build.side_effect = RuntimeError("boom")
        resp = _post_wav(c, _make_wav_bytes())
        assert resp.status_code == 500

    def test_builds_each_language_at_most_once(self, client):
        c, mock_build = client
        _post_wav(c, _make_wav_bytes(), lang="en")
        _post_wav(c, _make_wav_bytes(), lang="en")
        _post_wav(c, _make_wav_bytes(), lang="de")
        assert mock_build.call_count == 2  # once for "en", once for "de"

    def test_language_aliases_share_cache(self, client):
        """'en' and 'eng' resolve to the same ASR model and should not
        trigger separate builds."""
        c, mock_build = client
        _post_wav(c, _make_wav_bytes(), lang="en")
        _post_wav(c, _make_wav_bytes(), lang="eng")
        assert mock_build.call_count == 1


# ── CLI ───────────────────────────────────────────────────────────────────────

class TestParseServerArgs:
    def test_defaults(self):
        with patch("sys.argv", ["sherox.asr_server"]):
            args = asr_srv.parse_server_args()
        assert args.host == "0.0.0.0"
        assert args.port == 8200
        assert args.threads == 1
        assert args.sample_rate == 16000

    def test_custom_port_and_threads(self):
        with patch("sys.argv", ["sherox.asr_server", "--port", "9000", "--threads", "4"]):
            args = asr_srv.parse_server_args()
        assert args.port == 9000
        assert args.threads == 4
