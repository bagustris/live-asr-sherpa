"""Tests for sherox.tts_server — FastAPI HTTP TTS API."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Attempt to import FastAPI test deps; skip entire module if unavailable ────
pytest.importorskip("fastapi", reason="fastapi not installed (pip install 'sherox[server]')")
pytest.importorskip("httpx", reason="httpx not installed (pip install httpx)")

from fastapi.testclient import TestClient  # noqa: E402

import sherox.tts_server as tts_srv  # noqa: E402


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_samples(sr: int = 22050, duration: float = 0.3) -> tuple[np.ndarray, int]:
    return np.zeros(int(sr * duration), dtype=np.float32), sr


@pytest.fixture()
def client():
    """TestClient backed by mocked build_tts/synthesise_to_file (no model I/O)."""
    mock_tts = MagicMock()

    with (
        patch.object(tts_srv, "build_tts", return_value=mock_tts) as mock_build,
        patch.object(tts_srv, "synthesise_to_file", return_value=_make_samples()) as mock_synth,
    ):
        app = tts_srv._create_app(Path("/fake/project"), num_threads=1)
        with TestClient(app) as c:
            yield c, mock_build, mock_synth


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        c, _, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200

    def test_no_languages_loaded_before_first_request(self, client):
        c, _, _ = client
        assert c.get("/health").json() == {"status": "ok", "languages_loaded": []}

    def test_lists_language_after_synthesize(self, client):
        c, _, _ = client
        c.post("/synthesize", json={"text": "hello", "lang": "eng"})
        assert c.get("/health").json()["languages_loaded"] == ["eng"]


# ── /synthesize ───────────────────────────────────────────────────────────────

class TestSynthesize:
    def test_returns_wav_audio(self, client):
        c, _, _ = client
        resp = c.post("/synthesize", json={"text": "hello world", "lang": "eng"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content[:4] == b"RIFF"

    def test_missing_text_is_422(self, client):
        c, _, _ = client
        resp = c.post("/synthesize", json={"lang": "eng"})
        assert resp.status_code == 422

    def test_empty_text_is_422(self, client):
        c, _, _ = client
        resp = c.post("/synthesize", json={"text": "", "lang": "eng"})
        assert resp.status_code == 422

    def test_defaults_lang_to_eng(self, client):
        c, mock_build, _ = client
        c.post("/synthesize", json={"text": "hello"})
        mock_build.assert_called_once()

    def test_backend_failure_is_500(self, client):
        c, _, mock_synth = client
        mock_synth.side_effect = RuntimeError("boom")
        resp = c.post("/synthesize", json={"text": "hello", "lang": "eng"})
        assert resp.status_code == 500

    def test_builds_each_language_at_most_once(self, client):
        c, mock_build, _ = client
        c.post("/synthesize", json={"text": "hello", "lang": "eng"})
        c.post("/synthesize", json={"text": "world", "lang": "eng"})
        c.post("/synthesize", json={"text": "hallo", "lang": "deu"})
        assert mock_build.call_count == 2  # once for "eng", once for "deu"

    def test_language_aliases_share_cache(self, client):
        """'en' and 'eng' both resolve to the Piper 'eng' model and should
        not trigger separate builds."""
        c, mock_build, _ = client
        c.post("/synthesize", json={"text": "hello", "lang": "en"})
        c.post("/synthesize", json={"text": "hello", "lang": "eng"})
        assert mock_build.call_count == 1


# ── CLI ───────────────────────────────────────────────────────────────────────

class TestParseServerArgs:
    def test_defaults(self):
        with patch("sys.argv", ["sherox.tts_server"]):
            args = tts_srv.parse_server_args()
        assert args.host == "0.0.0.0"
        assert args.port == 8100
        assert args.threads == 1

    def test_custom_port_and_threads(self):
        with patch("sys.argv", ["sherox.tts_server", "--port", "9000", "--threads", "4"]):
            args = tts_srv.parse_server_args()
        assert args.port == 9000
        assert args.threads == 4
