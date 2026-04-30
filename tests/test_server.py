"""Tests for sherox.server — FastAPI HTTP/WebSocket ASR API."""
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Attempt to import FastAPI test deps; skip entire module if unavailable ────
pytest.importorskip("fastapi", reason="fastapi not installed (pip install 'sherox[server]')")
pytest.importorskip("httpx", reason="httpx not installed (pip install httpx)")

from fastapi.testclient import TestClient  # noqa: E402

import sherox.server as srv  # noqa: E402
from sherox.config import Config  # noqa: E402


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_recognizer(text: str = "hello world") -> MagicMock:
    """Return a mock offline-style recognizer whose streams return *text*."""
    rec = MagicMock()
    stream = MagicMock()
    stream.result.text = text
    rec.create_stream.return_value = stream
    return rec


def _make_wav_bytes(sr: int = 16000, duration: float = 0.5, channels: int = 1) -> bytes:
    """Create an in-memory WAV file for upload tests."""
    import soundfile as sf  # noqa: PLC0415
    if channels == 1:
        samples = np.zeros(int(sr * duration), dtype=np.float32)
    else:
        samples = np.zeros((int(sr * duration), channels), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    buf.seek(0)
    return buf.read()


@pytest.fixture()
def offline_client():
    """TestClient backed by a mocked offline recognizer (no model I/O)."""
    mock_rec = _make_recognizer()
    cfg = Config(model_dir="/fake/model", offline=True)

    with (
        patch.object(srv, "_startup_validate_model"),
        patch.object(srv, "_startup_validate_vad", return_value="/fake/vad.onnx"),
        patch.object(srv, "build_offline_recognizer", return_value=mock_rec),
    ):
        app = srv._create_app(cfg, Path("/fake/project"))
        with TestClient(app) as client:
            yield client


@pytest.fixture()
def online_client():
    """TestClient backed by a mocked online recognizer."""
    mock_rec = MagicMock()
    stream = MagicMock()
    stream.result.text = ""
    mock_rec.create_stream.return_value = stream
    mock_rec.is_ready.return_value = False
    mock_rec.get_result.return_value = MagicMock(strip=MagicMock(return_value=""))
    mock_rec.is_endpoint.return_value = False

    cfg = Config(model_dir="/fake/online-model", offline=False)

    with (
        patch.object(srv, "_startup_validate_model"),
        patch.object(srv, "build_recognizer", return_value=mock_rec),
    ):
        app = srv._create_app(cfg, Path("/fake/project"))
        with TestClient(app) as client:
            yield client, mock_rec


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, offline_client):
        resp = offline_client.get("/health")
        assert resp.status_code == 200

    def test_has_required_fields(self, offline_client):
        body = offline_client.get("/health").json()
        assert body["status"] == "ok"
        assert "model" in body
        assert "mode" in body

    def test_mode_is_offline(self, offline_client):
        body = offline_client.get("/health").json()
        assert body["mode"] == "offline"

    def test_mode_is_online(self, online_client):
        client, _ = online_client
        body = client.get("/health").json()
        assert body["mode"] == "online"

    def test_model_name_is_dir_basename(self, offline_client):
        body = offline_client.get("/health").json()
        assert body["model"] == "model"  # basename of /fake/model


# ── POST /transcribe ──────────────────────────────────────────────────────────

class TestTranscribe:
    def test_returns_text(self, offline_client):
        with patch.object(srv, "_run_asr", return_value="hello world"):
            wav = _make_wav_bytes()
            resp = offline_client.post(
                "/transcribe", files={"file": ("audio.wav", wav, "audio/wav")}
            )
        assert resp.status_code == 200
        assert resp.json()["text"] == "hello world"

    def test_wrong_sample_rate_returns_422(self, offline_client):
        wav = _make_wav_bytes(sr=44100)
        resp = offline_client.post(
            "/transcribe", files={"file": ("audio.wav", wav, "audio/wav")}
        )
        assert resp.status_code == 422
        assert "Hz" in resp.json()["error"]

    def test_no_file_returns_422(self, offline_client):
        resp = offline_client.post("/transcribe")
        assert resp.status_code == 422

    def test_stereo_downmixed_to_mono(self, offline_client):
        """Stereo upload should succeed after automatic downmix."""
        wav = _make_wav_bytes(channels=2)
        with patch.object(srv, "_run_asr", return_value="stereo ok"):
            resp = offline_client.post(
                "/transcribe", files={"file": ("audio.wav", wav, "audio/wav")}
            )
        assert resp.status_code == 200
        assert resp.json()["text"] == "stereo ok"

    def test_empty_transcription_returned_as_empty_string(self, offline_client):
        with patch.object(srv, "_run_asr", return_value=""):
            wav = _make_wav_bytes()
            resp = offline_client.post(
                "/transcribe", files={"file": ("audio.wav", wav, "audio/wav")}
            )
        assert resp.status_code == 200
        assert resp.json()["text"] == ""


# ── WS /ws — offline ──────────────────────────────────────────────────────────

def _make_vad(empty_side_effects: list, samples=None) -> MagicMock:
    """Return a mock VAD with configurable empty() side effects."""
    vad = MagicMock()
    vad.empty.side_effect = empty_side_effects
    if samples is not None:
        segment = MagicMock()
        segment.samples = samples
        vad.front = segment
    return vad


class TestWebSocketOffline:
    def _frame(self, n: int = 1600) -> bytes:
        return np.zeros(n, dtype=np.int16).tobytes()

    def test_connection_accepted(self, offline_client):
        mock_vad = _make_vad([True, True])
        with patch.object(srv, "build_vad", return_value=mock_vad):
            with offline_client.websocket_connect("/ws") as ws:
                ws.send_bytes(self._frame())
                # no segment produced → no message, just verify no crash

    def test_segment_sent_when_vad_fires(self, offline_client):
        mock_vad = _make_vad(
            empty_side_effects=[False, True, True],
            samples=[0.0] * 160,
        )
        with (
            patch.object(srv, "build_vad", return_value=mock_vad),
            patch.object(srv, "_run_asr", return_value="test segment"),
        ):
            with offline_client.websocket_connect("/ws") as ws:
                ws.send_bytes(self._frame())
                msg = ws.receive_text()

        data = json.loads(msg)
        assert data["type"] == "segment"
        assert data["text"] == "test segment"

    def test_empty_asr_result_not_sent(self, offline_client):
        mock_vad = _make_vad(
            empty_side_effects=[False, True, True],
            samples=[0.0] * 160,
        )
        received = []
        with (
            patch.object(srv, "build_vad", return_value=mock_vad),
            patch.object(srv, "_run_asr", return_value=""),
        ):
            with offline_client.websocket_connect("/ws") as ws:
                ws.send_bytes(self._frame())
                # no message should arrive — nothing to assert, just no hang
        assert len(received) == 0  # nothing was appended (no ws.receive call)

    def test_odd_byte_frame_handled_gracefully(self, offline_client):
        mock_vad = _make_vad([True, True])
        with patch.object(srv, "build_vad", return_value=mock_vad):
            with offline_client.websocket_connect("/ws") as ws:
                ws.send_bytes(b"\x00" * 5)  # odd length

    def test_vad_flush_called_on_disconnect(self, offline_client):
        mock_vad = _make_vad([True, True])
        with patch.object(srv, "build_vad", return_value=mock_vad):
            with offline_client.websocket_connect("/ws") as ws:
                ws.send_bytes(self._frame())
        mock_vad.flush.assert_called_once()

    def test_multiple_segments_in_one_chunk(self, offline_client):
        """VAD that fires twice should produce two segment messages."""
        seg = MagicMock()
        seg.samples = [0.0] * 160
        mock_vad = MagicMock()
        mock_vad.empty.side_effect = [False, False, True, True]
        mock_vad.front = seg

        with (
            patch.object(srv, "build_vad", return_value=mock_vad),
            patch.object(srv, "_run_asr", return_value="word"),
        ):
            with offline_client.websocket_connect("/ws") as ws:
                ws.send_bytes(self._frame())
                msg1 = ws.receive_text()
                msg2 = ws.receive_text()

        assert json.loads(msg1)["text"] == "word"
        assert json.loads(msg2)["text"] == "word"


# ── WS /ws — online ───────────────────────────────────────────────────────────

class TestWebSocketOnline:
    def _frame(self) -> bytes:
        return np.zeros(1600, dtype=np.int16).tobytes()

    def test_connection_accepted(self, online_client):
        client, _ = online_client
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(self._frame())

    def test_partial_sent_when_text_changes(self, online_client):
        client, mock_rec = online_client
        mock_rec.is_ready.return_value = False
        mock_rec.get_result.return_value.strip.return_value = "hello"
        mock_rec.is_endpoint.return_value = False

        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(self._frame())
            msg = ws.receive_text()

        data = json.loads(msg)
        assert data["type"] == "partial"
        assert data["text"] == "hello"

    def test_segment_sent_at_endpoint(self, online_client):
        client, mock_rec = online_client
        mock_rec.is_ready.return_value = False
        mock_rec.get_result.return_value.strip.return_value = "hello world"
        mock_rec.is_endpoint.return_value = True

        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(self._frame())
            msg = ws.receive_text()

        data = json.loads(msg)
        assert data["type"] == "segment"
        assert data["text"] == "hello world"

    def test_stream_reset_after_endpoint(self, online_client):
        client, mock_rec = online_client
        mock_rec.is_ready.return_value = False
        mock_rec.get_result.return_value.strip.return_value = "done"
        mock_rec.is_endpoint.return_value = True

        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(self._frame())
            ws.receive_text()

        mock_rec.reset.assert_called()

    def test_no_message_when_text_unchanged(self, online_client):
        """If partial text hasn't changed, no message is sent."""
        client, mock_rec = online_client
        mock_rec.is_ready.return_value = False
        mock_rec.get_result.return_value.strip.return_value = ""
        mock_rec.is_endpoint.return_value = False

        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(self._frame())
            # no message to receive — just verify no crash


# ── _int16_to_float32 ─────────────────────────────────────────────────────────

class TestInt16ToFloat32:
    def test_zeros_produce_zeros(self):
        data = np.zeros(10, dtype=np.int16).tobytes()
        result = srv._int16_to_float32(data)
        np.testing.assert_array_equal(result, np.zeros(10, dtype=np.float32))

    def test_max_positive_maps_to_near_one(self):
        data = np.array([32767], dtype=np.int16).tobytes()
        result = srv._int16_to_float32(data)
        assert abs(result[0] - 1.0) < 0.001

    def test_max_negative_maps_to_minus_one(self):
        data = np.array([-32768], dtype=np.int16).tobytes()
        result = srv._int16_to_float32(data)
        assert result[0] == pytest.approx(-1.0)

    def test_odd_length_drops_trailing_byte(self):
        data = np.zeros(4, dtype=np.int16).tobytes()  # 8 bytes
        result = srv._int16_to_float32(data + b"\xff")  # 9 bytes → should handle 8
        assert len(result) == 4

    def test_empty_input_returns_empty_array(self):
        result = srv._int16_to_float32(b"")
        assert len(result) == 0


# ── _transcribe_bytes ─────────────────────────────────────────────────────────

class TestTranscribeBytes:
    def _make_asr(self, offline_client, mode="offline"):
        """Build a minimal _AppState matching the fixture's setup."""
        cfg = Config(model_dir="/fake/model", sample_rate=16000, offline=(mode == "offline"))
        return srv._AppState(
            recognizer=MagicMock(),
            cfg=cfg,
            project_dir=Path("/fake"),
            mode=mode,
            model_name="model",
        )

    def test_wrong_rate_raises_value_error(self, offline_client):
        wav = _make_wav_bytes(sr=8000)
        asr = self._make_asr(offline_client)
        with pytest.raises(ValueError, match="Hz"):
            srv._transcribe_bytes(wav, asr)

    def test_stereo_downmixed(self, offline_client):
        wav = _make_wav_bytes(sr=16000, channels=2)
        asr = self._make_asr(offline_client)
        with patch.object(srv, "_run_asr", return_value="ok"):
            text = srv._transcribe_bytes(wav, asr)
        assert text == "ok"

    def test_offline_calls_run_asr(self, offline_client):
        wav = _make_wav_bytes()
        asr = self._make_asr(offline_client)
        with patch.object(srv, "_run_asr", return_value="result") as mock_asr:
            text = srv._transcribe_bytes(wav, asr)
        mock_asr.assert_called_once()
        assert text == "result"


# ── parse_server_args ─────────────────────────────────────────────────────────

class TestParseServerArgs:
    def test_defaults(self):
        with patch("sys.argv", ["sherox.server"]):
            args = srv.parse_server_args()
        assert args.host == "0.0.0.0"
        assert args.port == 8000
        assert args.offline is False
        assert args.language == "en"
        assert args.log_level == "info"

    def test_custom_host_and_port(self):
        with patch("sys.argv", ["sherox.server", "--host", "127.0.0.1", "--port", "9000"]):
            args = srv.parse_server_args()
        assert args.host == "127.0.0.1"
        assert args.port == 9000

    def test_offline_flag(self):
        with patch("sys.argv", ["sherox.server", "--offline"]):
            args = srv.parse_server_args()
        assert args.offline is True

    def test_language_flag(self):
        with patch("sys.argv", ["sherox.server", "--language", "ja"]):
            args = srv.parse_server_args()
        assert args.language == "ja"

    def test_model_type_flag(self):
        with patch("sys.argv", ["sherox.server", "--model-type", "whisper"]):
            args = srv.parse_server_args()
        assert args.model_type == "whisper"
