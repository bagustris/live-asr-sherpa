"""Tests for benchmark/benchmark.py helper functions."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# benchmark.py imports sherpa_onnx at runtime inside main(); the module-level
# imports only need soundfile and numpy.  We can test all helper functions
# independently without a live model.
import benchmark as bm
from metrics import AggregateMetrics, UtteranceResult


# ---------------------------------------------------------------------------
# manifest_from_librispeech
# ---------------------------------------------------------------------------

def test_manifest_from_librispeech_missing_dir(tmp_path):
    with pytest.raises(SystemExit):
        bm.manifest_from_librispeech(str(tmp_path / "nonexistent"))


def test_manifest_from_librispeech_empty_dir(tmp_path):
    with pytest.raises(SystemExit):
        bm.manifest_from_librispeech(str(tmp_path))


def test_manifest_from_librispeech_basic(tmp_path):
    """Verify records are built correctly from a minimal LibriSpeech structure."""
    chapter_dir = tmp_path / "1272" / "128104"
    chapter_dir.mkdir(parents=True)

    # Create a fake audio file
    (chapter_dir / "1272-128104-0000.flac").touch()

    # Create transcript
    trans = chapter_dir / "1272-128104.trans.txt"
    trans.write_text("1272-128104-0000 MISTER QUILTER\n")

    records = bm.manifest_from_librispeech(str(tmp_path))
    assert len(records) == 1
    path, text = records[0]
    assert path.endswith("1272-128104-0000.flac")
    assert text == "mister quilter"


def test_manifest_from_librispeech_skips_missing_audio(tmp_path):
    chapter_dir = tmp_path / "1272" / "128104"
    chapter_dir.mkdir(parents=True)
    # No audio file — just the transcript
    trans = chapter_dir / "1272-128104.trans.txt"
    trans.write_text("1272-128104-0000 SOME TEXT\n")

    with pytest.raises(SystemExit):
        bm.manifest_from_librispeech(str(tmp_path))


# ---------------------------------------------------------------------------
# load_manifest (TSV/CSV)
# ---------------------------------------------------------------------------

def test_load_manifest_tsv(tmp_path):
    tsv = tmp_path / "manifest.tsv"
    tsv.write_text("/abs/path/audio.wav\thello world\n")
    records = bm.load_manifest(str(tsv))
    assert records == [("/abs/path/audio.wav", "hello world")]


def test_load_manifest_csv(tmp_path):
    csv_file = tmp_path / "manifest.csv"
    csv_file.write_text("/abs/path/audio.wav,hello world\n")
    records = bm.load_manifest(str(csv_file))
    assert records == [("/abs/path/audio.wav", "hello world")]


def test_load_manifest_relative_path_resolved(tmp_path):
    tsv = tmp_path / "manifest.tsv"
    tsv.write_text("audio.wav\thello\n")
    records = bm.load_manifest(str(tsv))
    assert records[0][0] == str(tmp_path / "audio.wav")


def test_load_manifest_skips_comment_lines(tmp_path):
    tsv = tmp_path / "manifest.tsv"
    tsv.write_text("# comment\n/path/a.wav\thello\n")
    records = bm.load_manifest(str(tsv))
    assert len(records) == 1


def test_load_manifest_empty_file(tmp_path):
    tsv = tmp_path / "manifest.tsv"
    tsv.write_text("")
    with pytest.raises(SystemExit):
        bm.load_manifest(str(tsv))


# ---------------------------------------------------------------------------
# load_audio
# ---------------------------------------------------------------------------

def test_load_audio_mono(tmp_path):
    """load_audio returns (float32 ndarray, duration) for a mono file."""
    import soundfile as sf
    path = str(tmp_path / "test.wav")
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(path, audio, 16000)

    result, duration = bm.load_audio(path, target_sr=16000)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert duration == pytest.approx(1.0)


def test_load_audio_stereo_converted_to_mono(tmp_path):
    import soundfile as sf
    path = str(tmp_path / "stereo.wav")
    audio = np.zeros((8000, 2), dtype=np.float32)
    sf.write(path, audio, 16000)

    result, duration = bm.load_audio(path, target_sr=16000)
    assert result.ndim == 1


# ---------------------------------------------------------------------------
# transcribe_offline
# ---------------------------------------------------------------------------

def _make_offline_recognizer(text="hello world"):
    """Build a mock OfflineRecognizer that returns *text* for any input."""
    mock_result = MagicMock()
    mock_result.text = text

    mock_stream = MagicMock()
    mock_stream.result = mock_result

    recognizer = MagicMock()
    recognizer.create_stream.return_value = mock_stream

    return recognizer


def test_transcribe_offline_returns_text():
    rec = _make_offline_recognizer("hello world")
    audio = np.zeros(16000, dtype=np.float32)
    result = bm.transcribe_offline(rec, audio, 16000)
    assert result == "hello world"


def test_transcribe_offline_strips_whitespace():
    rec = _make_offline_recognizer("  hello world  ")
    audio = np.zeros(16000, dtype=np.float32)
    result = bm.transcribe_offline(rec, audio, 16000)
    assert result == "hello world"


def test_transcribe_offline_calls_decode_stream():
    rec = _make_offline_recognizer()
    audio = np.zeros(16000, dtype=np.float32)
    bm.transcribe_offline(rec, audio, 16000)
    rec.decode_stream.assert_called_once()


# ---------------------------------------------------------------------------
# transcribe_online
# ---------------------------------------------------------------------------

def _make_online_recognizer(text="online text"):
    """Build a mock OnlineRecognizer.

    get_result() returns a str (matching real sherpa-onnx OnlineRecognizer behaviour).
    """
    mock_stream = MagicMock()

    recognizer = MagicMock()
    recognizer.create_stream.return_value = mock_stream
    recognizer.is_ready.return_value = False
    recognizer.is_endpoint.return_value = False
    recognizer.get_result.return_value = text  # str, not object

    return recognizer


def test_transcribe_online_returns_text():
    rec = _make_online_recognizer("streaming result")
    audio = np.zeros(16000, dtype=np.float32)
    result = bm.transcribe_online(rec, audio, 16000, chunk_size=0.1)
    # No endpoint triggered so text comes from final flush
    assert result == "streaming result"


def test_transcribe_online_collects_endpoint_text():
    """When an endpoint fires mid-stream, that text is collected."""
    texts = ["first segment", ""]
    call_count = [0]

    def _get_result(_stream):
        idx = min(call_count[0], len(texts) - 1)
        call_count[0] += 1
        return texts[idx]  # str, not object

    endpoint_calls = [0]

    def _is_endpoint(_stream):
        endpoint_calls[0] += 1
        return endpoint_calls[0] == 1  # endpoint fires once

    rec = MagicMock()
    rec.create_stream.return_value = MagicMock()
    rec.is_ready.return_value = False
    rec.is_endpoint.side_effect = _is_endpoint
    rec.get_result.side_effect = _get_result

    audio = np.zeros(3200, dtype=np.float32)  # 0.2s at 16 kHz
    result = bm.transcribe_online(rec, audio, 16000, chunk_size=0.1)
    assert "first segment" in result


def test_load_audio_raises_when_resample_needed_and_resampy_missing(tmp_path):
    """load_audio raises RuntimeError when sr != target_sr and resampy is absent."""
    import soundfile as sf

    path = str(tmp_path / "audio48k.wav")
    sf.write(path, np.zeros(48000, dtype=np.float32), 48000)

    with patch.dict("sys.modules", {"resampy": None}):
        with pytest.raises(RuntimeError, match="resampy not installed"):
            bm.load_audio(path, target_sr=16000)


def test_load_audio_no_error_when_sample_rates_match(tmp_path):
    """load_audio does not attempt resampling when sr == target_sr."""
    import soundfile as sf

    path = str(tmp_path / "audio16k.wav")
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)

    # Should succeed even if resampy is unavailable
    with patch.dict("sys.modules", {"resampy": None}):
        audio, duration = bm.load_audio(path, target_sr=16000)
    assert duration == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

def test_run_benchmark_offline_collects_results(tmp_path):
    """run_benchmark returns correct number of UtteranceResult objects."""
    import soundfile as sf

    # Write two fake audio files
    records = []
    for i in range(2):
        path = str(tmp_path / f"utt{i}.wav")
        sf.write(path, np.zeros(8000, dtype=np.float32), 16000)
        records.append((path, f"reference text {i}"))

    rec = _make_offline_recognizer("hypothesis text")
    results, agg = bm.run_benchmark(
        rec, records, offline=True, sample_rate=16000, verbose=False
    )
    assert len(results) == 2
    assert agg.n_utterances == 2
    assert isinstance(agg, AggregateMetrics)


def test_run_benchmark_skips_bad_audio_file(tmp_path):
    """run_benchmark skips files that cannot be loaded."""
    records = [(str(tmp_path / "nonexistent.wav"), "ref")]
    rec = _make_offline_recognizer()
    results, agg = bm.run_benchmark(rec, records, offline=True)
    assert len(results) == 0
    assert agg.n_utterances == 0


def test_run_benchmark_handles_transcription_error(tmp_path):
    """run_benchmark records an empty hypothesis when transcription raises."""
    import soundfile as sf

    path = str(tmp_path / "utt.wav")
    sf.write(path, np.zeros(8000, dtype=np.float32), 16000)
    records = [(path, "reference")]

    rec = MagicMock()
    rec.create_stream.side_effect = RuntimeError("model broken")

    results, agg = bm.run_benchmark(rec, records, offline=True)
    assert len(results) == 1
    assert results[0].hypothesis == ""


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

def test_build_parser_defaults():
    parser = bm.build_parser()
    args = parser.parse_args([])
    assert args.offline is False
    assert args.threads == 4
    assert args.sample_rate == 16000
    assert args.language == "en"
    assert args.max_utts is None
    assert args.verbose is False
    assert args.output is None
    assert args.model_dir is None


def test_build_parser_offline_flag():
    parser = bm.build_parser()
    args = parser.parse_args(["--offline"])
    assert args.offline is True


def test_build_parser_data_dir_and_manifest_mutually_exclusive():
    parser = bm.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--data-dir", "/a", "--manifest", "/b"])


def test_build_parser_max_utts():
    parser = bm.build_parser()
    args = parser.parse_args(["--max-utts", "42"])
    assert args.max_utts == 42


def test_build_parser_model_type():
    parser = bm.build_parser()
    args = parser.parse_args(["--model-type", "whisper"])
    assert args.model_type == "whisper"


def test_build_parser_threads():
    parser = bm.build_parser()
    args = parser.parse_args(["--threads", "8"])
    assert args.threads == 8
