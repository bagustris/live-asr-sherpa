import io
import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from sherox.streaming import (
    _PREFIX,
    _clear_line,
    _flush_tail,
    run_offline_vad_streaming,
    run_streaming,
)


# ---------------------------------------------------------------------------
# _clear_line
# ---------------------------------------------------------------------------

class TestClearLine:
    def test_does_nothing_for_empty_partial(self, capsys):
        _clear_line("")
        assert capsys.readouterr().out == ""

    def test_writes_carriage_return_for_nonempty_partial(self, capsys):
        _clear_line("hello")
        assert "\r" in capsys.readouterr().out

    def test_clears_exactly_prefix_plus_partial_width(self):
        partial = "test text"
        expected_spaces = " " * (len(_PREFIX) + len(partial))
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _clear_line(partial)
        assert expected_spaces in buf.getvalue()

    def test_clears_single_character_partial(self, capsys):
        _clear_line("x")
        assert "\r" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _flush_tail
# ---------------------------------------------------------------------------

class TestFlushTail:
    def _make_recognizer(self, text: str) -> tuple:
        recognizer = MagicMock()
        stream = MagicMock()
        recognizer.create_stream.return_value = stream
        recognizer.is_ready.return_value = False
        recognizer.get_result.return_value = MagicMock(
            strip=MagicMock(return_value=text)
        )
        return recognizer, stream

    def test_prints_remaining_text(self, capsys):
        recognizer, stream = self._make_recognizer("hello world")
        _flush_tail(recognizer, stream, 16000, "")
        assert "hello world" in capsys.readouterr().out

    def test_no_content_printed_when_empty(self, capsys):
        recognizer, stream = self._make_recognizer("")
        _flush_tail(recognizer, stream, 16000, "")
        # Only the trailing newline should be written
        out = capsys.readouterr().out
        assert out == "\n"

    def test_appends_half_second_of_silence(self):
        recognizer, stream = self._make_recognizer("")
        _flush_tail(recognizer, stream, 16000, "")
        tail = stream.accept_waveform.call_args[0][1]
        assert len(tail) == 8000  # int(16000 * 0.5)
        assert np.all(tail == 0.0)

    def test_tail_silence_uses_correct_sample_rate(self):
        recognizer, stream = self._make_recognizer("")
        _flush_tail(recognizer, stream, 48000, "")
        tail = stream.accept_waveform.call_args[0][1]
        assert len(tail) == 24000  # int(48000 * 0.5)

    def test_decodes_remaining_frames(self):
        recognizer, stream = self._make_recognizer("")
        # Simulate two decode iterations needed
        recognizer.is_ready.side_effect = [True, True, False]
        _flush_tail(recognizer, stream, 16000, "")
        assert recognizer.decode_stream.call_count == 2

    def test_always_writes_newline(self, capsys):
        recognizer, stream = self._make_recognizer("some text")
        _flush_tail(recognizer, stream, 16000, "")
        assert capsys.readouterr().out.endswith("\n")


# ---------------------------------------------------------------------------
# run_streaming
# ---------------------------------------------------------------------------

class TestRunStreaming:
    def _make_recognizer(self):
        rec = MagicMock()
        stream = MagicMock()
        rec.create_stream.return_value = stream
        rec.is_ready.return_value = False
        return rec, stream

    def test_prints_finalized_segment_on_endpoint(self, capsys):
        rec, stream = self._make_recognizer()
        rec.get_result.return_value = MagicMock(
            strip=MagicMock(return_value="hello world")
        )
        rec.is_endpoint.return_value = True

        with patch("sherox.streaming._flush_tail"):
            run_streaming(rec, iter([np.zeros(2560, dtype="float32")]))

        assert "hello world" in capsys.readouterr().out

    def test_empty_endpoint_does_not_print(self, capsys):
        rec, stream = self._make_recognizer()
        rec.get_result.return_value = MagicMock(strip=MagicMock(return_value=""))
        rec.is_endpoint.return_value = True

        with patch("sherox.streaming._flush_tail"):
            run_streaming(rec, iter([np.zeros(2560, dtype="float32")]))

        assert capsys.readouterr().out == ""

    def test_partial_written_in_place(self, capsys):
        rec, stream = self._make_recognizer()
        results = ["partial", "partial longer"]
        rec.get_result.side_effect = [
            MagicMock(strip=MagicMock(return_value=r)) for r in results
        ]
        rec.is_endpoint.return_value = False

        with patch("sherox.streaming._flush_tail"):
            run_streaming(rec, iter([np.zeros(2560, dtype="float32"),
                                     np.zeros(2560, dtype="float32")]))

        out = capsys.readouterr().out
        assert "\r" in out
        assert "partial" in out

    def test_reset_called_on_endpoint(self):
        rec, stream = self._make_recognizer()
        rec.get_result.return_value = MagicMock(strip=MagicMock(return_value="text"))
        rec.is_endpoint.return_value = True

        with patch("sherox.streaming._flush_tail"):
            run_streaming(rec, iter([np.zeros(2560, dtype="float32")]))

        rec.reset.assert_called_once_with(stream)

    def test_accepts_waveform_for_each_chunk(self):
        rec, stream = self._make_recognizer()
        rec.get_result.return_value = MagicMock(strip=MagicMock(return_value=""))
        rec.is_endpoint.return_value = False
        chunks = [np.zeros(2560, dtype="float32")] * 3

        with patch("sherox.streaming._flush_tail"):
            run_streaming(rec, iter(chunks), sample_rate=16000)

        assert stream.accept_waveform.call_count == 3

    def test_handles_keyboard_interrupt_gracefully(self):
        rec, stream = self._make_recognizer()
        rec.get_result.return_value = MagicMock(strip=MagicMock(return_value=""))
        rec.is_endpoint.return_value = False

        def interrupt_gen():
            yield np.zeros(2560, dtype="float32")
            raise KeyboardInterrupt

        with patch("sherox.streaming._flush_tail"):
            run_streaming(rec, interrupt_gen())  # must not propagate

    def test_flush_tail_always_called(self):
        rec, stream = self._make_recognizer()
        rec.get_result.return_value = MagicMock(strip=MagicMock(return_value=""))
        rec.is_endpoint.return_value = False

        with patch("sherox.streaming._flush_tail") as mock_flush:
            run_streaming(rec, iter([np.zeros(2560, dtype="float32")]))

        mock_flush.assert_called_once()

    def test_mic_level_bar_shown_when_enabled(self, capsys):
        rec, stream = self._make_recognizer()
        rec.get_result.return_value = MagicMock(strip=MagicMock(return_value=""))
        rec.is_endpoint.return_value = False

        with patch("sherox.streaming._flush_tail"):
            run_streaming(
                rec,
                iter([np.ones(2560, dtype="float32") * 0.1]),
                show_mic_level=True,
            )

        assert "mic:" in capsys.readouterr().out

    def test_mic_level_bar_hidden_when_disabled(self, capsys):
        rec, stream = self._make_recognizer()
        rec.get_result.return_value = MagicMock(strip=MagicMock(return_value=""))
        rec.is_endpoint.return_value = False

        with patch("sherox.streaming._flush_tail"):
            run_streaming(
                rec,
                iter([np.ones(2560, dtype="float32") * 0.1]),
                show_mic_level=False,
            )

        assert "mic:" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run_offline_vad_streaming
# ---------------------------------------------------------------------------

class TestRunOfflineVadStreaming:
    def _make_recognizer_with_text(self, text: str):
        rec = MagicMock()
        stream = MagicMock()
        result = MagicMock()
        result.text = text
        stream.result = result
        rec.create_stream.return_value = stream
        return rec

    def test_processes_completed_speech_segment(self, capsys):
        rec = self._make_recognizer_with_text("  hello  ")
        vad = MagicMock()
        segment = MagicMock()
        segment.samples = np.ones(8000, dtype="float32").tolist()
        vad.front = segment
        # not empty once (has segment), empty after pop, empty in finally
        vad.empty.side_effect = [False, True, True]

        run_offline_vad_streaming(rec, vad, iter([np.zeros(1600, dtype="float32")]))

        assert "hello" in capsys.readouterr().out

    def test_vad_accept_waveform_called_per_chunk(self):
        rec = self._make_recognizer_with_text("")
        vad = MagicMock()
        vad.empty.return_value = True
        chunks = [np.zeros(1600, dtype="float32")] * 4

        run_offline_vad_streaming(rec, vad, iter(chunks))

        assert vad.accept_waveform.call_count == 4

    def test_vad_pop_called_after_segment_processed(self):
        rec = self._make_recognizer_with_text("")
        vad = MagicMock()
        segment = MagicMock()
        segment.samples = []
        vad.front = segment
        vad.empty.side_effect = [False, True, True]

        run_offline_vad_streaming(rec, vad, iter([np.zeros(1600, dtype="float32")]))

        vad.pop.assert_called()

    def test_flush_called_in_finally(self):
        rec = self._make_recognizer_with_text("")
        vad = MagicMock()
        vad.empty.return_value = True

        run_offline_vad_streaming(rec, vad, iter([]))

        vad.flush.assert_called_once()

    def test_flushes_buffered_segments_in_finally(self):
        rec = self._make_recognizer_with_text("buffered speech")
        vad = MagicMock()
        segment = MagicMock()
        segment.samples = np.ones(8000, dtype="float32").tolist()
        vad.front = segment
        # During iteration: empty (skip). In finally: not empty once, then empty.
        vad.empty.side_effect = [False, True]

        run_offline_vad_streaming(rec, vad, iter([]))

        vad.flush.assert_called_once()
        # pop must have been called for the buffered segment
        vad.pop.assert_called()

    def test_handles_keyboard_interrupt_gracefully(self):
        rec = self._make_recognizer_with_text("")
        vad = MagicMock()
        vad.empty.return_value = True

        def interrupt_gen():
            yield np.zeros(1600, dtype="float32")
            raise KeyboardInterrupt

        run_offline_vad_streaming(rec, vad, interrupt_gen())  # must not propagate

    def test_mic_level_bar_shown_when_enabled(self, capsys):
        rec = self._make_recognizer_with_text("")
        vad = MagicMock()
        vad.empty.return_value = True

        run_offline_vad_streaming(
            rec, vad,
            iter([np.ones(1600, dtype="float32") * 0.1]),
            show_mic_level=True,
        )

        assert "mic:" in capsys.readouterr().out

    def test_empty_segment_text_not_printed(self, capsys):
        rec = self._make_recognizer_with_text("   ")  # whitespace only
        vad = MagicMock()
        segment = MagicMock()
        segment.samples = np.ones(8000, dtype="float32").tolist()
        vad.front = segment
        vad.empty.side_effect = [False, True, True]

        run_offline_vad_streaming(rec, vad, iter([np.zeros(1600, dtype="float32")]))

        # "   ".strip() == "" so nothing content-ful is printed
        out = capsys.readouterr().out.strip()
        assert out == ""
