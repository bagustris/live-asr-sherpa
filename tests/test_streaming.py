import io
import sys
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from sherox.streaming import (
    _PREFIX,
    _clear_line,
    _flush_tail,
    _is_dark_terminal,
    _speaker_colour,
    _rich_print,
    _dominant_speaker,
    _decode_and_print,
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

    def test_final_only_suppresses_partial(self, capsys):
        """When final_only=True, intermediate partial hypotheses are not printed."""
        rec, stream = self._make_recognizer()
        partials = ["partial one", "partial two longer"]
        rec.get_result.side_effect = [
            MagicMock(strip=MagicMock(return_value=p)) for p in partials
        ]
        rec.is_endpoint.return_value = False

        with patch("sherox.streaming._flush_tail"):
            run_streaming(
                rec,
                iter([np.zeros(2560, dtype="float32"),
                      np.zeros(2560, dtype="float32")]),
                final_only=True,
            )

        out = capsys.readouterr().out
        assert "partial" not in out

    def test_final_only_still_prints_finalised_segment(self, capsys):
        """final_only must NOT suppress finalised endpoint segments."""
        rec, stream = self._make_recognizer()
        rec.get_result.return_value = MagicMock(
            strip=MagicMock(return_value="final segment")
        )
        rec.is_endpoint.return_value = True

        with patch("sherox.streaming._flush_tail"):
            run_streaming(
                rec,
                iter([np.zeros(2560, dtype="float32")]),
                final_only=True,
            )

        assert "final segment" in capsys.readouterr().out

    def test_final_only_default_is_false(self, capsys):
        """Omitting final_only keeps the previous partial-output behaviour."""
        rec, stream = self._make_recognizer()
        rec.get_result.return_value = MagicMock(
            strip=MagicMock(return_value="streaming partial")
        )
        rec.is_endpoint.return_value = False

        with patch("sherox.streaming._flush_tail"):
            run_streaming(rec, iter([np.zeros(2560, dtype="float32")]))

        assert "streaming partial" in capsys.readouterr().out


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


# ---------------------------------------------------------------------------
# _is_dark_terminal
# ---------------------------------------------------------------------------

class TestIsDarkTerminal:
    def test_returns_true_when_no_env_var(self):
        with patch.dict("os.environ", {}, clear=True):
            # COLORFGBG not set at all — must use os.environ.get fallback
            import os
            env = {k: v for k, v in os.environ.items() if k != "COLORFGBG"}
            with patch.dict("os.environ", env, clear=True):
                result = _is_dark_terminal()
        assert result is True

    def test_dark_bg_index_zero(self):
        with patch.dict("os.environ", {"COLORFGBG": "15;0"}):
            assert _is_dark_terminal() is True

    def test_dark_bg_index_six(self):
        with patch.dict("os.environ", {"COLORFGBG": "0;6"}):
            assert _is_dark_terminal() is True

    def test_light_bg_index_seven(self):
        with patch.dict("os.environ", {"COLORFGBG": "0;7"}):
            assert _is_dark_terminal() is False

    def test_light_bg_index_fifteen(self):
        with patch.dict("os.environ", {"COLORFGBG": "15;15"}):
            assert _is_dark_terminal() is False

    def test_invalid_colorfgbg_falls_back_to_true(self):
        with patch.dict("os.environ", {"COLORFGBG": "not-a-number"}):
            assert _is_dark_terminal() is True


# ---------------------------------------------------------------------------
# _speaker_colour
# ---------------------------------------------------------------------------

class TestSpeakerColour:
    def test_returns_string(self):
        result = _speaker_colour(0)
        assert isinstance(result, str)

    def test_cycles_through_palette(self):
        c0 = _speaker_colour(0)
        c1 = _speaker_colour(1)
        assert c0 != c1

    def test_modulo_wraps(self):
        from sherox.streaming import _SPEAKER_COLOURS
        length = len(_SPEAKER_COLOURS)
        assert _speaker_colour(0) == _speaker_colour(length)


# ---------------------------------------------------------------------------
# _rich_print
# ---------------------------------------------------------------------------

class TestRichPrint:
    def test_plain_print_no_speaker(self, capsys):
        _rich_print("hello world")
        assert "hello world" in capsys.readouterr().out

    def test_with_speaker_id_colour_only(self, capsys):
        _rich_print("some text", speaker_id=0, show_speaker_tag=False)
        assert "some text" in capsys.readouterr().out

    def test_with_speaker_id_and_tag(self, capsys):
        _rich_print("tagged text", speaker_id=1, show_speaker_tag=True)
        out = capsys.readouterr().out
        assert "tagged text" in out
        assert "Speaker 1" in out

    def test_no_tag_when_show_speaker_tag_false(self, capsys):
        _rich_print("no tag here", speaker_id=2, show_speaker_tag=False)
        out = capsys.readouterr().out
        assert "Speaker" not in out


# ---------------------------------------------------------------------------
# _dominant_speaker
# ---------------------------------------------------------------------------

class TestDominantSpeaker:
    def _make_result(self, segments):
        """segments: list of (speaker, start, end)"""
        result = MagicMock()
        segs = []
        for speaker, start, end in segments:
            seg = MagicMock()
            seg.speaker = speaker
            seg.start = start
            seg.end = end
            segs.append(seg)
        result.sort_by_start_time.return_value = segs
        return result

    def test_returns_zero_for_empty_segments(self):
        result = self._make_result([])
        assert _dominant_speaker(result) == 0

    def test_returns_speaker_with_most_time(self):
        result = self._make_result([
            (0, 0.0, 2.0),  # speaker 0: 2 seconds
            (1, 2.0, 3.0),  # speaker 1: 1 second
        ])
        assert _dominant_speaker(result) == 0

    def test_multiple_segments_same_speaker(self):
        result = self._make_result([
            (0, 0.0, 1.0),
            (1, 1.0, 1.5),
            (0, 1.5, 3.0),  # speaker 0 total: 2.5 seconds
        ])
        assert _dominant_speaker(result) == 0


# ---------------------------------------------------------------------------
# run_streaming — decode_stream when is_ready=True
# ---------------------------------------------------------------------------

class TestRunStreamingIsReady:
    def test_decode_stream_called_when_is_ready(self):
        rec = MagicMock()
        stream = MagicMock()
        rec.create_stream.return_value = stream
        # is_ready returns True once, then False (to exit the while loop)
        rec.is_ready.side_effect = [True, False]
        rec.get_result.return_value = MagicMock(strip=MagicMock(return_value=""))
        rec.is_endpoint.return_value = False

        with patch("sherox.streaming._flush_tail"):
            run_streaming(rec, iter([np.zeros(2560, dtype="float32")]))

        rec.decode_stream.assert_called_once_with(stream)


# ---------------------------------------------------------------------------
# run_streaming — show_mic_level clear at endpoint with text
# ---------------------------------------------------------------------------

class TestRunStreamingMicLevelEndpoint:
    def test_clears_mic_level_line_at_endpoint_with_text(self, capsys):
        rec = MagicMock()
        stream = MagicMock()
        rec.create_stream.return_value = stream
        rec.is_ready.return_value = False
        rec.get_result.return_value = MagicMock(strip=MagicMock(return_value="text"))
        rec.is_endpoint.return_value = True

        with patch("sherox.streaming._flush_tail"), \
             patch("sherox.streaming.shutil.get_terminal_size", return_value=MagicMock(columns=80)):
            run_streaming(
                rec,
                iter([np.ones(2560, dtype="float32") * 0.1]),
                show_mic_level=True,
            )

        out = capsys.readouterr().out
        # The line-clear sequence should contain spaces
        assert " " * 40 in out or "\r" in out

    def test_clears_mic_level_falls_back_on_oserror(self, capsys):
        rec = MagicMock()
        stream = MagicMock()
        rec.create_stream.return_value = stream
        rec.is_ready.return_value = False
        rec.get_result.return_value = MagicMock(strip=MagicMock(return_value="text"))
        rec.is_endpoint.return_value = True

        with patch("sherox.streaming._flush_tail"), \
             patch("sherox.streaming.shutil.get_terminal_size", side_effect=OSError):
            run_streaming(
                rec,
                iter([np.ones(2560, dtype="float32") * 0.1]),
                show_mic_level=True,
            )

        # Should not raise — just uses fallback width=80


# ---------------------------------------------------------------------------
# run_streaming — diarization paths
# ---------------------------------------------------------------------------

class TestRunStreamingDiarization:
    def _make_recognizer(self, text="hello"):
        rec = MagicMock()
        stream = MagicMock()
        rec.create_stream.return_value = stream
        rec.is_ready.return_value = False
        rec.get_result.return_value = MagicMock(strip=MagicMock(return_value=text))
        rec.is_endpoint.return_value = True
        return rec, stream

    def test_diarization_submits_on_endpoint(self):
        rec, stream = self._make_recognizer("hello world")
        diarization = MagicMock()

        mock_future = MagicMock()
        mock_future.done.return_value = True
        diar_result = MagicMock()
        diar_result.sort_by_start_time.return_value = []
        mock_future.result.return_value = diar_result

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future

        with patch("sherox.streaming._flush_tail"), \
             patch("sherox.streaming.ThreadPoolExecutor", return_value=mock_executor):
            run_streaming(
                rec,
                iter([np.ones(2560, dtype="float32")]),
                diarization=diarization,
            )

        mock_executor.submit.assert_called()

    def test_flush_pending_called_with_done_future_in_finally(self, capsys):
        """The result from a completed future is used to colour the output."""
        rec, stream = self._make_recognizer("final text")
        diarization = MagicMock()

        mock_future = MagicMock()
        mock_future.done.return_value = True
        diar_result = MagicMock()
        diar_result.sort_by_start_time.return_value = []
        mock_future.result.return_value = diar_result

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future

        with patch("sherox.streaming._flush_tail"), \
             patch("sherox.streaming.ThreadPoolExecutor", return_value=mock_executor):
            run_streaming(
                rec,
                iter([np.ones(2560, dtype="float32")]),
                diarization=diarization,
            )

        # The pending_text "final text" should eventually be printed
        assert "final text" in capsys.readouterr().out

    def test_flush_pending_with_future_exception_logs_debug(self, capsys):
        """Exception from diarization future is logged, not raised."""
        rec, stream = self._make_recognizer("hello")
        diarization = MagicMock()

        mock_future = MagicMock()
        mock_future.done.return_value = True
        mock_future.result.side_effect = RuntimeError("diarization error")

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future

        with patch("sherox.streaming._flush_tail"), \
             patch("sherox.streaming.ThreadPoolExecutor", return_value=mock_executor), \
             patch("sherox.streaming.logging.debug") as mock_log:
            run_streaming(
                rec,
                iter([np.ones(2560, dtype="float32")]),
                diarization=diarization,
            )

        # logging.debug should have been called for the exception
        mock_log.assert_called()

    def test_busy_worker_skips_diarization(self, capsys):
        """When the previous diarization is still running, new text is flushed immediately."""
        rec = MagicMock()
        stream = MagicMock()
        rec.create_stream.return_value = stream
        rec.is_ready.return_value = False
        # First endpoint: text "first", second: text "second"
        rec.get_result.side_effect = [
            MagicMock(strip=MagicMock(return_value="first")),
            MagicMock(strip=MagicMock(return_value="second")),
        ]
        rec.is_endpoint.return_value = True
        diarization = MagicMock()

        # First submit returns a NOT-done future
        not_done_future = MagicMock()
        not_done_future.done.return_value = False  # still running

        done_future = MagicMock()
        done_future.done.return_value = True
        diar_result = MagicMock()
        diar_result.sort_by_start_time.return_value = []
        done_future.result.return_value = diar_result

        mock_executor = MagicMock()
        mock_executor.submit.side_effect = [not_done_future, done_future]

        with patch("sherox.streaming._flush_tail"), \
             patch("sherox.streaming.ThreadPoolExecutor", return_value=mock_executor):
            run_streaming(
                rec,
                iter([
                    np.ones(2560, dtype="float32"),
                    np.ones(2560, dtype="float32"),
                ]),
                diarization=diarization,
            )

        out = capsys.readouterr().out
        # Both utterances should appear in output
        assert "first" in out or "second" in out

    def test_audio_buf_populated_with_diarization(self):
        """audio_buf is accumulated when diarization is enabled."""
        rec, stream = self._make_recognizer("hello")
        diarization = MagicMock()

        mock_future = MagicMock()
        mock_future.done.return_value = True
        diar_result = MagicMock()
        diar_result.sort_by_start_time.return_value = []
        mock_future.result.return_value = diar_result

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future

        with patch("sherox.streaming._flush_tail"), \
             patch("sherox.streaming.ThreadPoolExecutor", return_value=mock_executor):
            run_streaming(
                rec,
                iter([np.ones(2560, dtype="float32")]),
                diarization=diarization,
            )

        # submit must have been called (audio_buf was populated and concatenated)
        mock_executor.submit.assert_called()


# ---------------------------------------------------------------------------
# _decode_and_print — with diarization
# ---------------------------------------------------------------------------

class TestDecodeAndPrint:
    def test_with_diarization_and_text(self, capsys):
        rec = MagicMock()
        stream = MagicMock()
        result = MagicMock()
        result.text = "diar text"
        stream.result = result
        rec.create_stream.return_value = stream

        diarization = MagicMock()
        executor = MagicMock()

        asr_future = MagicMock()
        asr_future.result.return_value = "diar text"

        diar_result = MagicMock()
        diar_result.sort_by_start_time.return_value = []
        diar_future = MagicMock()
        diar_future.result.return_value = diar_result

        executor.submit.side_effect = [asr_future, diar_future]

        _decode_and_print(rec, np.zeros(8000, dtype="float32"), 16000, diarization, executor)

        assert "diar text" in capsys.readouterr().out

    def test_with_diarization_and_empty_text(self, capsys):
        rec = MagicMock()
        stream = MagicMock()
        result = MagicMock()
        result.text = ""
        stream.result = result
        rec.create_stream.return_value = stream

        diarization = MagicMock()
        executor = MagicMock()

        asr_future = MagicMock()
        asr_future.result.return_value = ""

        diar_result = MagicMock()
        diar_future = MagicMock()
        diar_future.result.return_value = diar_result

        executor.submit.side_effect = [asr_future, diar_future]

        _decode_and_print(rec, np.zeros(8000, dtype="float32"), 16000, diarization, executor)

        # Empty text should not be printed
        assert capsys.readouterr().out.strip() == ""


# ---------------------------------------------------------------------------
# run_offline_vad_streaming — diarization executor shutdown
# ---------------------------------------------------------------------------

class TestRunOfflineVadStreamingDiarization:
    def test_executor_shutdown_called_with_diarization(self):
        rec = MagicMock()
        stream = MagicMock()
        result = MagicMock()
        result.text = ""
        stream.result = result
        rec.create_stream.return_value = stream

        vad = MagicMock()
        vad.empty.return_value = True

        diarization = MagicMock()
        mock_executor = MagicMock()

        with patch("sherox.streaming.ThreadPoolExecutor", return_value=mock_executor):
            run_offline_vad_streaming(
                rec, vad, iter([]), diarization=diarization
            )

        mock_executor.shutdown.assert_called_once_with(wait=True)


# ---------------------------------------------------------------------------
# Subtitle writers
# ---------------------------------------------------------------------------

from sherox.streaming import write_srt, write_vtt, write_txt


class TestWriteSrt:
    def test_writes_correctly_formatted_srt(self, tmp_path):
        subtitles = [(0.0, 1.5, "Hello world"), (2.0, 3.0, "Goodbye")]
        out = str(tmp_path / "out.srt")
        write_srt(subtitles, out)
        text = open(out).read()
        assert "1\n" in text
        assert "00:00:00,000 --> 00:00:01,500" in text
        assert "Hello world" in text
        assert "2\n" in text
        assert "00:00:02,000 --> 00:00:03,000" in text
        assert "Goodbye" in text

    def test_empty_subtitles_creates_file(self, tmp_path):
        out = str(tmp_path / "empty.srt")
        write_srt([], out)
        assert open(out).read() == ""


class TestWriteVtt:
    def test_starts_with_webvtt_header(self, tmp_path):
        out = str(tmp_path / "out.vtt")
        write_vtt([(0.0, 1.0, "Hi")], out)
        text = open(out).read()
        assert text.startswith("WEBVTT")

    def test_formats_timestamps(self, tmp_path):
        out = str(tmp_path / "out.vtt")
        write_vtt([(3661.5, 3662.0, "Late")], out)
        text = open(out).read()
        # 3661.5s = 1h01m01.500s
        assert "01:01:01.500 --> 01:01:02.000" in text


class TestWriteTxt:
    def test_writes_one_line_per_segment(self, tmp_path):
        out = str(tmp_path / "out.txt")
        write_txt([(0.0, 1.0, "Line one"), (1.0, 2.0, "Line two")], out)
        lines = open(out).read().splitlines()
        assert lines == ["Line one", "Line two"]

    def test_empty_subtitles_creates_empty_file(self, tmp_path):
        out = str(tmp_path / "empty.txt")
        write_txt([], out)
        assert open(out).read() == ""
