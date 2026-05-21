from __future__ import annotations

import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Deque, Generator, NamedTuple, Optional, Tuple

import logging
import numpy as np
from rich.console import Console
from rich.text import Text
import shutil

# ── Rich console (no markup auto-escaping needed; we build Text objects) ────
_console = Console(highlight=False, markup=False)


def _is_dark_terminal() -> bool:
    """Detect whether the terminal has a dark background.

    Primary signal: the COLORFGBG environment variable (set by many
    xterm-compatible terminals). Its format is ``foreground;background`` where
    the background field is a standard ANSI colour index:

    +---------+-------------------------------------------+-----------+
    | Index   | Colour                                    | BG type   |
    +=========+===========================================+===========+
    | 0–6     | black, red, green, yellow, blue,          | dark      |
    |         | magenta, cyan                             |           |
    +---------+-------------------------------------------+-----------+
    | 7       | white / light-grey                        | light     |
    +---------+-------------------------------------------+-----------+
    | 8–15    | bright variants (bright-black … white)    | light     |
    +---------+-------------------------------------------+-----------+

    Falls back to ``True`` (dark) when the signal is absent — dark backgrounds
    are the common default in developer environments.
    """
    colorfgbg = os.environ.get("COLORFGBG", "")
    if colorfgbg:
        try:
            bg = int(colorfgbg.split(";")[-1])
            # Indices 0-6: standard dark colours (black … cyan) → dark bg.
            # Index 7 (white) and 8-15 (bright variants) → treat as light.
            return bg <= 6
        except (ValueError, IndexError):
            pass
    return True  # Safe default: most developer terminals use a dark background.


# ── Speaker colour palettes ──────────────────────────────────────────────────
# Dark-background palette: bright/saturated colours contrast well against dark.
_DARK_SPEAKER_COLOURS = [
    "bright_cyan",
    "bright_magenta",
    "bright_yellow",
    "bright_green",
    "bright_blue",
    "bright_red",
    "cyan",
    "magenta",
    "yellow",
    "green",
]

# Light-background palette: darker/richer shades that contrast against white.
_LIGHT_SPEAKER_COLOURS = [
    "dark_cyan",
    "dark_magenta",
    "dark_green",
    "blue",
    "red",
    "dark_orange",
    "dark_violet",
    "dark_goldenrod",
    "teal",
    "purple",
]

# Palette chosen once at import time based on the detected terminal background.
_SPEAKER_COLOURS = _DARK_SPEAKER_COLOURS if _is_dark_terminal() else _LIGHT_SPEAKER_COLOURS

_PREFIX = "  "


def _speaker_colour(speaker_id: int) -> str:
    return _SPEAKER_COLOURS[speaker_id % len(_SPEAKER_COLOURS)]


def _rich_print(
    text: str,
    speaker_id: Optional[int] = None,
    show_speaker_tag: bool = False,
    console: Optional[Console] = None,
) -> None:
    """Print a finalised line, optionally coloured (and tagged) by speaker.

    When *speaker_id* is set, the text is coloured with that speaker's colour.
    The ``[Speaker N]`` label prefix is only shown when *show_speaker_tag* is
    ``True`` (default: colour-only, no tag).

    Pass *console* to override the module-level ``_console`` (e.g. pass a
    ``Console(no_color=True)`` instance for plain-text output).
    """
    c = console if console is not None else _console
    if speaker_id is not None:
        colour = _speaker_colour(speaker_id)
        t = Text()
        if show_speaker_tag:
            t.append(f"{_PREFIX}[Speaker {speaker_id}] ", style=f"bold {colour}")
            t.append(text, style=colour)
        else:
            t.append(f"{_PREFIX}{text}", style=colour)
        c.print(t)
    else:
        c.print(f"{_PREFIX}{text}")


def _emit_segment(
    text: str,
    start_s: float,
    end_s: float,
    speaker_id: Optional[int],
    show_speaker_tag: bool,
    json_output: bool,
    console: Console,
) -> None:
    """Emit a finalised segment — JSON line or Rich-formatted text.

    JSON mode (``--json``)::

        {"type": "segment", "text": "hello world", "start": 0.0, "end": 1.5}
        {"type": "segment", "text": "how are you", "start": 1.5, "end": 3.2, "speaker": 0}

    The ``"speaker"`` key is only included when diarization is active.  Times
    are in seconds, rounded to 3 decimal places.

    Plain mode uses Rich-formatted coloured output per speaker.
    """
    if json_output:
        obj: dict = {
            "type": "segment",
            "text": text,
            "start": round(start_s, 3),
            "end": round(end_s, 3),
        }
        if speaker_id is not None:
            obj["speaker"] = speaker_id
        print(json.dumps(obj, ensure_ascii=False), flush=True)
    else:
        _rich_print(text, speaker_id, show_speaker_tag=show_speaker_tag, console=console)


def _dominant_speaker(result: Any) -> int:
    """Return the speaker id that covers the most time in this segment."""
    segments = result.sort_by_start_time()
    if not segments:
        return 0
    duration: dict[int, float] = {}
    for seg in segments:
        duration[seg.speaker] = duration.get(seg.speaker, 0.0) + (seg.end - seg.start)
    return max(duration, key=duration.__getitem__)


# ── Result types ─────────────────────────────────────────────────────────────

class _ASRResult(NamedTuple):
    text: str
    tokens: list[str]
    timestamps: list[float]


class _PendingSegment(NamedTuple):
    asr_future: Future
    diar_future: Optional[Future]
    start_s: float
    end_s: float


# ── Core ASR helpers ─────────────────────────────────────────────────────────

def _run_asr(
    recognizer: Any,
    samples: np.ndarray,
    sample_rate: int,
) -> str:
    """Decode *samples* with an offline recognizer and return stripped text."""
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def _run_asr_full(
    recognizer: Any,
    samples: np.ndarray,
    sample_rate: int,
) -> _ASRResult:
    """Decode *samples* and return text + per-token timing (when available)."""
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    result = stream.result
    text = result.text.strip()
    tokens: list[str] = list(getattr(result, "tokens", []) or [])
    timestamps: list[float] = list(getattr(result, "timestamps", []) or [])
    return _ASRResult(text=text, tokens=tokens, timestamps=timestamps)


def _print_word_timestamps(tokens: list[str], timestamps: list[float]) -> None:
    """Print per-token timing in a compact table below the transcript line."""
    if not tokens or not timestamps:
        return
    pairs = list(zip(tokens, timestamps, strict=False))
    parts = "  ".join(f"{t}@{ts:.2f}s" for t, ts in pairs)
    _console.print(f"{_PREFIX}  [{parts}]", style="dim")


# ── Subtitle / caption writers ───────────────────────────────────────────────

def _fmt_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_srt(
    subtitles: list[tuple[float, float, str]],
    path: str,
) -> None:
    """Write *subtitles* as an SRT file to *path*.

    Each subtitle is a (start_s, end_s, text) tuple.
    """
    lines: list[str] = []
    for idx, (start, end, text) in enumerate(subtitles, start=1):
        lines.append(str(idx))
        lines.append(f"{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}")
        lines.append(text)
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_vtt(
    subtitles: list[tuple[float, float, str]],
    path: str,
) -> None:
    """Write *subtitles* as a WebVTT file to *path*."""
    lines: list[str] = ["WEBVTT", ""]
    for start, end, text in subtitles:
        lines.append(f"{_fmt_vtt_time(start)} --> {_fmt_vtt_time(end)}")
        lines.append(text)
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_txt(
    subtitles: list[tuple[float, float, str]],
    path: str,
) -> None:
    """Write plain transcript text (one line per segment) to *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        for _, _, text in subtitles:
            fh.write(text + "\n")


# ── Online (streaming) recogniser loop ──────────────────────────────────────

def run_streaming(
    recognizer: Any,
    audio_gen: Generator[np.ndarray, None, None],
    sample_rate: int = 16000,
    show_mic_level: bool = False,
    diarization: Any = None,
    show_speaker_tag: bool = False,
    word_timestamps: bool = False,
    punctuation: Any = None,
    subtitles: Optional[list[tuple[float, float, str]]] = None,
    final_only: bool = False,
    json_output: bool = False,
    no_color: bool = False,
) -> None:
    """Feed incremental audio chunks into the recognizer and render output.

    Display strategy:
      - Partial hypotheses: overwrite the current terminal line with \\r
        (near-zero latency feedback, avoids line spam).  Suppressed when
        *json_output* is ``True`` or *final_only* is ``True``.
      - Finalized segments: printed on a new line when an endpoint is detected.

    When *diarization* is provided the accumulated audio for each utterance is
    sent to the diarization pipeline in a background thread so that it runs
    concurrently with the next ASR utterance, keeping added latency near zero.
    Each speaker's output is colour-coded; when *show_speaker_tag* is ``True``
    a ``[Speaker N]`` prefix is also printed.

    When *word_timestamps* is ``True``, per-token timing is printed after each
    finalised segment (model-dependent; silently skipped when unavailable).

    When *punctuation* is set (an ``OfflinePunctuation`` instance), the
    finalised text is punctuated before display.

    When *subtitles* is a list, finalised (start_s, end_s, text) tuples are
    appended to it for later serialisation.

    When *final_only* is ``True``, intermediate partial hypotheses are suppressed
    and only finalised segments are printed.

    When *json_output* is ``True``, each finalised segment is emitted as a
    JSON line (``{"type": "segment", "text": ..., "start": ..., "end": ...}``)
    suitable for piping to downstream tools.  Partial hypotheses are suppressed
    in this mode.

    When *no_color* is ``True``, terminal ANSI colour codes are disabled so
    the output can be redirected to a file or piped to tools that do not
    interpret colour escapes.
    """
    # Build the output console once — reuse for every segment in this call.
    _out_console = (
        Console(no_color=True, highlight=False, markup=False) if no_color else _console
    )
    stream = recognizer.create_stream()
    last_partial = ""
    # Buffer raw audio for the current utterance (used for diarization).
    audio_buf: list[np.ndarray] = []
    elapsed_s = 0.0
    utt_start_s = 0.0  # start of the current (in-progress) utterance
    # In JSON mode the mic-level bar would corrupt the JSON stream; disable it.
    if json_output:
        show_mic_level = False
    executor: Optional[ThreadPoolExecutor] = (
        ThreadPoolExecutor(max_workers=1) if diarization is not None else None
    )
    pending: Optional[Future] = None  # diarization future for the *previous* utterance

    def _submit_diarization(samples: np.ndarray) -> Optional[Future]:
        if executor is None or diarization is None:  # pragma: no cover
            return None
        return executor.submit(diarization.process, samples)

    def _flush_pending(pending_future: Optional[Future], pending_text: str, pending_start: float, pending_end: float) -> None:
        """Print the pending utterance with its diarization label (if available)."""
        if not pending_text:
            return
        final_text = pending_text
        if punctuation is not None:
            try:
                final_text = punctuation.add_punctuation(final_text)
            except Exception as exc:
                logging.debug("Punctuation failed: %s", exc)
        speaker_id = None
        if pending_future is not None and pending_future.done():
            try:
                result = pending_future.result()
                speaker_id = _dominant_speaker(result)
            except Exception as exc:
                logging.debug(
                    "Diarization failed for utterance %r: %s",
                    final_text,
                    exc,
                    exc_info=True,
                )
        _emit_segment(final_text, pending_start, pending_end, speaker_id, show_speaker_tag, json_output, _out_console)
        if word_timestamps and not json_output:
            result_obj = recognizer.get_result(stream)
            _print_word_timestamps(
                list(getattr(result_obj, "tokens", []) or []),
                list(getattr(result_obj, "timestamps", []) or []),
            )
        if subtitles is not None:
            subtitles.append((pending_start, pending_end, final_text))

    pending_text = ""
    pending_start = 0.0
    pending_end = 0.0

    try:
        for chunk in audio_gen:
            chunk_s = len(chunk) / sample_rate
            stream.accept_waveform(sample_rate, chunk)
            if diarization is not None:
                audio_buf.append(chunk)

            # Decode all queued frames immediately
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            text = recognizer.get_result(stream).strip()

            if show_mic_level and not text:
                energy = float(np.sqrt(np.mean(chunk ** 2)))
                bar = "█" * min(int(energy * 500), 40)
                sys.stdout.write(f"\r{_PREFIX}mic: {bar:<40} {energy:.4f}")
                sys.stdout.flush()

            if recognizer.is_endpoint(stream):
                if text:
                    if show_mic_level:
                        try:
                            width = shutil.get_terminal_size(fallback=(80, 20)).columns
                        except OSError:
                            width = 80
                        sys.stdout.write("\r" + " " * width + "\r")
                        sys.stdout.flush()
                    _clear_line(last_partial)
                    # Flush the previous utterance (diarization may now be done).
                    _flush_pending(pending, pending_text, pending_start, pending_end)
                    seg_start = utt_start_s
                    seg_end = elapsed_s + chunk_s
                    # Submit diarization for this utterance, but avoid queueing
                    # multiple diarization tasks when using a single worker.
                    if diarization is not None and audio_buf:
                        seg_audio = np.concatenate(audio_buf)
                        if pending is None or pending.done():
                            pending = _submit_diarization(seg_audio)
                            pending_text = text
                            pending_start = seg_start
                            pending_end = seg_end
                        else:
                            # Diarization worker is still busy; skip diarization
                            # for this utterance and flush plain text immediately.
                            _flush_pending(None, text, seg_start, seg_end)
                            pending_text = ""
                    else:
                        pending = None
                        pending_text = text
                        pending_start = seg_start
                        pending_end = seg_end
                        _flush_pending(None, pending_text, pending_start, pending_end)
                        pending_text = ""
                recognizer.reset(stream)
                utt_start_s = elapsed_s + chunk_s  # next utterance starts here
                audio_buf.clear()
                last_partial = ""
            elif text != last_partial:
                if not final_only and not json_output:
                    sys.stdout.write(f"\r{_PREFIX}{text}")
                    sys.stdout.flush()
                    last_partial = text
                else:
                    last_partial = ""

            elapsed_s += chunk_s

    except KeyboardInterrupt:
        pass
    finally:
        _flush_tail(recognizer, stream, sample_rate, last_partial,
                    json_output=json_output, console=_out_console,
                    start_s=utt_start_s, end_s=elapsed_s)
        # Flush the last pending diarization result.
        _flush_pending(pending, pending_text, pending_start, pending_end)
        if executor is not None:
            executor.shutdown(wait=True)


# ── Offline VAD-segmented loop ───────────────────────────────────────────────

def run_offline_vad_streaming(
    recognizer: Any,
    vad: Any,
    audio_gen: Generator[np.ndarray, None, None],
    sample_rate: int = 48000,
    show_mic_level: bool = False,
    diarization: Any = None,
    show_speaker_tag: bool = False,
    word_timestamps: bool = False,
    punctuation: Any = None,
    subtitles: Optional[list[tuple[float, float, str]]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    json_output: bool = False,
    no_color: bool = False,
) -> None:
    """VAD-segmented offline ASR with optional concurrent speaker diarization.

    ASR runs in a background executor so the mic level bar keeps updating
    while transcription is in progress. Results are flushed in submission
    order (FIFO) so the transcript stays sequential. Each speaker's output
    is colour-coded; when *show_speaker_tag* is ``True`` a ``[Speaker N]``
    prefix is also printed.

    *word_timestamps*: print per-token timing after each segment (model-dependent).
    *punctuation*: an OfflinePunctuation instance for post-processing.
    *subtitles*: if a list, (start_s, end_s, text) tuples are appended.
    *progress_callback*: called with elapsed_seconds after each audio chunk.
    *json_output*: emit each segment as a JSON line instead of styled text.
    *no_color*: disable ANSI colour codes in transcript output.
    """
    # Build the output console once — reuse for every segment in this call.
    _out_console = (
        Console(no_color=True, highlight=False, markup=False) if no_color else _console
    )
    # In JSON mode the mic-level bar would corrupt the JSON stream; disable it.
    if json_output:
        show_mic_level = False
    max_workers = 4 if diarization is not None else 2
    executor = ThreadPoolExecutor(max_workers=max_workers)
    pending: Deque[_PendingSegment] = deque()
    elapsed_s = 0.0

    def _submit(samples: np.ndarray, start_s: float, end_s: float) -> None:
        asr_f = executor.submit(_run_asr_full, recognizer, samples, sample_rate)
        diar_f = (
            executor.submit(diarization.process, samples)
            if diarization is not None
            else None
        )
        pending.append(_PendingSegment(asr_f, diar_f, start_s, end_s))

    def _print_result(seg: _PendingSegment) -> None:
        asr_result: _ASRResult = seg.asr_future.result()
        text = asr_result.text
        if not text:
            return
        if punctuation is not None:
            try:
                text = punctuation.add_punctuation(text)
            except Exception as exc:
                logging.debug("Punctuation failed: %s", exc)
        speaker_id: Optional[int] = None
        if seg.diar_future is not None:
            try:
                speaker_id = _dominant_speaker(seg.diar_future.result())
            except Exception as exc:
                logging.debug("Diarization failed: %s", exc, exc_info=True)
        if not json_output:
            try:
                width = shutil.get_terminal_size(fallback=(80, 20)).columns
            except OSError:
                width = 80
            sys.stdout.write("\r" + " " * width + "\r")
            sys.stdout.flush()
        _emit_segment(text, seg.start_s, seg.end_s, speaker_id, show_speaker_tag, json_output, _out_console)
        if word_timestamps and not json_output:
            _print_word_timestamps(asr_result.tokens, asr_result.timestamps)
        if subtitles is not None:
            subtitles.append((seg.start_s, seg.end_s, text))

    def _flush_ready() -> None:
        """Print results for futures at the front of the queue that are done."""
        while pending:
            seg = pending[0]
            if not seg.asr_future.done():
                break
            if seg.diar_future is not None and not seg.diar_future.done():
                break
            pending.popleft()
            _print_result(seg)

    def _drain_all() -> None:
        """Block until every pending result has been printed."""
        while pending:
            _print_result(pending.popleft())

    try:
        for chunk in audio_gen:
            elapsed_s += len(chunk) / sample_rate
            vad.accept_waveform(chunk)

            if show_mic_level:
                energy = float(np.sqrt(np.mean(chunk ** 2)))
                bar = "█" * min(int(energy * 500), 40)
                sys.stdout.write(f"\r{_PREFIX}mic: {bar:<40} {energy:.4f}")
                sys.stdout.flush()

            if progress_callback is not None:
                progress_callback(elapsed_s)

            while not vad.empty():
                segment = vad.front
                samples = np.array(segment.samples, dtype=np.float32)
                start_s = getattr(segment, "start", 0) / sample_rate
                end_s = start_s + len(samples) / sample_rate
                vad.pop()
                _submit(samples, start_s, end_s)

            _flush_ready()

    except KeyboardInterrupt:
        pass
    finally:
        _drain_all()
        vad.flush()
        while not vad.empty():
            segment = vad.front
            samples = np.array(segment.samples, dtype=np.float32)
            start_s = getattr(segment, "start", 0) / sample_rate
            end_s = start_s + len(samples) / sample_rate
            vad.pop()
            _submit(samples, start_s, end_s)
        executor.shutdown(wait=True)
        _drain_all()
        sys.stdout.write("\n")
        sys.stdout.flush()


def _decode_and_print(
    recognizer: Any,
    samples: np.ndarray,
    sample_rate: int,
    diarization: Any = None,
    executor: Optional[ThreadPoolExecutor] = None,
    show_speaker_tag: bool = False,
    show_mic_level: bool = False,
) -> None:
    """Run ASR (and optionally diarization) on *samples* and print the result.

    ASR and diarization are submitted concurrently to *executor* so that the
    combined latency is max(asr_time, diarization_time) rather than the sum.
    """
    if diarization is not None and executor is not None:
        asr_future = executor.submit(_run_asr, recognizer, samples, sample_rate)
        diar_future = executor.submit(diarization.process, samples)
        text = asr_future.result()
        diar_result = diar_future.result()
        speaker_id: Optional[int] = _dominant_speaker(diar_result) if text else None
    else:
        text = _run_asr(recognizer, samples, sample_rate)
        speaker_id = None

    if text:
        try:
            width = shutil.get_terminal_size(fallback=(80, 20)).columns
        except OSError:
            width = 80
        sys.stdout.write("\r" + " " * width + "\r")
        sys.stdout.flush()
        _rich_print(text, speaker_id, show_speaker_tag=show_speaker_tag)


def _clear_line(partial: str) -> None:
    """Overwrite the partial hypothesis line with spaces to prevent leftover text."""
    if partial:
        width = len(_PREFIX) + len(partial)
        sys.stdout.write(f"\r{' ' * width}\r")
        sys.stdout.flush()


def _flush_tail(
    recognizer: Any,
    stream: Any,
    sample_rate: int,
    last_partial: str,
    json_output: bool = False,
    console: Optional[Console] = None,
    start_s: float = 0.0,
    end_s: float = 0.0,
) -> None:
    """Flush any audio left in the recognizer pipeline after the loop ends.

    *json_output*: when ``True``, emit the tail segment as a JSON line.
    *console*: override the default module-level console (e.g. no-colour variant).
    *start_s*: utterance start time in seconds (set to ``utt_start_s``).
    *end_s*: elapsed time at loop exit in seconds (used as the segment end).
    """
    tail = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    text = recognizer.get_result(stream).strip()
    if text:
        c = console if console is not None else _console
        _clear_line(last_partial)
        if json_output:
            obj = {"type": "segment", "text": text, "start": round(start_s, 3), "end": round(end_s, 3)}
            print(json.dumps(obj, ensure_ascii=False), flush=True)
        else:
            c.print(f"{_PREFIX}{text}")
    if not json_output:
        sys.stdout.write("\n")
        sys.stdout.flush()

