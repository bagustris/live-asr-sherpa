#!/usr/bin/env python3
"""
Japanese ASR Diarization Benchmark on Callhome Japan and Sakura datasets.

Evaluates diarization-aware ASR by loading full conversation audio from
HuggingFace (talkbank/callhome Japanese subset, talkbank/sakura) and
reference transcriptions from local .cha zip archives.

Metrics:
  - RTF         real-time factor (processing_time / audio_duration)
  - Latency     mean per-conversation processing time (ms)
  - DER         diarization error rate (missed + false alarm + confusion)
  - CER         character error rate (content-level, chronological concat)
  - cpCER       concatenated minimum-permutation CER (optimal speaker mapping)

Usage examples:
    # Smoke test: 2 conversations, verbose output
    python benchmark/benchmark_diarization.py \\
      --dataset callhome --offline --max-convs 2 --verbose

    # Full run, save results
    python benchmark/benchmark_diarization.py \\
      --dataset both --offline \\
      --output benchmark/results_diarization_parakeet.json

    # Regenerate leaderboard after saving results
    python benchmark/update_asr_diarization_leaderboard_jp.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import zipfile
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

_PROJECT_DIR = _BENCH_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from benchmark_utils import (  # noqa: E402
    _bootstrap_ci,
    _compute_cer,
    _compute_ker,
    transcribe_offline,
)

DEFAULT_CALLHOME_ZIP = str(_PROJECT_DIR / "data" / "jpn.zip")
DEFAULT_SAKURA_ZIP = str(_PROJECT_DIR / "data" / "Sakura.zip")
DEFAULT_OFFLINE_MODEL_TYPE = "parakeet-ctc-ja"

# Maps --model-type to the default model directory name under models/
# Must stay in sync with the target names in sherox/asr.py
_DEFAULT_MODEL_DIR_BY_TYPE: Dict[str, str] = {
    "parakeet-ctc-ja":            "parakeet-ctc-ja-int8",
    "reazonspeech-ja":            "reazonspeech-ja",
    "reazonspeech-ja-en":         "reazonspeech-ja-en",
    "reazonspeech-ja-en-mls-5k":  "reazonspeech-ja-en-mls-5k",
    "whisper":                    "sherpa-onnx-whisper-large-v3",
    "sense_voice":                "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    "cohere_transcribe":          "cohere-transcribe-14-lang-int8",
}

_MODEL_TYPE_ALIASES = {
    "parakeet-ctc-ja": "nemo_ctc",
    "reazonspeech-ja": "",
    "reazonspeech-ja-en": "",
    "reazonspeech-ja-en-mls-5k": "",
}

# CHAT annotation cleanup regexes
_RE_CORRECTION = re.compile(r"\[:\s*([^\]]+)\]")   # [: word] → word
_RE_NONVERBAL = re.compile(r"&=\w+")               # &=noise, &=laugh etc.
_RE_RETRACE = re.compile(r"\[/{1,3}\]")            # [/], [//], [///]
_RE_OVERLAP_OPEN = re.compile(r"<")                 # < (keep inner content)
_RE_OVERLAP_CLOSE = re.compile(r">")
_RE_PROSODIC = re.compile(r"[⇘⇗→↑↓⇒⇑]")
_RE_AT_SUFFIX = re.compile(r"@\w+")                # @g, @l, @s suffixes
_RE_TIMESTAMP = re.compile(r"\x15(\d+)_(\d+)\x15")  # CHAT bullet time-code \x15start_end\x15
_RE_UNINTELLIGIBLE = re.compile(r"\bxxx\b")
_RE_EXTRA_SPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# CHAT (.cha) file parsing
# ---------------------------------------------------------------------------

def _clean_chat_text(raw: str) -> str:
    """Strip CHAT annotations from utterance text, return plain Japanese text."""
    text = _RE_TIMESTAMP.sub("", raw)
    text = text.replace("\x15", "")  # remove any stray CHAT bullet markers
    text = _RE_CORRECTION.sub(r"\1", text)
    text = _RE_NONVERBAL.sub("", text)
    text = _RE_UNINTELLIGIBLE.sub("", text)
    text = _RE_RETRACE.sub("", text)
    text = re.sub(r"<([^>]*)>", r"\1", text)
    text = _RE_PROSODIC.sub("", text)
    text = _RE_AT_SUFFIX.sub("", text)
    text = text.replace(",", " ").replace(".", " ")
    text = _RE_EXTRA_SPACE.sub(" ", text).strip()
    return text


def parse_cha(content: str) -> List[Dict]:
    """Parse CHAT-format .cha file into a list of utterance dicts.

    Returns list of {speaker, start_ms, end_ms, text}.
    Utterances without a timestamp get start_ms=end_ms=None.
    """
    lines = content.splitlines()
    utterances: List[Dict] = []
    current_speaker: Optional[str] = None
    current_lines: List[str] = []

    def _flush():
        if current_speaker is None or not current_lines:
            return
        raw = " ".join(current_lines)
        m = _RE_TIMESTAMP.search(raw)
        start_ms = int(m.group(1)) if m else None
        end_ms = int(m.group(2)) if m else None
        text = _clean_chat_text(raw)
        if text:
            utterances.append({
                "speaker": current_speaker,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text,
            })

    for line in lines:
        if line.startswith("@") or line.startswith("%"):
            _flush()
            current_speaker = None
            current_lines = []
            continue
        if line.startswith("*"):
            _flush()
            current_speaker = None
            current_lines = []
            colon = line.find(":")
            if colon == -1:
                continue
            current_speaker = line[1:colon].strip()
            current_lines = [line[colon + 1:].strip()]
        elif line.startswith("\t") and current_speaker is not None:
            current_lines.append(line.strip())

    _flush()
    return utterances


def load_cha_from_zip(zip_path: Path) -> Dict[str, List[Dict]]:
    """Load all .cha files from a zip archive.

    Returns {conversation_id: [utterances]} where id is the stem (e.g. "2238").
    """
    result: Dict[str, List[Dict]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".cha"):
                continue
            stem = Path(name).stem
            content = zf.read(name).decode("utf-8", errors="replace")
            result[stem] = parse_cha(content)
    return result


# ---------------------------------------------------------------------------
# Audio loading from HuggingFace
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio
    try:
        import soxr  # noqa: PLC0415
        return soxr.resample(audio, src_sr, tgt_sr, quality="HQ").astype(np.float32)
    except ImportError:
        from math import gcd  # noqa: PLC0415
        from scipy.signal import resample_poly  # noqa: PLC0415
        g = gcd(tgt_sr, src_sr)
        return resample_poly(
            audio, tgt_sr // g, src_sr // g,
            window=("kaiser", 14.0), padtype="line",
        ).astype(np.float32)


def load_hf_samples(
    hf_name: str,
    hf_config: Optional[str],
    cha_ids: List[str],
    cha_map: Dict[str, List[Dict]],
    max_convs: Optional[int],
    target_sr: int = 16000,
) -> List[Dict]:
    """Stream audio from HuggingFace and pair with local .cha transcriptions.

    The HF records have no explicit conversation ID.  They are matched to .cha
    files by position: record[i] ↔ cha_ids[i] (both sorted alphabetically).

    The HF dataset's timestamps_start/end/speakers fields are used as the
    ground-truth diarization reference for DER.  The .cha text is used for CER.

    Returns list of {id, dataset, audio, audio_duration,
                      ref_diar_segs, ref_text_by_spk}.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("pip install datasets") from exc

    load_kwargs: Dict = {"streaming": True}
    for split_name in ("test", "data", "train"):
        try:
            if hf_config:
                ds = load_dataset(hf_name, hf_config, split=split_name, **load_kwargs)
            else:
                ds = load_dataset(hf_name, split=split_name, **load_kwargs)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError(
            f"Could not find a usable split in {hf_name}. "
            "Try specifying the dataset split manually."
        )

    samples: List[Dict] = []
    for idx, record in enumerate(ds):
        if idx >= len(cha_ids):
            break

        conv_id = cha_ids[idx]

        audio_data = record["audio"]
        audio = np.array(audio_data["array"], dtype=np.float32)
        sr = int(audio_data["sampling_rate"])
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = _resample(audio, sr, target_sr)

        # Ground-truth diarization from HF (complete, second-precision)
        ref_diar_segs = [
            {"start_s": s, "end_s": e, "speaker": spk}
            for s, e, spk in zip(
                record.get("timestamps_start", []),
                record.get("timestamps_end", []),
                record.get("speakers", []),
            )
        ]

        # Ground-truth text from .cha file (per speaker)
        cha_utts = cha_map.get(conv_id, [])
        ref_text_by_spk: Dict[str, List[str]] = {}
        for utt in cha_utts:
            if utt["text"]:
                ref_text_by_spk.setdefault(utt["speaker"], []).append(utt["text"])

        samples.append({
            "id": conv_id,
            "dataset": hf_name,
            "audio": audio,
            "audio_duration": len(audio) / target_sr,
            "ref_diar_segs": ref_diar_segs,
            "ref_text_by_spk": ref_text_by_spk,
        })

        if max_convs and len(samples) >= max_convs:
            break

    return samples


# ---------------------------------------------------------------------------
# DER computation
# ---------------------------------------------------------------------------

def compute_der(
    ref_segs: List[Dict],
    hyp_segs: List[Tuple[float, float, int]],
    collar: float = 0.25,
    frame_ms: int = 10,
) -> float:
    """Compute Diarization Error Rate at frame resolution.

    ref_segs: list of {start_s, end_s, speaker}  (from HF dataset)
    hyp_segs: list of (start_s, end_s, speaker_id_int)  (from diarizer)
    collar:   ignore frames within ±collar seconds of reference boundaries
    Returns DER float (0.0–1.0+).
    """
    if not ref_segs or not hyp_segs:
        return 1.0 if ref_segs else 0.0

    # Duration in ms → number of frames
    max_ms = max(
        max(int(s["end_s"] * 1000) for s in ref_segs),
        max(int(s[1] * 1000) for s in hyp_segs),
    )
    n_frames = max_ms // frame_ms + 1

    ref_speakers = sorted({s["speaker"] for s in ref_segs})
    hyp_speakers = sorted({s[2] for s in hyp_segs})
    ref_idx = {spk: i for i, spk in enumerate(ref_speakers)}
    hyp_idx = {spk: i for i, spk in enumerate(hyp_speakers)}

    # Frame arrays: -1 = silence
    ref_arr = np.full(n_frames, -1, dtype=np.int16)
    hyp_arr = np.full(n_frames, -1, dtype=np.int16)

    # Collar mask: True = excluded from scoring
    collar_mask = np.zeros(n_frames, dtype=bool)
    collar_frames = int(collar * 1000 / frame_ms)

    for seg in ref_segs:
        f_start = max(0, int(seg["start_s"] * 1000 / frame_ms))
        f_end = min(n_frames - 1, int(seg["end_s"] * 1000 / frame_ms))
        ref_arr[f_start:f_end + 1] = ref_idx[seg["speaker"]]
        for f in range(max(0, f_start - collar_frames), min(n_frames, f_end + 1 + collar_frames)):
            collar_mask[f] = True
        # Narrow collar to boundary regions only (overwrite interior)
        if f_end > f_start + 2 * collar_frames:
            interior_start = f_start + collar_frames
            interior_end = f_end - collar_frames
            collar_mask[interior_start:interior_end] = False

    for start_s, end_s, spk in hyp_segs:
        f_start = max(0, int(start_s * 1000 / frame_ms))
        f_end = min(n_frames - 1, int(end_s * 1000 / frame_ms))
        hyp_arr[f_start:f_end + 1] = hyp_idx[spk]

    # Scoring mask: exclude collar and silence-in-both
    active = ~collar_mask
    ref_speech = active & (ref_arr >= 0)
    total_ref = ref_speech.sum()
    if total_ref == 0:
        return 0.0

    # Missed speech: ref has speaker, hyp has silence
    missed = (ref_speech & (hyp_arr < 0)).sum()

    # False alarm: ref has silence, hyp has speaker
    fa = (active & (ref_arr < 0) & (hyp_arr >= 0)).sum()

    # Build confusion matrix on frames where both have a speaker
    both_speech = active & (ref_arr >= 0) & (hyp_arr >= 0)
    n_ref = len(ref_speakers)
    n_hyp = len(hyp_speakers)
    confusion_matrix = np.zeros((n_ref, n_hyp), dtype=np.int64)
    if both_speech.any():
        r = ref_arr[both_speech]
        h = hyp_arr[both_speech]
        for ri, hi in zip(r, h):
            confusion_matrix[ri, hi] += 1

    # Optimal speaker mapping via Hungarian algorithm
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        correct = confusion_matrix[row_ind, col_ind].sum()
    except ImportError:
        # Brute-force fallback for small cases
        if n_ref <= 8 and n_hyp <= 8:
            best = 0
            n_min = min(n_ref, n_hyp)
            for perm in permutations(range(n_hyp), n_min):
                c = sum(confusion_matrix[i, perm[i]] for i in range(n_min))
                if c > best:
                    best = c
            correct = best
        else:
            # Greedy diagonal approximation
            correct = min(confusion_matrix[i, i] for i in range(min(n_ref, n_hyp)))

    total_both = both_speech.sum()
    speaker_confusion = total_both - correct

    der = (missed + fa + speaker_confusion) / total_ref
    return float(der)


# ---------------------------------------------------------------------------
# cpCER computation
# ---------------------------------------------------------------------------

def compute_cpcer(
    ref_by_spk: Dict[str, str],
    hyp_by_spk: Dict[int, str],
) -> float:
    """Concatenated minimum-permutation CER.

    Finds the mapping of hypothesis speaker IDs to reference speakers that
    minimises total edit distance, then returns the CER for that mapping.

    ref_by_spk: {speaker_label: concatenated_reference_text}
    hyp_by_spk: {speaker_id_int: concatenated_hypothesis_text}
    """
    ref_labels = sorted(ref_by_spk.keys())
    hyp_ids = sorted(hyp_by_spk.keys())
    n_ref = len(ref_labels)
    n_hyp = len(hyp_ids)

    if n_ref == 0 or n_hyp == 0:
        return 1.0

    # Build CER cost matrix [ref × hyp]
    cost = np.zeros((n_ref, n_hyp), dtype=np.float64)
    edit_dist = np.zeros((n_ref, n_hyp), dtype=np.int64)
    for i, rl in enumerate(ref_labels):
        for j, hid in enumerate(hyp_ids):
            _, ed, _ = _compute_cer(ref_by_spk[rl], hyp_by_spk[hid], [])
            edit_dist[i, j] = ed
            cost[i, j] = ed

    # Find optimal assignment
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
        row_ind, col_ind = linear_sum_assignment(cost)
        total_edit = edit_dist[row_ind, col_ind].sum()
    except ImportError:
        if n_ref <= 8 and n_hyp <= 8:
            n_min = min(n_ref, n_hyp)
            best_edit = None
            for perm in permutations(range(n_hyp), n_min):
                ed = sum(edit_dist[i, perm[i]] for i in range(n_min))
                if best_edit is None or ed < best_edit:
                    best_edit = ed
            total_edit = best_edit if best_edit is not None else 0
        else:
            row_ind = list(range(min(n_ref, n_hyp)))
            col_ind = list(range(min(n_ref, n_hyp)))
            total_edit = edit_dist[row_ind, col_ind].sum()

    # Unmatched reference speakers contribute their full ref length as errors
    matched_ref = set(row_ind) if n_ref <= n_hyp else set(row_ind)
    total_ref_chars = 0
    for i, rl in enumerate(ref_labels):
        _, _, ref_len = _compute_cer(ref_by_spk[rl], "", [])
        total_ref_chars += ref_len
        if i not in matched_ref:
            total_edit += ref_len

    if total_ref_chars == 0:
        return 0.0
    return min(total_edit / total_ref_chars, 1.0)


def compute_cpker(
    ref_by_spk: Dict[str, str],
    hyp_by_spk: Dict[int, str],
) -> float:
    """Concatenated minimum-permutation KER (cpKER).

    Like compute_cpcer but uses kana-converted text for all comparisons.

    Algorithm (per the issue specification):
      1. Each speaker's text has already been concatenated by the caller.
      2. Convert each speaker's concatenated reference text to kana/hiragana.
      3. Convert each speaker's concatenated hypothesis text to kana/hiragana.
      4. Punctuation/whitespace is stripped by _text_to_kana consistently.
      5. Build the speaker cost matrix using kana edit distance.
      6. Pick the minimum speaker permutation (Hungarian algorithm).
      7. Return total_kana_edit_distance / total_reference_kana_chars.

    ref_by_spk: {speaker_label: concatenated_reference_text}
    hyp_by_spk: {speaker_id_int: concatenated_hypothesis_text}
    """
    from kana_utils import _compute_ker as _ker, _text_to_kana  # noqa: PLC0415

    ref_labels = sorted(ref_by_spk.keys())
    hyp_ids = sorted(hyp_by_spk.keys())
    n_ref = len(ref_labels)
    n_hyp = len(hyp_ids)

    if n_ref == 0 or n_hyp == 0:
        return 1.0

    # Build KER cost matrix [ref × hyp]
    cost = np.zeros((n_ref, n_hyp), dtype=np.float64)
    edit_dist = np.zeros((n_ref, n_hyp), dtype=np.int64)
    for i, rl in enumerate(ref_labels):
        for j, hid in enumerate(hyp_ids):
            _, ed, _ = _ker(ref_by_spk[rl], hyp_by_spk[hid])
            edit_dist[i, j] = ed
            cost[i, j] = ed

    # Find optimal assignment
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
        row_ind, col_ind = linear_sum_assignment(cost)
        total_edit = edit_dist[row_ind, col_ind].sum()
    except ImportError:
        if n_ref <= 8 and n_hyp <= 8:
            n_min = min(n_ref, n_hyp)
            best_edit = None
            for perm in permutations(range(n_hyp), n_min):
                ed = sum(edit_dist[i, perm[i]] for i in range(n_min))
                if best_edit is None or ed < best_edit:
                    best_edit = ed
            total_edit = best_edit if best_edit is not None else 0
        else:
            row_ind = list(range(min(n_ref, n_hyp)))
            col_ind = list(range(min(n_ref, n_hyp)))
            total_edit = edit_dist[row_ind, col_ind].sum()

    # Unmatched reference speakers contribute their full kana ref length as errors
    matched_ref = set(row_ind) if n_ref <= n_hyp else set(row_ind)
    total_ref_kana = 0
    for i, rl in enumerate(ref_labels):
        _, _, kana_len = _ker(ref_by_spk[rl], "")
        total_ref_kana += kana_len
        if i not in matched_ref:
            total_edit += kana_len

    if total_ref_kana == 0:
        return 0.0
    return min(total_edit / total_ref_kana, 1.0)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    diarizer,
    recognizer,
    samples: List[Dict],
    sample_rate: int = 16000,
    verbose: bool = False,
) -> Tuple[List[Dict], Dict]:
    """Run diarization+ASR on each conversation and compute DER/CER/cpCER/KER/cpKER."""
    results: List[Dict] = []

    total_edit_dist = 0
    total_ref_chars = 0
    total_cp_edit_dist = 0
    total_cp_ref_chars = 0
    total_ker_edit = 0
    total_ker_ref = 0
    total_cpker_edit = 0
    total_cpker_ref = 0
    total_der_num = 0.0
    total_der_den = 0
    total_audio = 0.0
    total_proc = 0.0
    cer_pairs: List[Tuple[int, int]] = []
    ker_pairs: List[Tuple[int, int]] = []

    for i, sample in enumerate(samples, 1):
        conv_id = sample["id"]
        audio = sample["audio"]
        duration = sample["audio_duration"]
        ref_diar_segs = sample["ref_diar_segs"]        # from HF: {start_s,end_s,speaker}
        ref_text_by_spk = sample["ref_text_by_spk"]   # from .cha: {speaker: [texts]}

        print(f"  [{i:3d}/{len(samples)}] {conv_id}  ({duration:.1f}s)", flush=True)

        t_start = time.monotonic()
        try:
            raw_hyp_segs = diarizer.process(audio).sort_by_start_time()
            hyp_segs: List[Tuple[float, float, int, str]] = []
            for seg in raw_hyp_segs:
                seg_audio = audio[int(seg.start * sample_rate): int(seg.end * sample_rate)]
                if len(seg_audio) < 160:
                    continue
                text = transcribe_offline(recognizer, seg_audio, sample_rate)
                hyp_segs.append((seg.start, seg.end, seg.speaker, text))
        except Exception as exc:
            print(f"    ERROR: {exc}", flush=True)
            hyp_segs = []
        proc_time = time.monotonic() - t_start
        rtf = proc_time / duration if duration > 0 else float("inf")

        # DER: HF ground-truth diarization vs hypothesis
        der = compute_der(
            ref_diar_segs,
            [(s[0], s[1], s[2]) for s in hyp_segs],
        )

        # CER: chronological concat of all .cha texts vs hypothesis
        ref_text_chron = " ".join(
            t for texts in ref_text_by_spk.values() for t in texts
        )
        hyp_text_chron = " ".join(s[3] for s in hyp_segs if s[3])
        _, edit_dist, ref_len = _compute_cer(ref_text_chron, hyp_text_chron, [])
        cer = min(edit_dist / ref_len, 1.0) if ref_len > 0 else 0.0

        # KER: chronological concat, kana-converted
        _, ker_edit_dist, ker_ref_len = _compute_ker(ref_text_chron, hyp_text_chron)
        ker = min(ker_edit_dist / ker_ref_len, 1.0) if ker_ref_len > 0 else 0.0

        # cpCER: per-speaker concat
        ref_by_spk_str = {k: " ".join(v) for k, v in ref_text_by_spk.items()}
        hyp_by_spk: Dict[int, List[str]] = {}
        for s in hyp_segs:
            if s[3]:
                hyp_by_spk.setdefault(s[2], []).append(s[3])
        hyp_by_spk_str = {k: " ".join(v) for k, v in hyp_by_spk.items()}

        cpcer = compute_cpcer(ref_by_spk_str, hyp_by_spk_str)
        cpker = compute_cpker(ref_by_spk_str, hyp_by_spk_str)

        # Aggregate tracking
        total_edit_dist += edit_dist
        total_ref_chars += ref_len
        total_ker_edit += ker_edit_dist
        total_ker_ref += ker_ref_len
        cer_pairs.append((edit_dist, ref_len))
        ker_pairs.append((ker_edit_dist, ker_ref_len))

        # cpCER aggregation
        for rt in ref_by_spk_str.values():
            _, _, rl_len = _compute_cer(rt, "", [])
            total_cp_ref_chars += rl_len
        total_cp_edit_dist += int(cpcer * sum(
            _compute_cer(rt, "", [])[2] for rt in ref_by_spk_str.values()
        ))

        # cpKER aggregation
        for rt in ref_by_spk_str.values():
            _, _, kana_len = _compute_ker(rt, "")
            total_cpker_ref += kana_len
        total_cpker_edit += int(cpker * sum(
            _compute_ker(rt, "")[2] for rt in ref_by_spk_str.values()
        ))

        ref_dur = sum(s["end_s"] - s["start_s"] for s in ref_diar_segs)
        total_der_num += der * ref_dur
        total_der_den += ref_dur

        total_audio += duration
        total_proc += proc_time

        conv_result = {
            "id": conv_id,
            "dataset": sample["dataset"],
            "audio_duration_s": duration,
            "processing_time_s": proc_time,
            "rtf": rtf,
            "latency_ms": proc_time * 1000.0,
            "der": der,
            "cer": cer,
            "cpcer": cpcer,
            "ker": ker,
            "cpker": cpker,
            "ref_diar_segs": ref_diar_segs,
            "hypothesis_segments": [
                {"start_s": s[0], "end_s": s[1], "speaker": s[2], "text": s[3]}
                for s in hyp_segs
            ],
        }
        results.append(conv_result)

        marker = "✓" if der < 0.2 else "✗"
        print(
            f"    {marker}  RTF={rtf:.3f}  DER={der * 100:.1f}%"
            f"  CER={cer * 100:.1f}%  cpCER={cpcer * 100:.1f}%"
            f"  KER={ker * 100:.1f}%  cpKER={cpker * 100:.1f}%"
            f"  Lat={proc_time * 1000:.0f}ms",
            flush=True,
        )
        if verbose and hyp_segs:
            print(f"    REF (concat): {ref_text_chron[:120]}")
            print(f"    HYP (concat): {hyp_text_chron[:120]}")

    n = len(results)
    micro_cer = min(total_edit_dist / total_ref_chars, 1.0) if total_ref_chars > 0 else 0.0
    micro_ker = min(total_ker_edit / total_ker_ref, 1.0) if total_ker_ref > 0 else 0.0
    micro_cpcer = min(total_cp_edit_dist / total_cp_ref_chars, 1.0) if total_cp_ref_chars > 0 else 0.0
    micro_cpker = min(total_cpker_edit / total_cpker_ref, 1.0) if total_cpker_ref > 0 else 0.0
    macro_der = total_der_num / total_der_den if total_der_den > 0 else 0.0
    mean_rtf = total_proc / total_audio if total_audio > 0 else float("inf")
    mean_lat = total_proc / n * 1000.0 if n > 0 else float("inf")
    cer_ci = _bootstrap_ci(cer_pairs)
    ker_ci = _bootstrap_ci(ker_pairs)

    print(f"\n  ── Aggregate ──")
    print(f"  Conversations    : {n}")
    print(f"  DER              : {macro_der * 100:.2f}%")
    print(f"  CER              : {micro_cer * 100:.2f}%"
          f"  (95% CI: {cer_ci[0] * 100:.1f}% – {cer_ci[1] * 100:.1f}%)")
    print(f"  cpCER            : {micro_cpcer * 100:.2f}%")
    print(f"  KER              : {micro_ker * 100:.2f}%"
          f"  (95% CI: {ker_ci[0] * 100:.1f}% – {ker_ci[1] * 100:.1f}%)")
    print(f"  cpKER            : {micro_cpker * 100:.2f}%")
    print(f"  Mean RTF         : {mean_rtf:.4f}")
    print(f"  Mean Latency     : {mean_lat:.1f} ms")
    print(f"  Audio total      : {total_audio:.1f}s")
    print(f"  Proc total       : {total_proc:.1f}s")

    agg = {
        "n_conversations": n,
        "der": macro_der,
        "cer": micro_cer,
        "cer_ci_95": list(cer_ci),
        "cpcer": micro_cpcer,
        "ker": micro_ker,
        "ker_ci_95": list(ker_ci),
        "cpker": micro_cpker,
        "mean_rtf": mean_rtf,
        "mean_latency_ms": mean_lat,
        "total_audio_s": total_audio,
        "total_proc_s": total_proc,
    }
    return results, agg


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(model_name: str, results: List[Dict], agg: Dict) -> None:
    datasets = sorted({r["dataset"] for r in results})
    print("\n" + "=" * 72)
    print("  SUMMARY — Japanese ASR Diarization Benchmark")
    print("=" * 72)
    print(f"  Model             : {model_name}")
    print(f"  Datasets          : {', '.join(datasets)}")
    print(f"  Conversations     : {agg['n_conversations']}")
    print(f"  DER               : {agg['der'] * 100:.2f}%")
    print(f"  CER               : {agg['cer'] * 100:.2f}%"
          f"  (95% CI: {agg['cer_ci_95'][0] * 100:.1f}% – {agg['cer_ci_95'][1] * 100:.1f}%)")
    print(f"  cpCER             : {agg['cpcer'] * 100:.2f}%")
    print(f"  KER               : {agg['ker'] * 100:.2f}%"
          f"  (95% CI: {agg['ker_ci_95'][0] * 100:.1f}% – {agg['ker_ci_95'][1] * 100:.1f}%)")
    print(f"  cpKER             : {agg['cpker'] * 100:.2f}%")
    print(f"  Mean RTF          : {agg['mean_rtf']:.4f}")
    print(f"  Mean Latency (ms) : {agg['mean_latency_ms']:.1f}")
    print("=" * 72)

    # Per-dataset breakdown
    for ds in datasets:
        ds_results = [r for r in results if r["dataset"] == ds]
        if not ds_results:
            continue
        avg_der = sum(r["der"] for r in ds_results) / len(ds_results)
        avg_cer = sum(r["cer"] for r in ds_results) / len(ds_results)
        avg_cpcer = sum(r["cpcer"] for r in ds_results) / len(ds_results)
        avg_ker = sum(r["ker"] for r in ds_results) / len(ds_results)
        avg_cpker = sum(r["cpker"] for r in ds_results) / len(ds_results)
        print(f"\n  {ds}  ({len(ds_results)} conversations)")
        print(f"    DER   : {avg_der * 100:.2f}%")
        print(f"    CER   : {avg_cer * 100:.2f}%")
        print(f"    cpCER : {avg_cpcer * 100:.2f}%")
        print(f"    KER   : {avg_ker * 100:.2f}%")
        print(f"    cpKER : {avg_cpker * 100:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark Japanese ASR diarization on Callhome Japan and Sakura datasets. "
            "Audio is streamed from HuggingFace; reference transcriptions are parsed from "
            "local .cha zip archives."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", default="both", choices=["callhome", "sakura", "both"],
        help="Dataset(s) to benchmark",
    )
    p.add_argument(
        "--callhome-zip", default=DEFAULT_CALLHOME_ZIP, metavar="PATH",
        help="Path to jpn.zip with Callhome Japan .cha transcriptions",
    )
    p.add_argument(
        "--sakura-zip", default=DEFAULT_SAKURA_ZIP, metavar="PATH",
        help="Path to Sakura.zip with Sakura .cha transcriptions",
    )
    p.add_argument("--model-dir", default=None, metavar="PATH")
    p.add_argument("--model-type", default=DEFAULT_OFFLINE_MODEL_TYPE, metavar="TYPE")
    p.add_argument("--offline", action="store_true", help="Use offline (non-streaming) recognizer")
    p.add_argument("--diarization-seg-model", default="", metavar="PATH",
                   help="Path to pyannote segmentation model.onnx (auto-downloaded if empty)")
    p.add_argument("--diarization-emb-model", default="", metavar="PATH",
                   help="Path to speaker embedding model.onnx (auto-downloaded if empty)")
    p.add_argument("--num-speakers", type=int, default=-1, metavar="N",
                   help="Known number of speakers per conversation (-1 = auto-detect)")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--language", default="ja", metavar="LANG")
    p.add_argument("--max-convs", type=int, default=None, metavar="N",
                   help="Limit number of conversations (smoke test)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print concatenated REF/HYP per conversation")
    p.add_argument("--output", metavar="FILE",
                   help="Save full results to JSON")
    return p


def _validate_args(args: argparse.Namespace) -> None:
    if not args.offline:
        print("Warning: --offline is recommended for diarization benchmarks.", file=sys.stderr)
    if args.dataset in ("callhome", "both"):
        zp = Path(args.callhome_zip)
        if not zp.exists():
            print(f"Error: Callhome zip not found: {zp}", file=sys.stderr)
            sys.exit(1)
    if args.dataset in ("sakura", "both"):
        zp = Path(args.sakura_zip)
        if not zp.exists():
            print(f"Error: Sakura zip not found: {zp}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args)

    from sherox.asr import _validate_diarization_models, _validate_model  # noqa: PLC0415
    from sherox.asr_engine import build_diarization, build_offline_recognizer, build_recognizer  # noqa: PLC0415
    from sherox.config import Config  # noqa: PLC0415

    # Resolve model directory: explicit > per-type default > parakeet fallback
    if args.model_dir:
        model_dir = args.model_dir
    else:
        dir_name = _DEFAULT_MODEL_DIR_BY_TYPE.get(
            args.model_type, "parakeet-ctc-ja-int8"
        )
        model_dir = str(_PROJECT_DIR / "models" / dir_name)

    engine_model_type = _MODEL_TYPE_ALIASES.get(args.model_type, args.model_type)

    # Auto-download ASR model if the directory is missing
    _validate_model(model_dir, args.model_type)

    cfg = Config(
        model_dir=model_dir,
        model_type=engine_model_type,
        offline=args.offline,
        num_threads=args.threads,
        sample_rate=args.sample_rate,
        language=args.language,
        diarization=True,
        diarization_num_speakers=args.num_speakers,
    )

    # Resolve / download diarization models
    print("\nResolving diarization models…")
    cfg.diarization_seg_model, cfg.diarization_emb_model = _validate_diarization_models(
        args.diarization_seg_model, args.diarization_emb_model, _PROJECT_DIR
    )

    mode_str = "offline" if args.offline else "online (streaming)"
    print(f"\nLoading {mode_str} recognizer: {Path(model_dir).name}")
    print(f"  model_type : '{args.model_type or '(auto-detect)'}'")
    print(f"  language   : {cfg.language}")
    print(f"  threads    : {cfg.num_threads}")
    t0 = time.monotonic()
    recognizer = build_offline_recognizer(cfg) if args.offline else build_recognizer(cfg)
    print(f"  Loaded in {time.monotonic() - t0:.1f}s")

    print("\nLoading diarization models…")
    t0 = time.monotonic()
    diarizer = build_diarization(cfg)
    print(f"  Loaded in {time.monotonic() - t0:.1f}s\n")

    # Load transcriptions from zip files
    all_samples: List[Dict] = []

    if args.dataset in ("callhome", "both"):
        print(f"Parsing Callhome Japan transcriptions from {args.callhome_zip} …")
        cha_map = load_cha_from_zip(Path(args.callhome_zip))
        cha_ids = sorted(cha_map.keys())  # sorted order matches HF record order
        print(f"  {len(cha_map)} conversations found in zip")
        print("Loading Callhome Japan audio from HuggingFace (talkbank/callhome) …")
        samples = load_hf_samples(
            "talkbank/callhome", "jpn", cha_ids, cha_map,
            max_convs=args.max_convs,
            target_sr=args.sample_rate,
        )
        print(f"  {len(samples)} conversations loaded\n")
        for s in samples:
            s["dataset"] = "callhome-jpn"
        all_samples.extend(samples)

    if args.dataset in ("sakura", "both"):
        remaining = None
        if args.max_convs:
            remaining = max(0, args.max_convs - len(all_samples))
            if remaining == 0:
                print("Skipping Sakura (max-convs already reached)")
        if remaining != 0:
            print(f"Parsing Sakura transcriptions from {args.sakura_zip} …")
            cha_map = load_cha_from_zip(Path(args.sakura_zip))
            cha_ids = sorted(cha_map.keys())
            print(f"  {len(cha_map)} conversations found in zip")
            print("Loading Sakura audio from HuggingFace (talkbank/sakura) …")
            samples = load_hf_samples(
                "talkbank/sakura", None, cha_ids, cha_map,
                max_convs=remaining,
                target_sr=args.sample_rate,
            )
            print(f"  {len(samples)} conversations loaded\n")
            for s in samples:
                s["dataset"] = "sakura"
            all_samples.extend(samples)

    if not all_samples:
        print(
            "Error: No conversations loaded. "
            "Check that the HuggingFace datasets are accessible and the zip paths are correct.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Running benchmark on {len(all_samples)} conversation(s)…\n")
    results, agg = run_benchmark(
        diarizer, recognizer, all_samples,
        sample_rate=args.sample_rate,
        verbose=args.verbose,
    )

    print_summary(Path(model_dir).name, results, agg)

    if args.output:
        datasets_run = sorted({r["dataset"] for r in results})
        output_data = {
            "dataset": datasets_run[0] if len(datasets_run) == 1 else "mixed",
            "datasets": datasets_run,
            "model_dir": model_dir,
            "model_type": args.model_type,
            "offline": args.offline,
            "language": args.language,
            "threads": cfg.num_threads,
            "num_speakers": args.num_speakers,
            "aggregate": agg,
            "conversations": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
