"""Shared utilities for Japanese ASR benchmarks.

Normalization, CER, KER, bootstrap CI, and transcription helpers used by
benchmark_ja.py (adlib-devterm), benchmark_jvnv.py (JVNV), and other Japanese
benchmark scripts.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from kana_utils import _compute_ker, _katakana_to_hiragana, _text_to_kana  # noqa: F401


# ---------------------------------------------------------------------------
# Adlib-compatible normalization & term matching
# Ported from https://github.com/holotherapper/adlib (MIT license)
# ---------------------------------------------------------------------------

_RE_ASCII = re.compile(r"[a-zA-Z0-9_.+#]")
_RE_KATAKANA = re.compile(r"[゠-ヿ]")
_RE_HIRAGANA = re.compile(r"[぀-ゟ]")
_RE_KANJI = re.compile(r"[一-鿿㐀-䶿]")


def _adlib_normalize(text: str) -> str:
    """NFC normalization + newline removal — used for term matching only."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "").replace("\r", "").replace("\n", "")
    return text


# Japanese punctuation stripped for content-level CER.
# ー (katakana long vowel mark) is intentionally excluded — it is part of words.
_JP_PUNCT_RE = re.compile(r"[。、！？・「」『』【】（）〔〕〈〉《》…―～゛゜　]")


def _content_normalize(text: str) -> str:
    """Normalize for content-level CER: strip notation variance, keep meaning.

    1. NFKC — unify fullwidth/halfwidth (Ａ→A, ａ→a, ｶ→カ)
    2. Lowercase — eliminate case differences
    3. Strip Japanese punctuation — 。、！？ etc. are not part of what was said
    4. Strip all whitespace — tokenization style should not affect the score
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _JP_PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", "", text)
    return text


def _char_category(ch: str) -> str:
    if _RE_ASCII.match(ch):
        return "ascii"
    if _RE_KATAKANA.match(ch):
        return "katakana"
    if _RE_HIRAGANA.match(ch):
        return "hiragana"
    if _RE_KANJI.match(ch):
        return "kanji"
    return "other"


def _check_boundary(text: str, start: int, end: int) -> bool:
    """True when match at text[start:end] sits at a character-category boundary."""
    if start > 0:
        if _char_category(text[start - 1]) == _char_category(text[start]):
            return False
    if end < len(text):
        if _char_category(text[end - 1]) == _char_category(text[end]):
            return False
    return True


def _find_span(
    text: str, term: str, used: Optional[List[Tuple[int, int]]] = None
) -> Optional[Tuple[int, int]]:
    start = 0
    while True:
        idx = text.find(term, start)
        if idx == -1:
            return None
        end = idx + len(term)
        if _check_boundary(text, idx, end):
            if used is None or not any(idx < ue and end > us for us, ue in used):
                return (idx, end)
        start = idx + 1


def _replace_flexible_terms(text: str, terms: List[Dict]) -> str:
    """Replace flexible alternative surface forms with their reference canonical form."""
    pairs: List[Tuple[str, str]] = []
    for term in terms:
        if term.get("type") != "flexible":
            continue
        ref = unicodedata.normalize("NFC", term["text"])
        for alt in term.get("alternatives", []):
            pairs.append((unicodedata.normalize("NFC", alt), ref))

    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    locked: List[Tuple[int, int]] = []

    def _locked(s: int, e: int) -> bool:
        return any(s < le and e > ls for ls, le in locked)

    for alt, ref in pairs:
        if not alt:
            continue
        search = 0
        while True:
            idx = text.find(alt, search)
            if idx == -1:
                break
            end = idx + len(alt)
            if _locked(idx, end) or not _check_boundary(text, idx, end):
                search = idx + 1
                continue
            text = text[:idx] + ref + text[end:]
            ref_end = idx + len(ref)
            shift = len(ref) - len(alt)
            new_locked: List[Tuple[int, int]] = []
            for ls, le in locked:
                if ls == idx and le == ref_end:
                    new_locked.append((ls, le))
                elif ls >= end:
                    new_locked.append((ls + shift, le + shift))
                else:
                    new_locked.append((ls, le))
            locked = new_locked
            locked.append((idx, ref_end))
            search = ref_end

    return text


def _levenshtein(left: List[str], right: List[str]) -> int:
    try:
        import editdistance  # noqa: PLC0415
        return editdistance.eval(left, right)
    except ImportError:
        pass
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, lt in enumerate(left, 1):
        curr = [i]
        for j, rt in enumerate(right, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (lt != rt)))
        prev = curr
    return prev[-1]


def _compute_cer(
    reference: str, hypothesis: str, terms: List[Dict]
) -> Tuple[float, int, int]:
    """Content-level CER: measures what was said, not how it was written.

    Pipeline:
    1. NFC + strip newlines (prepare text for term matching)
    2. Flexible term replacement on hypothesis (katakana↔English etc.)
    3. Content normalization on both: NFKC + lowercase + strip JP punctuation
       + strip whitespace

    Returns (cer, edit_distance, ref_char_len).
    Pass terms=[] when no term annotations are available.
    """
    ref_nfc = _adlib_normalize(reference)
    out_nfc = _adlib_normalize(hypothesis)

    if terms:
        out_nfc = _replace_flexible_terms(out_nfc, terms)

    ref_norm = _content_normalize(ref_nfc)
    out_norm = _content_normalize(out_nfc)

    if not ref_norm and not out_norm:
        return 0.0, 0, 0
    if not ref_norm:
        n = len(out_norm)
        return 1.0, n, n

    dist = _levenshtein(list(ref_norm), list(out_norm))
    ref_len = len(ref_norm)
    cer = min(dist / ref_len, 1.0)
    return cer, dist, ref_len


def _compute_term_accuracy(output: str, terms: List[Dict]) -> Dict:
    """Compute exact/flexible term accuracy for one utterance."""
    out_norm = _adlib_normalize(output)
    exact_total = exact_correct = 0
    flexible_total = flexible_correct = 0
    details: List[Dict] = []

    for term in terms:
        text = unicodedata.normalize("NFC", term["text"])
        ttype = term["type"]
        alts = [unicodedata.normalize("NFC", a) for a in term.get("alternatives", [])]
        matched = False
        matched_variant = None

        if _find_span(out_norm, text) is not None:
            matched = True
        elif ttype == "flexible":
            for alt in alts:
                if _find_span(out_norm, alt) is not None:
                    matched = True
                    matched_variant = alt
                    break

        detail: Dict = {"text": text, "type": ttype, "matched": matched}
        if matched_variant:
            detail["matched_variant"] = matched_variant
        details.append(detail)

        if ttype == "exact":
            exact_total += 1
            if matched:
                exact_correct += 1
        else:
            flexible_total += 1
            if matched:
                flexible_correct += 1

    total = exact_total + flexible_total
    correct = exact_correct + flexible_correct
    return {
        "term_accuracy": correct / total if total > 0 else 1.0,
        "exact_term_accuracy": exact_correct / exact_total if exact_total > 0 else 1.0,
        "flexible_term_accuracy": flexible_correct / flexible_total if flexible_total > 0 else 1.0,
        "correct": correct,
        "total": total,
        "exact_correct": exact_correct,
        "exact_total": exact_total,
        "flexible_correct": flexible_correct,
        "flexible_total": flexible_total,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    pairs: List[Tuple[int, int]],
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Ratio-based bootstrap CI consistent with micro-averaging."""
    if not pairs:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed=42)
    arr = np.array(pairs)
    n = len(arr)
    stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = arr[idx]
        denom = sample[:, 1].sum()
        stats[i] = np.nan if denom == 0 else min(sample[:, 0].sum() / denom, 1.0)
    valid = stats[~np.isnan(stats)]
    if len(valid) == 0:
        return (0.0, 0.0)
    alpha = (1 - ci) / 2
    return float(np.percentile(valid, alpha * 100)), float(np.percentile(valid, (1 - alpha) * 100))


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------

def transcribe_offline(recognizer, audio: np.ndarray, sample_rate: int) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def transcribe_online(
    recognizer, audio: np.ndarray, sample_rate: int, chunk_size: float = 0.1
) -> str:
    stream = recognizer.create_stream()
    chunk_samples = max(1, int(chunk_size * sample_rate))
    texts: List[str] = []
    for i in range(0, len(audio), chunk_samples):
        stream.accept_waveform(sample_rate, audio[i : i + chunk_samples])
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        if recognizer.is_endpoint(stream):
            text = recognizer.get_result(stream).strip()
            if text:
                texts.append(text)
            recognizer.reset(stream)
    tail = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    text = recognizer.get_result(stream).strip()
    if text:
        texts.append(text)
    return " ".join(texts)


# ---------------------------------------------------------------------------
# Per-group breakdown table (shared by both benchmarks)
# ---------------------------------------------------------------------------

def print_group_breakdown(results: List[Dict], group_key: str, title: str) -> None:
    """Print a CER/KER (and optionally TermAcc) breakdown table by group_key."""
    has_terms = results and "correct_terms" in results[0]
    has_ker = results and "ker_edit_distance" in results[0]
    groups: Dict[str, Dict] = defaultdict(
        lambda: {"edit_dist": 0, "ref_chars": 0, "ker_edit": 0, "ker_ref": 0,
                 "correct": 0, "total": 0, "n": 0}
    )
    for r in results:
        g = groups[r[group_key]]
        g["edit_dist"] += r["char_edit_distance"]
        g["ref_chars"] += r["ref_chars"]
        if has_ker:
            g["ker_edit"] += r["ker_edit_distance"]
            g["ker_ref"] += r["ker_ref_chars"]
        if has_terms:
            g["correct"] += r["correct_terms"]
            g["total"] += r["total_terms"]
        g["n"] += 1

    print(f"\n  ── {title} ──")
    header = f"  {'Group':<22}  {'N':>4}  {'CER':>8}"
    if has_ker:
        header += f"  {'KER':>8}"
    if has_terms:
        header += f"  {'TermAcc':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in sorted(groups):
        g = groups[name]
        cer = min(g["edit_dist"] / g["ref_chars"], 1.0) if g["ref_chars"] > 0 else 0.0
        row = f"  {name:<22}  {g['n']:>4}  {cer * 100:>7.2f}%"
        if has_ker:
            ker = min(g["ker_edit"] / g["ker_ref"], 1.0) if g["ker_ref"] > 0 else 0.0
            row += f"  {ker * 100:>7.2f}%"
        if has_terms:
            term_acc = g["correct"] / g["total"] if g["total"] > 0 else 1.0
            row += f"  {term_acc * 100:>8.2f}%"
        print(row)
