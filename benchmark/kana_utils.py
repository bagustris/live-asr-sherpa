"""Kana conversion utilities for Kana Error Rate (KER) computation.

KER is a character-level edit distance metric on hiragana, proposed in:
  https://github.com/nyosegawa/hiragana-asr

Unlike standard CER (which compares raw/normalized written characters including
kanji, katakana, ASCII, etc.), KER converts both reference and hypothesis to
hiragana before comparison.  This makes it a phonetic/kana-sequence accuracy
metric: e.g. reference 東京 and hypothesis とうきょう are treated as equivalent.

Main pipeline for _text_to_kana:
  1. Convert text to kana pronunciation using pyopenjtalk.g2p(text, kana=True)
  2. Convert katakana output to hiragana
  3. Strip all characters except hiragana (ぁ–ゖ) and the long vowel mark ー

If pyopenjtalk is not installed, _text_to_kana falls back to a lightweight
approach: NFKC normalise → strip punctuation/spaces → convert remaining
katakana to hiragana.  Kanji and other characters are dropped, so KER becomes
effectively a hiragana-only CER; this is less accurate than the pyopenjtalk
path but still provides the metric without a hard dependency.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

# Lazily imported so the module loads even without pyopenjtalk installed.
_pyopenjtalk = None
_pyopenjtalk_tried = False


def _try_import_pyopenjtalk():
    global _pyopenjtalk, _pyopenjtalk_tried
    if _pyopenjtalk_tried:
        return _pyopenjtalk
    _pyopenjtalk_tried = True
    try:
        import pyopenjtalk  # noqa: PLC0415
        _pyopenjtalk = pyopenjtalk
    except ImportError:
        _pyopenjtalk = None
    return _pyopenjtalk


# Regex matching hiragana (U+3041–U+3096) and the long vowel mark ー (U+30FC)
_RE_KEEP_KANA = re.compile(r"[^\u3041-\u3096\u30FC]")

# Katakana range U+30A1–U+30F6 (voiced/semi-voiced covered by the shift)
_KATA_START = 0x30A1
_KATA_END = 0x30F6
_KATA_TO_HIRA_SHIFT = 0x60  # katakana codepoint − 0x60 = hiragana codepoint


def _katakana_to_hiragana(text: str) -> str:
    """Convert katakana characters to their hiragana equivalents.

    Characters outside the katakana range are passed through unchanged.
    ー (U+30FC, long vowel mark) is intentionally preserved as-is.
    """
    result: List[str] = []
    for ch in text:
        cp = ord(ch)
        if _KATA_START <= cp <= _KATA_END:
            result.append(chr(cp - _KATA_TO_HIRA_SHIFT))
        else:
            result.append(ch)
    return "".join(result)


def _text_to_kana(text: str) -> str:
    """Convert Japanese text to a hiragana-only string for KER computation.

    With pyopenjtalk installed:
      1. pyopenjtalk.g2p(text, kana=True) → katakana pronunciation string
      2. Katakana → hiragana shift
      3. Strip everything except hiragana + ー

    Without pyopenjtalk (fallback):
      1. NFKC normalise
      2. Strip Japanese punctuation and whitespace
      3. Convert katakana → hiragana
      4. Strip characters that are not hiragana or ー
         (kanji, ASCII, etc. are dropped — less accurate but dependency-free)

    Returns a string containing only hiragana characters and ー.
    """
    pyoj = _try_import_pyopenjtalk()
    if pyoj is not None:
        try:
            kana = pyoj.g2p(text, kana=True)
        except Exception:
            # If G2P fails (e.g. empty input), fall through to fallback
            kana = text
    else:
        # Fallback: normalise and convert what we can
        kana = unicodedata.normalize("NFKC", text)

    kana = _katakana_to_hiragana(kana)
    kana = _RE_KEEP_KANA.sub("", kana)
    return kana


# ---------------------------------------------------------------------------
# Levenshtein helper (re-uses editdistance when available)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# KER: Kana Error Rate
# ---------------------------------------------------------------------------

def _compute_ker(
    reference: str,
    hypothesis: str,
) -> Tuple[float, int, int]:
    """Compute Kana Error Rate (KER).

    Converts both reference and hypothesis to hiragana via _text_to_kana,
    then computes character-level edit distance on the kana sequences.

    Returns (ker, kana_edit_distance, ref_kana_len).

    KER = kana_edit_distance / ref_kana_len  (capped at 1.0)
    """
    ref_kana = _text_to_kana(reference)
    hyp_kana = _text_to_kana(hypothesis)

    if not ref_kana and not hyp_kana:
        return 0.0, 0, 0
    if not ref_kana:
        n = len(hyp_kana)
        return 1.0, n, n

    dist = _levenshtein(list(ref_kana), list(hyp_kana))
    ref_len = len(ref_kana)
    ker = min(dist / ref_len, 1.0)
    return ker, dist, ref_len
