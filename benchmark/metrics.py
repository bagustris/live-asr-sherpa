"""
WER, CER, RTF, latency, and composite score metrics for the live-asr-sherpa benchmark.

Metrics:
    WER (Word Error Rate):
        edit_distance(hyp_words, ref_words) / len(ref_words)
        Lower is better.  0.0 = perfect transcript.

    CER (Character Error Rate):
        edit_distance(hyp_chars, ref_chars) / len(ref_chars)
        Spaces are removed before comparison so that tokenization style does
        not affect the score.  Primary metric for Japanese (ja/jpn).
        Lower is better.

    RTF (Real-Time Factor):
        processing_time / audio_duration
        RTF < 1 → faster than real-time (required for live ASR)
        RTF > 1 → slower than real-time
        Lower is better.

    Latency (ms):
        processing_time * 1000
        Time in milliseconds to produce the transcription for one utterance.
        Lower is better.  Captures wall-clock cost per segment.

    Composite Score:
        (primary_error_rate + mean_rtf) / 2
        A single number that balances transcription quality and processing
        speed.  Uses CER for Japanese (primary_error_metric="cer") and WER
        for all other languages.  Lower is better.

WER/CER calculation:
  - tokenise hypothesis and reference by splitting on whitespace (WER) or
    individual characters after space removal (CER)
  - compute Levenshtein distance (editdistance package)
  - WER = total_edit_distance / total_reference_words
  - CER = total_char_edit_distance / total_reference_chars
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

try:
    import editdistance
except ImportError:  # pragma: no cover - exercised indirectly in lean envs
    editdistance = None


def _levenshtein_distance(left: List[str], right: List[str]) -> int:
    if editdistance is not None:
        return editdistance.eval(left, right)

    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, 1):
        current = [i]
        for j, right_token in enumerate(right, 1):
            substitution_cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation to get a canonical word sequence."""
    text = text.lower()
    # Remove punctuation except apostrophes inside words (e.g. "don't")
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse multiple spaces
    text = " ".join(text.split())
    return text


def normalize_text_for_cer(text: str) -> str:
    """Normalize text for CER computation.

    Applies Unicode NFKC normalization and lowercasing, then removes all
    whitespace so characters can be compared directly regardless of
    tokenization.  Punctuation is preserved to reflect transcription errors.
    Works correctly for Japanese (kanji, hiragana, katakana) and other
    non-space-delimited scripts.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Remove all whitespace; CER operates on characters, not words
    text = re.sub(r"\s+", "", text)
    return text


# ---------------------------------------------------------------------------
# Per-utterance metrics
# ---------------------------------------------------------------------------

@dataclass
class UtteranceResult:
    audio_path: str
    reference: str
    hypothesis: str
    audio_duration: float       # seconds
    processing_time: float      # seconds

    # Filled in by compute()
    ref_words: List[str] = field(default_factory=list)
    hyp_words: List[str] = field(default_factory=list)
    edit_distance: int = 0
    wer: float = 0.0
    ref_chars: List[str] = field(default_factory=list)
    hyp_chars: List[str] = field(default_factory=list)
    char_edit_distance: int = 0
    cer: float = 0.0
    rtf: float = 0.0
    latency_ms: float = 0.0

    def compute(self) -> "UtteranceResult":
        ref_norm = normalize_text(self.reference)
        hyp_norm = normalize_text(self.hypothesis)

        self.ref_words = ref_norm.split()
        self.hyp_words = hyp_norm.split()

        self.edit_distance = _levenshtein_distance(self.hyp_words, self.ref_words)

        ref_len = len(self.ref_words)
        self.wer = (self.edit_distance / ref_len) if ref_len > 0 else 0.0

        ref_cer_norm = normalize_text_for_cer(self.reference)
        hyp_cer_norm = normalize_text_for_cer(self.hypothesis)
        self.ref_chars = list(ref_cer_norm)
        self.hyp_chars = list(hyp_cer_norm)
        self.char_edit_distance = _levenshtein_distance(self.hyp_chars, self.ref_chars)
        ref_char_len = len(self.ref_chars)
        self.cer = (self.char_edit_distance / ref_char_len) if ref_char_len > 0 else 0.0

        self.rtf = (
            self.processing_time / self.audio_duration
            if self.audio_duration > 0
            else float("inf")
        )
        # Latency: total wall-clock time to produce this transcription, in ms
        self.latency_ms = self.processing_time * 1000.0
        return self


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """Corpus-level WER, CER, mean RTF, mean latency, and composite score.

    Composite score = (primary_error_rate + mean_rtf) / 2

    primary_error_metric controls which error rate drives the composite:
      "wer"  — default; used for English and most space-delimited languages
      "cer"  — used for Japanese (ja/jpn) and other non-space-delimited scripts

    Provides a single ranking metric that jointly considers transcription
    quality and processing speed (RTF).  Lower is better for all metrics.
    """

    total_edit_distance: int = 0
    total_ref_words: int = 0
    total_char_edit_distance: int = 0
    total_ref_chars: int = 0
    total_audio_duration: float = 0.0
    total_processing_time: float = 0.0
    n_utterances: int = 0
    primary_error_metric: str = "wer"  # "wer" or "cer"

    @property
    def wer(self) -> float:
        """Corpus-level WER: sum(edit_dist) / sum(ref_words)."""
        return (
            self.total_edit_distance / self.total_ref_words
            if self.total_ref_words > 0
            else 0.0
        )

    @property
    def wer_pct(self) -> float:
        return self.wer * 100

    @property
    def cer(self) -> float:
        """Corpus-level CER: sum(char_edit_dist) / sum(ref_chars)."""
        return (
            self.total_char_edit_distance / self.total_ref_chars
            if self.total_ref_chars > 0
            else 0.0
        )

    @property
    def cer_pct(self) -> float:
        return self.cer * 100

    @property
    def mean_rtf(self) -> float:
        """Average per-utterance processing time / audio duration."""
        return (
            self.total_processing_time / self.total_audio_duration
            if self.total_audio_duration > 0
            else float("inf")
        )

    @property
    def mean_latency_ms(self) -> float:
        """Average latency (processing time) per utterance in milliseconds."""
        return (
            self.total_processing_time / self.n_utterances * 1000.0
            if self.n_utterances > 0
            else float("inf")
        )

    @property
    def composite_score(self) -> float:
        """Single ranking metric: (primary_error_rate + mean_RTF) / 2.

        Uses CER when primary_error_metric == "cer" (e.g. Japanese),
        otherwise uses WER.  Lower is better.
        """
        error_rate = self.cer if self.primary_error_metric == "cer" else self.wer
        return (error_rate + self.mean_rtf) / 2.0

    @classmethod
    def from_results(
        cls,
        results: List[UtteranceResult],
        primary_error_metric: str = "wer",
    ) -> "AggregateMetrics":
        m = cls(primary_error_metric=primary_error_metric)
        for r in results:
            m.total_edit_distance += r.edit_distance
            m.total_ref_words += len(r.ref_words)
            m.total_char_edit_distance += r.char_edit_distance
            m.total_ref_chars += len(r.ref_chars)
            m.total_audio_duration += r.audio_duration
            m.total_processing_time += r.processing_time
            m.n_utterances += 1
        return m
