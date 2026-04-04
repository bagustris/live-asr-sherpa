"""
WER, RTF, latency, and composite score metrics for the live-asr-sherpa benchmark.

Metrics:
    WER (Word Error Rate):
        edit_distance(hyp_words, ref_words) / len(ref_words)
        Lower is better.  0.0 = perfect transcript.

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
        (wer + mean_rtf) / 2
        A single number that balances transcription quality (WER) and
        processing speed (RTF).  Lower is better.

WER calculation follows speechain/criterion/error_rate.py:
  - tokenise hypothesis and reference by splitting on whitespace
  - compute Levenshtein distance (editdistance package)
  - WER = total_edit_distance / total_reference_words
"""

from __future__ import annotations

import re
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
    """Corpus-level WER, mean RTF, mean latency, and composite score.

    Composite score = (wer + mean_rtf) / 2

    Provides a single ranking metric that jointly considers transcription
    quality (WER) and processing speed (RTF).  Lower is better for all metrics.
    """

    total_edit_distance: int = 0
    total_ref_words: int = 0
    total_audio_duration: float = 0.0
    total_processing_time: float = 0.0
    n_utterances: int = 0

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
        """Single ranking metric: (WER + mean_RTF) / 2.

        Balances transcription quality (WER) and real-time speed (RTF).
        Lower is better.  A model with both low WER and low RTF will score best.
        """
        return (self.wer + self.mean_rtf) / 2.0

    @classmethod
    def from_results(cls, results: List[UtteranceResult]) -> "AggregateMetrics":
        m = cls()
        for r in results:
            m.total_edit_distance += r.edit_distance
            m.total_ref_words += len(r.ref_words)
            m.total_audio_duration += r.audio_duration
            m.total_processing_time += r.processing_time
            m.n_utterances += 1
        return m
