"""Tests for benchmark/metrics.py."""

import pytest
from metrics import AggregateMetrics, UtteranceResult, normalize_text


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def test_normalize_text_lowercase():
    assert normalize_text("Hello World") == "hello world"


def test_normalize_text_removes_punctuation():
    assert normalize_text("Hello, world!") == "hello world"


def test_normalize_text_preserves_apostrophes_in_words():
    assert normalize_text("don't stop") == "don't stop"


def test_normalize_text_collapses_spaces():
    assert normalize_text("  hello   world  ") == "hello world"


def test_normalize_text_empty():
    assert normalize_text("") == ""


def test_normalize_text_only_punctuation():
    assert normalize_text("!!!") == ""


# ---------------------------------------------------------------------------
# UtteranceResult.compute()
# ---------------------------------------------------------------------------

def _make_result(reference, hypothesis, duration=5.0, proc_time=0.5):
    return UtteranceResult(
        audio_path="fake.flac",
        reference=reference,
        hypothesis=hypothesis,
        audio_duration=duration,
        processing_time=proc_time,
    ).compute()


def test_utterance_perfect_match():
    r = _make_result("hello world", "hello world")
    assert r.wer == 0.0
    assert r.edit_distance == 0


def test_utterance_one_substitution():
    r = _make_result("hello world", "hello earth")
    assert r.edit_distance == 1
    assert r.wer == pytest.approx(1 / 2)


def test_utterance_wer_greater_than_one():
    # All reference words wrong plus extra hypotheses words
    r = _make_result("a", "x y z")
    assert r.wer > 1.0


def test_utterance_empty_reference():
    # WER is 0.0 when reference is empty (nothing to get wrong)
    r = _make_result("", "something")
    assert r.wer == 0.0


def test_utterance_rtf():
    r = _make_result("hello", "hello", duration=4.0, proc_time=0.4)
    assert r.rtf == pytest.approx(0.1)


def test_utterance_rtf_zero_duration():
    r = _make_result("hello", "hello", duration=0.0, proc_time=0.5)
    assert r.rtf == float("inf")


def test_utterance_latency_ms():
    r = _make_result("hello", "hello", proc_time=0.25)
    assert r.latency_ms == pytest.approx(250.0)


def test_utterance_compute_returns_self():
    utt = UtteranceResult(
        audio_path="x",
        reference="test",
        hypothesis="test",
        audio_duration=1.0,
        processing_time=0.1,
    )
    result = utt.compute()
    assert result is utt


def test_utterance_normalises_text_before_wer():
    # Punctuation and capitalisation should not affect WER
    r = _make_result("Hello, World!", "hello world")
    assert r.wer == 0.0


# ---------------------------------------------------------------------------
# AggregateMetrics.from_results()
# ---------------------------------------------------------------------------

def _build_agg(*args):
    results = [_make_result(*a) for a in args]
    return AggregateMetrics.from_results(results)


def test_aggregate_n_utterances():
    agg = _build_agg(("hello", "hello"), ("world", "word"))
    assert agg.n_utterances == 2


def test_aggregate_corpus_wer():
    # ref: "hello world" (2 words), hyp: "hello earth" (1 error)
    # ref: "foo bar"    (2 words), hyp: "foo bar"    (0 errors)
    # corpus WER = 1 / 4 = 0.25
    results = [
        _make_result("hello world", "hello earth"),
        _make_result("foo bar", "foo bar"),
    ]
    agg = AggregateMetrics.from_results(results)
    assert agg.wer == pytest.approx(0.25)
    assert agg.wer_pct == pytest.approx(25.0)


def test_aggregate_mean_rtf():
    # Two utterances with duration=4s, proc=0.4s → RTF = 0.1 each
    results = [
        _make_result("a b", "a b", duration=4.0, proc_time=0.4),
        _make_result("c d", "c d", duration=4.0, proc_time=0.4),
    ]
    agg = AggregateMetrics.from_results(results)
    assert agg.mean_rtf == pytest.approx(0.1)


def test_aggregate_mean_latency_ms():
    # proc_time=0.5s per utterance → mean_latency_ms = 500ms
    results = [
        _make_result("a", "a", proc_time=0.5),
        _make_result("b", "b", proc_time=0.5),
    ]
    agg = AggregateMetrics.from_results(results)
    assert agg.mean_latency_ms == pytest.approx(500.0)


def test_aggregate_composite_score():
    # WER=0 and RTF=0.1 → composite = (0 + 0.1) / 2 = 0.05
    results = [
        _make_result("hello world", "hello world", duration=4.0, proc_time=0.4),
    ]
    agg = AggregateMetrics.from_results(results)
    assert agg.composite_score == pytest.approx((agg.wer + agg.mean_rtf) / 2.0)


def test_aggregate_empty_results():
    agg = AggregateMetrics.from_results([])
    assert agg.n_utterances == 0
    assert agg.wer == 0.0
    assert agg.mean_rtf == float("inf")
    assert agg.mean_latency_ms == float("inf")


def test_aggregate_total_audio_duration():
    results = [
        _make_result("a", "a", duration=3.0, proc_time=0.1),
        _make_result("b", "b", duration=5.0, proc_time=0.2),
    ]
    agg = AggregateMetrics.from_results(results)
    assert agg.total_audio_duration == pytest.approx(8.0)


def test_aggregate_total_processing_time():
    results = [
        _make_result("a", "a", duration=3.0, proc_time=0.1),
        _make_result("b", "b", duration=5.0, proc_time=0.3),
    ]
    agg = AggregateMetrics.from_results(results)
    assert agg.total_processing_time == pytest.approx(0.4)


def test_composite_score_lower_for_better_model():
    # Perfect model: WER=0, RTF=0.05 → composite=0.025
    perfect = [_make_result("hello world", "hello world", duration=4.0, proc_time=0.2)]
    agg_perfect = AggregateMetrics.from_results(perfect)

    # Worse model: WER=0.5, RTF=0.2 → composite=0.35
    worse = [_make_result("hello world", "hello earth", duration=4.0, proc_time=0.8)]
    agg_worse = AggregateMetrics.from_results(worse)

    assert agg_perfect.composite_score < agg_worse.composite_score
