from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

import update_asr_leaderboard_jp as leaderboard  # noqa: E402


def test_backfill_aggregate_metrics_recovers_ker_from_utterances():
    raw = {
        "aggregate": {
            "mean_rtf": 0.125,
        },
        "utterances": [
            {
                "reference": "かな",
                "hypothesis": "かな",
            },
            {
                "reference": "とうきょう",
                "hypothesis": "とうきょ",
            },
        ],
    }

    agg = leaderboard._backfill_aggregate_metrics(raw)

    assert agg["ker"] > 0
    assert "ker_ci_95" in agg
    assert agg["composite_score"] == (agg["ker"] + raw["aggregate"]["mean_rtf"]) / 2.0


def test_backfill_aggregate_metrics_keeps_existing_ker():
    raw = {
        "aggregate": {
            "ker": 0.25,
            "mean_rtf": 0.5,
        },
        "utterances": [
            {
                "reference": "かな",
                "hypothesis": "かな",
                "ker": 0.0,
            },
        ],
    }

    agg = leaderboard._backfill_aggregate_metrics(raw)

    assert agg == raw["aggregate"]
