from pathlib import Path

import sherpa_onnx

from config import Config


def _find(directory: Path, pattern: str) -> str:
    """Return the first file matching glob pattern inside directory."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    return str(matches[0])


def build_recognizer(cfg: Config) -> sherpa_onnx.OnlineRecognizer:
    """Load a Sherpa-ONNX streaming transducer (Zipformer/Conformer) recognizer.

    Model choice: sherpa-onnx-streaming-zipformer-en-2023-06-26
    Rationale: lowest published WER among CPU-only English streaming transducers;
    encoder runs ~3–4× real-time on a single modern CPU core.

    Endpoint rules (tradeoff: adds ~1–2 s boundary latency, prevents runaway lines):
      rule1: 2.4 s silence → hard endpoint
      rule2: 1.2 s silence after sufficient speech → early endpoint
      rule3: force endpoint after 20 s utterance (prevents infinite segments)
    """
    d = Path(cfg.model_dir)
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=_find(d, "tokens.txt"),
        encoder=_find(d, "encoder*.onnx"),
        decoder=_find(d, "decoder*.onnx"),
        joiner=_find(d, "joiner*.onnx"),
        num_threads=cfg.num_threads,
        sample_rate=cfg.sample_rate,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20.0,
        decoding_method="greedy_search",
    )
    return recognizer
