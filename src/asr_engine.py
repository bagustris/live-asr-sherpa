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

    Endpoint rules (tradeoff: adds ~1–2 s boundary latency, prevents runaway lines):
      rule1: 2.4 s silence → hard endpoint
      rule2: 1.2 s silence after sufficient speech → early endpoint
      rule3: force endpoint after 20 s utterance (prevents infinite segments)
    """
    d = Path(cfg.model_dir)
    return sherpa_onnx.OnlineRecognizer.from_transducer(
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


def build_offline_recognizer(cfg: Config) -> sherpa_onnx.OfflineRecognizer:
    """Load a Sherpa-ONNX offline NeMo transducer (e.g. Parakeet TDT).

    Uses OfflineRecognizer with model_type="nemo_transducer". Pair this with
    build_vad() and run_offline_vad_streaming() for live microphone use.
    """
    d = Path(cfg.model_dir)
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=_find(d, "tokens.txt"),
        encoder=_find(d, "encoder*.onnx"),
        decoder=_find(d, "decoder*.onnx"),
        joiner=_find(d, "joiner*.onnx"),
        num_threads=cfg.num_threads,
        sample_rate=cfg.sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemo_transducer",
    )


def build_vad(cfg: Config) -> sherpa_onnx.VoiceActivityDetector:
    """Build a Silero VAD for segmenting live audio into utterances.

    Required for offline models (model_type != "online") because they cannot
    decode incrementally — the VAD accumulates audio until silence is detected,
    then the full segment is sent to the offline recognizer.
    """
    if not cfg.vad_model:
        raise ValueError(
            "--vad-model is required for offline model types.\n"
            "Download silero_vad.onnx with:\n"
            "  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "asr-models/silero_vad.onnx"
        )
    vad_config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=cfg.vad_model,
            threshold=0.5,
            min_silence_duration=0.5,
            min_speech_duration=0.25,
        ),
        sample_rate=cfg.sample_rate,
        num_threads=cfg.num_threads,
    )
    return sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)
