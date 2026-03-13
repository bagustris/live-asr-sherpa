from pathlib import Path

import sherpa_onnx

from config import Config


def _find(directory: Path, pattern: str) -> str:
    """Return the first file matching glob pattern inside directory."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    return str(matches[0])


# ── Online (streaming) model type routing ────────────────────────────────────
# Supported --model-type values for online mode:
#   transducer variants : "" (auto), transducer, zipformer, zipformer2, conformer, lstm
#   paraformer          : streaming paraformer (encoder + decoder, no joiner)
#   ctc                 : generic online CTC
#   wenet_ctc           : WeNet CTC models
#   zipformer2_ctc      : Zipformer2 CTC models

_ONLINE_PARAFORMER = {"paraformer"}
_ONLINE_CTC = {"ctc"}
_ONLINE_WENET_CTC = {"wenet_ctc"}
_ONLINE_ZIPFORMER2_CTC = {"zipformer2_ctc"}

# ── Offline model type routing ───────────────────────────────────────────────
# Supported --model-type values for offline mode:
#   transducer variants : "" (auto), transducer, nemo_transducer
#   paraformer          : offline paraformer (single model file)
#   whisper             : OpenAI Whisper (encoder + decoder)
#   ctc / nemo_ctc      : CTC / NVIDIA NeMo CTC (single model file)
#   sense_voice         : FunAudioLLM SenseVoice (single model file)
#   moonshine           : UsefulSensors Moonshine (4 separate files)
#   fire_red_asr        : FireRedASR (encoder + decoder)

_OFFLINE_PARAFORMER = {"paraformer"}
_OFFLINE_WHISPER = {"whisper"}
_OFFLINE_CTC = {"ctc", "nemo_ctc"}
_OFFLINE_SENSE_VOICE = {"sense_voice"}
_OFFLINE_MOONSHINE = {"moonshine"}
_OFFLINE_FIRE_RED_ASR = {"fire_red_asr"}

_ENDPOINT = dict(
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=2.4,
    rule2_min_trailing_silence=1.2,
    rule3_min_utterance_length=300.0,  # effectively disabled (upstream recommendation)
)


def build_recognizer(cfg: Config) -> sherpa_onnx.OnlineRecognizer:
    """Load a Sherpa-ONNX streaming recognizer for any supported online model type.

    Endpoint rules (tradeoff: adds ~1–2 s boundary latency, prevents runaway lines):
      rule1: 2.4 s silence → hard endpoint
      rule2: 1.2 s silence after sufficient speech → early endpoint
      rule3: force endpoint after 300 s utterance (effectively disabled)
    """
    d = Path(cfg.model_dir)
    mt = cfg.model_type.lower()
    tokens = _find(d, "tokens.txt")

    if mt in _ONLINE_PARAFORMER:
        return sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens=tokens,
            encoder=_find(d, "encoder*.onnx"),
            decoder=_find(d, "decoder*.onnx"),
            num_threads=cfg.num_threads,
            sample_rate=cfg.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            **_ENDPOINT,
        )
    if mt in _ONLINE_WENET_CTC:
        return sherpa_onnx.OnlineRecognizer.from_wenet_ctc(
            tokens=tokens,
            model=_find(d, "model*.onnx"),
            num_threads=cfg.num_threads,
            sample_rate=cfg.sample_rate,
            feature_dim=80,
            **_ENDPOINT,
        )
    if mt in _ONLINE_ZIPFORMER2_CTC:
        return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=tokens,
            model=_find(d, "model*.onnx"),
            num_threads=cfg.num_threads,
            sample_rate=cfg.sample_rate,
            feature_dim=80,
            **_ENDPOINT,
        )
    if mt in _ONLINE_CTC:
        return sherpa_onnx.OnlineRecognizer.from_ctc(
            tokens=tokens,
            model=_find(d, "model*.onnx"),
            num_threads=cfg.num_threads,
            sample_rate=cfg.sample_rate,
            feature_dim=80,
            **_ENDPOINT,
        )
    # Default: transducer (zipformer, zipformer2, conformer, lstm, or auto-detect)
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=_find(d, "encoder*.onnx"),
        decoder=_find(d, "decoder*.onnx"),
        joiner=_find(d, "joiner*.onnx"),
        num_threads=cfg.num_threads,
        sample_rate=cfg.sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type=cfg.model_type,
        **_ENDPOINT,
    )


def build_offline_recognizer(cfg: Config) -> sherpa_onnx.OfflineRecognizer:
    """Load a Sherpa-ONNX offline recognizer for any supported offline model type.

    Pair with build_vad() and run_offline_vad_streaming() for live microphone use.
    """
    d = Path(cfg.model_dir)
    mt = cfg.model_type.lower()
    tokens = _find(d, "tokens.txt")

    if mt in _OFFLINE_WHISPER:
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            tokens=tokens,
            encoder=_find(d, "encoder*.onnx"),
            decoder=_find(d, "decoder*.onnx"),
            num_threads=cfg.num_threads,
            language=cfg.language,
            task="transcribe",
        )
    if mt in _OFFLINE_PARAFORMER:
        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            tokens=tokens,
            model=_find(d, "model*.onnx"),
            num_threads=cfg.num_threads,
            sample_rate=cfg.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
        )
    if mt in _OFFLINE_CTC:
        return sherpa_onnx.OfflineRecognizer.from_ctc(
            tokens=tokens,
            model=_find(d, "model*.onnx"),
            num_threads=cfg.num_threads,
            sample_rate=cfg.sample_rate,
            feature_dim=80,
        )
    if mt in _OFFLINE_SENSE_VOICE:
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            tokens=tokens,
            model=_find(d, "model*.onnx"),
            num_threads=cfg.num_threads,
            language=cfg.language,
            use_itn=True,
        )
    if mt in _OFFLINE_MOONSHINE:
        return sherpa_onnx.OfflineRecognizer.from_moonshine(
            tokens=tokens,
            preprocessor=_find(d, "preprocess*.onnx"),
            encoder=_find(d, "encode.onnx"),
            uncached_decoder=_find(d, "uncached_decode*.onnx"),
            cached_decoder=_find(d, "cached_decode*.onnx"),
            num_threads=cfg.num_threads,
        )
    if mt in _OFFLINE_FIRE_RED_ASR:
        return sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
            tokens=tokens,
            encoder=_find(d, "encoder*.onnx"),
            decoder=_find(d, "decoder*.onnx"),
            num_threads=cfg.num_threads,
        )
    # Default: transducer (nemo_transducer or auto-detect)
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens,
        encoder=_find(d, "encoder*.onnx"),
        decoder=_find(d, "decoder*.onnx"),
        joiner=_find(d, "joiner*.onnx"),
        num_threads=cfg.num_threads,
        sample_rate=cfg.sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type=cfg.model_type,
    )


def build_vad(cfg: Config) -> sherpa_onnx.VoiceActivityDetector:
    """Build a VAD (Silero or Ten-VAD) for segmenting live audio into utterances.

    Required for offline models because they cannot decode incrementally —
    the VAD accumulates audio until silence is detected, then the full
    segment is sent to the offline recognizer.
    """
    if not cfg.vad_model:
        raise ValueError(
            "--vad-model is required for offline model types.\n"
            "Choose a VAD type with --vad-model silero (default) or --vad-model ten-vad."
        )
    if cfg.vad_type == "ten-vad":
        vad_config = sherpa_onnx.VadModelConfig(
            ten_vad=sherpa_onnx.TenVadModelConfig(
                model=cfg.vad_model,
                threshold=cfg.vad_threshold,
                min_silence_duration=cfg.vad_min_silence_duration,
                min_speech_duration=cfg.vad_min_speech_duration,
            ),
            sample_rate=cfg.sample_rate,
            num_threads=cfg.num_threads,
        )
    else:
        vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=cfg.vad_model,
                threshold=cfg.vad_threshold,
                min_silence_duration=cfg.vad_min_silence_duration,
                min_speech_duration=cfg.vad_min_speech_duration,
            ),
            sample_rate=cfg.sample_rate,
            num_threads=cfg.num_threads,
        )
    return sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)


def build_diarization(cfg: Config) -> sherpa_onnx.OfflineSpeakerDiarization:
    """Build an OfflineSpeakerDiarization pipeline using pyannote segmentation.

    Models required (auto-downloaded by main.py when --diarization is set):
      - diarization_seg_model: path to pyannote model.onnx
      - diarization_emb_model: path to speaker embedding extractor .onnx
    """
    if not cfg.diarization_seg_model or not cfg.diarization_emb_model:
        raise ValueError(
            "Both diarization_seg_model and diarization_emb_model must be set."
        )
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=cfg.diarization_seg_model,
            ),
            num_threads=cfg.num_threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=cfg.diarization_emb_model,
            num_threads=cfg.num_threads,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=cfg.diarization_num_speakers,
            threshold=cfg.diarization_cluster_threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError(
            "Diarization config is invalid. "
            "Check that the model files exist and are valid .onnx files."
        )
    return sherpa_onnx.OfflineSpeakerDiarization(config)
