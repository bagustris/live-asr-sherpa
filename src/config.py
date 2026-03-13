from dataclasses import dataclass


@dataclass
class Config:
    model_dir: str = "models/zipformer-en-2023"
    sample_rate: int = 16000
    chunk_size: float = 0.16   # seconds (~2560 samples at 16 kHz)
    num_threads: int = 4
    # Passed directly to sherpa-onnx as the model_type hint.
    # Online values:  "" (auto), transducer, zipformer, zipformer2, conformer, lstm,
    #                 paraformer, ctc, wenet_ctc, zipformer2_ctc
    # Offline values: "" (auto), transducer, nemo_transducer, paraformer, whisper,
    #                 ctc, nemo_ctc, sense_voice, moonshine, fire_red_asr
    # Leave blank to let sherpa-onnx auto-detect from model metadata.
    model_type: str = ""
    # Set to True to use the offline (VAD-segmented) pipeline instead of streaming.
    offline: bool = False
    # VAD type to use: "silero" (default) or "ten-vad".
    vad_type: str = "silero"
    # Path to the resolved VAD model .onnx file; populated by main._validate_vad().
    vad_model: str = ""
    # Ten-VAD model file variant: "ten-vad.onnx" or "ten-vad.int8.onnx" (default).
    ten_vad_model: str = "ten-vad.int8.onnx"
    vad_threshold: float = 0.5
    vad_min_silence_duration: float = 0.5
    vad_min_speech_duration: float = 0.25
    # Language code for Whisper and SenseVoice models (e.g. "en", "zh", "ja").
    language: str = "en"
    # Show a live RMS energy bar in the terminal for mic calibration.
    show_mic_level: bool = False
    # Speaker diarization options (requires --diarization flag)
    diarization: bool = False
    # Path to pyannote segmentation model file (model.onnx inside the extracted tar).
    diarization_seg_model: str = ""
    # Path to speaker embedding extractor model file (.onnx).
    diarization_emb_model: str = ""
    # Number of speakers; -1 = auto-detect via clustering threshold.
    diarization_num_speakers: int = -1
    # Clustering threshold when num_speakers == -1 (lower → more speakers).
    diarization_cluster_threshold: float = 0.5
