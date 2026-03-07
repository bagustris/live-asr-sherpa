from dataclasses import dataclass


@dataclass
class Config:
    model_dir: str = "models/zipformer-en-2023"
    sample_rate: int = 16000
    chunk_size: float = 0.16   # seconds (~2560 samples at 16 kHz)
    num_threads: int = 4
    # online-transducer  – streaming transducer (default, e.g. Zipformer)
    # online-paraformer  – streaming Paraformer
    # online-ctc         – streaming CTC (e.g. Zipformer-CTC)
    # offline-transducer – offline transducer (e.g. NeMo Parakeet TDT)
    # offline-paraformer – offline Paraformer
    # offline-ctc        – offline CTC (NeMo, WeNet, icefall)
    # whisper            – OpenAI Whisper
    # sense-voice        – SenseVoice
    model_type: str = "online-transducer"
    # Path to silero_vad.onnx; required when model_type != "online"
    vad_model: str = ""
    vad_threshold: float = 0.5
    vad_min_silence_duration: float = 0.5
    vad_min_speech_duration: float = 0.25
