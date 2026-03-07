from dataclasses import dataclass


@dataclass
class Config:
    model_dir: str = "model"
    sample_rate: int = 16000
    chunk_size: float = 0.16   # seconds (~2560 samples at 16 kHz)
    num_threads: int = 4
    # "online"           – streaming transducer (default, e.g. Zipformer)
    # "nemo_transducer"  – offline NeMo transducer (e.g. Parakeet TDT)
    model_type: str = "nemo_transducer"
    # Path to silero_vad.onnx; required when model_type != "online"
    vad_model: str = ""
