from dataclasses import dataclass


@dataclass
class Config:
    model_dir: str = "model"
    sample_rate: int = 16000
    chunk_size: float = 0.16   # seconds (~2560 samples at 16 kHz)
    num_threads: int = 4
