# Project Structure

src/
│
├── main.py
├── asr_engine.py
├── streaming.py
├── audio.py
├── config.py
└── requirements.txt

## Responsibilities

main.py
- CLI
- Mode selection
- Graceful shutdown

asr_engine.py
- Sherpa-ONNX model loading
- Streaming recognizer setup
- Inference API wrapper

streaming.py
- Incremental audio feeding
- Partial decoding loop
- Terminal printing logic

audio.py
- Microphone capture (sounddevice preferred)
- WAV reading
- Resampling if needed

config.py
- Sample rate
- Chunk size
- Thread count
- Model path

## Code Requirements

- Clean
- Type hinted
- Minimal but clear comments
- No unnecessary abstraction layers
- Linux-first (Ubuntu compatible)
