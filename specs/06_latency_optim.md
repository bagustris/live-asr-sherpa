# Latency Optimization Strategy

Implement:

- Small frame chunks (~0.1–0.2 sec)
- Immediate recognizer.accept_waveform()
- Frequent recognizer.decode()
- Avoid blocking loops
- Do not reinitialize recognizer repeatedly

Optional:

- Integrate WebRTC VAD for speech-only decoding
- Support endpoint detection
- Allow configurable decoding beam

Explain tradeoffs briefly in comments.

Keep LOC reasonable.
Avoid premature optimization.
