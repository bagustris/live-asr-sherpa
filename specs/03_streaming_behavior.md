# Streaming Requirements

## Microphone Mode (--mic)

- True streaming recognition
- Use Sherpa-ONNX streaming recognizer
- Show partial results live
- Overwrite line instead of printing repeatedly
- Print finalized segments clearly
- Avoid flicker
- Handle Ctrl+C gracefully

Streaming constraints:

- 16 kHz mono
- Low-latency chunk size (~0.1–0.2 sec)
- Avoid buffering full history
- Feed incremental frames to recognizer

## Offline Mode (--wav)

- Read WAV file
- Stream it chunk-by-chunk through same pipeline
- Produce final transcript
- Do NOT use separate offline recognizer unless necessary

## Latency Target

- Partial hypothesis latency < 500 ms on CPU
- No large internal buffers
