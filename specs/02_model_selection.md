# Model Selection (Sherpa-ONNX Optimized)

Use Sherpa-ONNX streaming models optimized for CPU.

## Preferred Model Type

Streaming transducer models (Zipformer or Conformer transducer).

Preferred examples:

- sherpa-onnx-streaming-zipformer-en-2023-06-26
- sherpa-onnx-streaming-conformer-en

Select the best CPU-efficient English streaming model available from Sherpa-ONNX.

## Why Sherpa-ONNX?

- Designed for streaming
- Low memory footprint
- Efficient C++ backend
- Optimized for CPU
- Incremental decoding supported natively

If multiple options exist:
- Choose the lowest latency CPU-friendly model
- Explain your choice briefly

## Runtime Configuration

- CPU execution provider only
- Allow configurable num_threads
- Enable graph optimization
- No GPU dependencies
