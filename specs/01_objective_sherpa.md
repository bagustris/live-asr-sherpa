# Objective

You are a senior speech AI engineer.

Build a minimal, production-quality streaming ASR application in Python using Sherpa-ONNX.

## Core Goals

- Terminal-based application
- CPU-first (must run efficiently without GPU)
- Ultra-low latency streaming
- Real-time microphone transcription
- Optional offline WAV transcription
- Production-grade streaming behavior
- Clean, modular, research-friendly structure

## Entry Commands

python3 main.py --mic
python3 main.py --wav path/to/audio.wav

No GUI.
No web server.
No Docker.
Inference only.
Keep it minimal and efficient.
