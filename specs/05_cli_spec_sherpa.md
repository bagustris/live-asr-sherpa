# CLI Specification

Use argparse.

Options:

--mic                 Run streaming from microphone
--wav PATH            Transcribe WAV file
--model-dir PATH      Path to Sherpa-ONNX model directory
--sample-rate INT     Default 16000
--chunk-size FLOAT    Default 0.16
--threads INT         CPU thread count

Validation:

- Check model files exist
- Check mic availability
- Check WAV format
- Clear error messages
- Graceful exit on Ctrl+C
