# Required Output Format from Claude

Claude must:

1. Briefly explain architecture decisions.
2. Provide complete code for all files.
3. Provide installation instructions.
4. Provide model download instructions.
5. Provide example usage.
6. Provide optional improvements section including:
   - Quantization
   - VAD integration
   - Endpoint tuning
   - Word timestamps
   - Confidence scores
   - Phoneme output extension

## Strict Constraints

- No Docker
- No web server
- No GUI
- No training code
- Inference only
- Keep implementation clean and hackable

## Development Philosophy

This is for a speech AI researcher.

The system must be:

- Easy to extend
- Suitable for experimentation
- Modular
- Ready for:
  - phoneme-level decoding
  - forced alignment extension
  - multilingual switching
  - downstream NLP integration
