# Japanese ASR Leaderboard

HF Open ASR Leaderboard–style ranking of Japanese ASR models on [adlib-devterm](https://huggingface.co/datasets/holotherapper/adlib-devterm) and [JVNV v1](https://ss-takashi.sakura.ne.jp/corpus/jvnv/). Lower is better for every column.

Numbers come from `benchmark/benchmark_ja.py` and `benchmark/benchmark_jvnv.py` (offline pipeline, 4 threads, `--language ja`). Regenerate this file with `python benchmark/update_asr_leaderboard_jp.py`.

| Model | Avg CER | Avg RTF | Avg Composite | ADLIB-DEVTERM CER | JVNV CER |
|-------|--------:|--------:|--------------:|------------------:|---------:|
| ReazonSpeech JA | 28.95% | 0.0225 | 0.1560 | 46.09% | 11.81% |
| ReazonSpeech JA-EN | 29.54% | 0.0248 | 0.1601 | 47.15% | 11.94% |
| ReazonSpeech JA-EN-MLS-5k | 29.54% | 0.0225 | 0.1590 | 47.15% | 11.94% |
| Parakeet CTC JA (int8) | 29.63% | 0.0310 | 0.1636 | 44.04% | 15.22% |
| SenseVoice | 33.84% | 0.0181 | 0.1782 | 59.20% | 8.48% |
| Whisper Large-V3 | 34.97% | 2.0866 | 1.2182 | 55.88% | 14.06% |
| Cohere Transcribe (int8) | 40.43% | 0.3085 | 0.3564 | 52.75% | 28.10% |

## Metric definitions

- **CER** — character error rate (NFKC + lowercase + punctuation-stripped)
- **RTF** — real-time factor (processing time / audio duration)
- **Composite** — `(CER + RTF) / 2`

Full details in [`benchmark/README.md`](README.md).

## Reproduce

```bash
MODEL=parakeet-ctc-ja   # or reazonspeech-ja, whisper, sense_voice, ...

python benchmark/benchmark_ja.py --offline --model-type "$MODEL" \
  --output "benchmark/results_${MODEL}_adlib.json"

python benchmark/benchmark_jvnv.py --offline --jvnv-dir /data/jvnv_v1 \
  --model-type "$MODEL" --output "benchmark/results_${MODEL}_jvnv.json"

python benchmark/update_asr_leaderboard_jp.py
```

## Sources

- `benchmark/results_cohere-transcribe-14-lang-int8_adlib.json`
- `benchmark/results_cohere-transcribe-14-lang-int8_jvnv.json`
- `benchmark/results_parakeet-ctc-ja-int8_adlib.json`
- `benchmark/results_reazonspeech-ja-en-mls-5k_adlib.json`
- `benchmark/results_reazonspeech-ja-en-mls-5k_jvnv.json`
- `benchmark/results_reazonspeech-ja-en_adlib.json`
- `benchmark/results_reazonspeech-ja-en_jvnv.json`
- `benchmark/results_reazonspeech-ja_adlib.json`
- `benchmark/results_reazonspeech-ja_jvnv.json`
- `benchmark/results_sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17_adlib.json`
- `benchmark/results_sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17_jvnv.json`
- `benchmark/results_sherpa-onnx-whisper-large-v3_adlib.json`
- `benchmark/results_whisper_large_v3_jvnv.json`
- `results_jvnv.json`
