# Japanese ASR Leaderboard

HF Open ASR Leaderboard–style ranking of Japanese ASR models on [adlib-devterm](https://huggingface.co/datasets/holotherapper/adlib-devterm), [JVNV v1](https://ss-takashi.sakura.ne.jp/corpus/jvnv/), [JSUT Basic5000](https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000), and [JVS](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus). Lower is better for every column.

Numbers come from `benchmark/benchmark_ja.py`, `benchmark/benchmark_jvnv.py`, `benchmark/benchmark_jsut.py`, and `benchmark/benchmark_jvs.py` (offline pipeline, 4 threads, `--language ja`). Regenerate this file with `python benchmark/update_asr_leaderboard_jp.py`.

| Model | Avg KER | Avg CER | Avg RTF | Avg Composite | ADLIB-DEVTERM KER | JVNV KER | JSUT5000 KER | JVS KER |
|-------|--------:|--------:|--------:|--------------:|----------:|----------:|----------:|----------:|
| Parakeet CTC JA (int8) | 14.04% | 18.22% | 0.0311 | 0.0857 | 45.57% | 4.32% | 2.90% | 3.36% |
| ReazonSpeech JA | 23.03% | 17.89% | 0.0299 | 0.1301 | 75.87% | 11.03% | 2.09% | 3.14% |
| ReazonSpeech JA-EN-MLS-5k | 24.16% | 18.38% | 0.0231 | 0.1323 | 79.85% | 10.99% | 2.86% | 2.95% |
| ReazonSpeech JA-EN | 24.16% | 18.38% | 0.0237 | 0.1327 | 79.85% | 10.99% | 2.86% | 2.95% |
| SenseVoice | 24.41% | 21.03% | 0.0252 | 0.1346 | 81.36% | 7.06% | 3.77% | 5.43% |
| Whisper Turbo | 27.97% | 25.55% | 0.5738 | 0.4268 | 53.46% | 16.28% | 21.21% | 20.93% |
| Cohere Transcribe (int8) | 32.92% | 25.84% | 0.4173 | 0.3732 | 78.79% | 33.20% | 10.31% | 9.37% |
| Whisper Large-V3 | 33.41% | 29.29% | 2.1632 | 1.2487 | 72.05% | 6.99% | 21.18% | — |
| Whisper Distil Large-V3.5 | 93.47% | 100.00% | 0.2914 | 0.6131 | 86.95% | 100.00% | — | — |

## Metric definitions

- **KER** — Kana Error Rate (text converted to hiragana/kana before edit distance)
- **CER** — character error rate (NFKC + lowercase + punctuation-stripped)
- **RTF** — real-time factor (processing time / audio duration)
- **Composite** — `(KER + RTF) / 2`

KER is the primary metric: it handles kanji/kana equivalence (e.g., 東京 ≡ とうきょう) by converting text to hiragana via pyopenjtalk G2P before comparison, making it a more phonetically accurate measure of ASR quality.

Full details in [`benchmark/README.md`](README.md).

## Reproduce

```bash
MODEL=parakeet-ctc-ja   # or reazonspeech-ja, whisper, sense_voice, ...

python benchmark/benchmark_ja.py --offline --model-type "$MODEL" \
  --output "benchmark/results_${MODEL}_adlib.json"

python benchmark/benchmark_jvnv.py --offline --jvnv-dir /data/jvnv_v1 \
  --model-type "$MODEL" --output "benchmark/results_${MODEL}_jvnv.json"

python benchmark/benchmark_jsut.py --offline --model-type "$MODEL" \
  --output "benchmark/results_${MODEL}_jsut.json"

python benchmark/benchmark_jvs.py --offline --jvs-dir /data/jvs_ver1 \
  --model-type "$MODEL" --output "benchmark/results_${MODEL}_jvs.json"

python benchmark/update_asr_leaderboard_jp.py
```

## Sources

- `benchmark/results_cohere-transcribe-14-lang-int8_adlib.json`
- `benchmark/results_cohere-transcribe-14-lang-int8_jsut.json`
- `benchmark/results_cohere-transcribe-14-lang-int8_jvnv.json`
- `benchmark/results_cohere-transcribe-14-lang-int8_jvs.json`
- `benchmark/results_parakeet-ctc-ja-int8_adlib.json`
- `benchmark/results_parakeet-ctc-ja-int8_jsut.json`
- `benchmark/results_parakeet-ctc-ja-int8_jvnv.json`
- `benchmark/results_parakeet-ctc-ja-int8_jvs.json`
- `benchmark/results_reazonspeech-ja-en-mls-5k_adlib.json`
- `benchmark/results_reazonspeech-ja-en-mls-5k_jsut.json`
- `benchmark/results_reazonspeech-ja-en-mls-5k_jvnv.json`
- `benchmark/results_reazonspeech-ja-en-mls-5k_jvs.json`
- `benchmark/results_reazonspeech-ja-en_adlib.json`
- `benchmark/results_reazonspeech-ja-en_jsut.json`
- `benchmark/results_reazonspeech-ja-en_jvnv.json`
- `benchmark/results_reazonspeech-ja-en_jvs.json`
- `benchmark/results_reazonspeech-ja_jsut.json`
- `benchmark/results_reazonspeech-ja_jvs.json`
- `benchmark/results_reazonspeech_ja_adlib.json`
- `benchmark/results_reazonspeech_ja_jvnv.json`
- `benchmark/results_sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17_adlib.json`
- `benchmark/results_sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17_jsut.json`
- `benchmark/results_sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17_jvnv.json`
- `benchmark/results_sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17_jvs.json`
- `benchmark/results_sherpa-onnx-whisper-distil-large-v3.5_adlib.json`
- `benchmark/results_sherpa-onnx-whisper-distil-large-v3.5_jvnv.json`
- `benchmark/results_sherpa-onnx-whisper-large-v3_jsut.json`
- `benchmark/results_sherpa-onnx-whisper-turbo_adlib.json`
- `benchmark/results_sherpa-onnx-whisper-turbo_jsut.json`
- `benchmark/results_sherpa-onnx-whisper-turbo_jvnv.json`
- `benchmark/results_sherpa-onnx-whisper-turbo_jvs.json`
- `benchmark/results_whisper_large_v3_adlib.json`
- `benchmark/results_whisper_large_v3_jvnv.json`
