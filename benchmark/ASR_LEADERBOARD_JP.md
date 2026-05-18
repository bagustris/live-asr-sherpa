# Japanese ASR Leaderboard

HF Open ASR Leaderboard–style ranking of Japanese ASR models on [adlib-devterm](https://huggingface.co/datasets/holotherapper/adlib-devterm) and [JVNV v1](https://ss-takashi.sakura.ne.jp/corpus/jvnv/). Lower is better for every column.

Numbers come from `benchmark/benchmark_ja.py` and `benchmark/benchmark_jvnv.py` (offline pipeline, 4 threads, `--language ja`). Regenerate this file with `python benchmark/update_asr_leaderboard_jp.py`.

| Model | Avg CER | Avg RTF | Avg Composite | ADLIB-DEVTERM CER | JVNV CER |
|-------|--------:|--------:|--------------:|------------------:|---------:|
| ReazonSpeech JA | 28.95% | 0.0357 | 0.1626 | 46.09% | 11.81% |
| Parakeet CTC JA (int8) | 32.31% | 0.0316 | 0.1677 | 49.41% | 15.22% |

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

- `benchmark/results_parakeet_ctc_ja.json`
- `benchmark/results_reazonspeech_ja_adlib.json`
- `benchmark/results_reazonspeech_ja_jvnv.json`
- `results_jvnv.json`
