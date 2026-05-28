# Benchmark

Tools for evaluating live-asr-sherpa models on accuracy, speed, and composite score.

## Contents

| Script | Dataset | Language | Primary metric |
|--------|---------|----------|----------------|
| [`benchmark.py`](#benchmarkpy--english-librispeech) | LibriSpeech or custom manifest | Any | WER (or CER for `--language ja`) |
| [`benchmark_ja.py`](#benchmark_japy--japanese-adlib-devterm) | [adlib-devterm](https://huggingface.co/datasets/holotherapper/adlib-devterm) (HuggingFace) | Japanese | KER / CER |
| [`benchmark_jvnv.py`](#benchmark_jvnvpy--japanese-jvnv-emotional-speech) | [JVNV v1](https://ss-takashi.sakura.ne.jp/corpus/jvnv/) (local) | Japanese | KER / CER |
| [`benchmark_jsut.py`](#benchmark_jsutpy--japanese-jsut-basic5000) | [JSUT Basic5000](https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000) (HuggingFace) | Japanese | KER / CER |
| [`benchmark_jvs.py`](#benchmark_jvspy--japanese-jvs-corpus) | [JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) (local) | Japanese | KER / CER |
| [`bench_resample.py`](#bench_resamplepy--resampler-quality) | Synthetic signal | — | SNR / passband ripple |
| [`update_asr_leaderboard_jp.py`](#updating-the-japanese-asr-leaderboard) | — | Japanese | Regenerates [`ASR_LEADERBOARD_JP.md`](ASR_LEADERBOARD_JP.md) |

## Install

```bash
# From the repo root
pip install -e '.[dev]'
pip install -r benchmark/requirements.txt
```

The `requirements.txt` adds:

| Package | Purpose |
|---------|---------|
| `editdistance` | Fast Levenshtein (WER/CER/KER) |
| `soundfile` | WAV/FLAC decoding |
| `datasets` | HuggingFace dataset streaming (`benchmark_ja.py`, `benchmark_jsut.py`) |
| `numpy` | Array math, bootstrap CI |
| `soxr` | High-quality resampler (optional; scipy fallback used if absent) |
| `pyopenjtalk` | G2P for accurate KER (optional; character-level fallback used if absent) |

---

## Metrics

All benchmarks report the same core metrics so results are comparable across scripts:

| Metric | Formula | Direction |
|--------|---------|-----------|
| **CER** | `char_edit_distance / ref_chars` | lower is better |
| **KER** | `kana_edit_distance / ref_kana_chars` | lower is better |
| **WER** | `word_edit_distance / ref_words` | lower is better |
| **RTF** | `processing_time / audio_duration` | lower is better; RTF < 1 = faster than real-time |
| **Latency** | `processing_time × 1000 ms` | lower is better |
| **Composite** | `(KER + mean_RTF) / 2` | lower is better |

**CER normalization** (Japanese benchmarks): NFKC → lowercase → strip Japanese punctuation (`。、！？・「」` etc.) → strip whitespace. This measures what was *said*, independent of notation variants and punctuation style.

**KER (Kana Error Rate)** is the primary metric for all Japanese benchmarks. Both reference and hypothesis are first converted to hiragana (via `pyopenjtalk` G2P when available, otherwise katakana→hiragana shift + drop non-kana), then character-level edit distance is computed on the hiragana sequences. This makes KER phonetically robust: kanji and kana representations of the same word (e.g. `東京` vs `とうきょう`) are counted as equivalent. KER is implemented in [`kana_utils.py`](kana_utils.py) and re-exported via `benchmark_utils.py`.

**Composite score** balances transcription accuracy against processing speed in a single number. Use it to rank models when both quality and latency matter.

**Bootstrap CI**: 95% confidence intervals use ratio-based bootstrap (B = 10,000, seed = 42).

---

## benchmark.py — English / LibriSpeech

Evaluates any live-asr-sherpa model on LibriSpeech or a custom audio manifest. Reports WER (CER when `--language ja`), RTF, latency, and composite score.

### Data

**LibriSpeech** — download a split such as `dev-clean-2` from the [OpenSLR page](https://www.openslr.org/12) and extract it:

```
LibriSpeech/
  dev-clean-2/
    103/
      1240/
        103-1240-0000.flac
        103-1240-0000.normalized.txt
```

Pass the split root via `--data-dir`. The script auto-discovers `.flac` files and their matching `.normalized.txt` transcriptions.

**Custom manifest** — instead of `--data-dir`, pass a TSV or CSV with two columns (no header):

```
/path/to/audio.wav    reference text here
```

Supported delimiters: tab, comma, pipe.

### Usage

```bash
# Offline Parakeet TDT (default) on LibriSpeech dev-clean-2
python benchmark/benchmark.py \
  --data-dir /data/LibriSpeech/dev-clean-2 \
  --offline

# Online Zipformer streaming
python benchmark/benchmark.py \
  --data-dir /data/LibriSpeech/dev-clean-2

# Whisper small offline
python benchmark/benchmark.py \
  --data-dir /data/LibriSpeech/dev-clean-2 \
  --offline --model-type whisper \
  --model-dir models/sherpa-onnx-whisper-small.en

# Smoke test (first 20 utterances)
python benchmark/benchmark.py \
  --data-dir /data/LibriSpeech/dev-clean-2 \
  --offline --max-utts 20 --verbose

# Custom manifest, save results
python benchmark/benchmark.py \
  --manifest my_audio.tsv \
  --offline --output benchmark/results_custom.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir PATH` | `/data/LibriSpeech/dev-clean-2` | LibriSpeech split root |
| `--manifest FILE` | — | Custom TSV/CSV manifest (audio path + transcript) |
| `--model-dir PATH` | `models/parakeet-tdt-0.6b-v2-int8` | Model directory |
| `--model-type TYPE` | *(auto)* | Architecture hint; see `ASR_MODEL.md` |
| `--offline` | streaming | Use offline (VAD-segmented) pipeline |
| `--language LANG` | `en` | Language code for Whisper/SenseVoice |
| `--threads N` | `4` | ONNX runtime thread count |
| `--sample-rate HZ` | `16000` | Target sample rate |
| `--chunk-size SEC` | `0.1` | Streaming chunk size (online mode) |
| `--max-utts N` | all | Limit to first N utterances |
| `--verbose` | off | Print REF/HYP per utterance |
| `--output FILE` | — | Save full results to JSON |

---

## benchmark_ja.py — Japanese / adlib-devterm

Evaluates Japanese ASR on the [adlib-devterm](https://huggingface.co/datasets/holotherapper/adlib-devterm) dataset — 247 IT-domain utterances with technical term annotations (3 speakers, 6 categories). Follows the [adlib evaluation protocol](https://github.com/holotherapper/adlib).

Primary metric: **KER** (Kana Error Rate). CER is also reported.

### Data

The dataset is streamed from HuggingFace on first run — no manual download needed.

The adlib term-annotation file (`devterm_test_cases.jsonl`) is auto-downloaded from GitHub and cached in `benchmark/` on first run.

### Usage

```bash
# Default model (parakeet-ctc-ja, auto-downloaded)
python benchmark/benchmark_ja.py --offline

# Smoke test: first 10 utterances, show REF/HYP
python benchmark/benchmark_ja.py --offline --max-utts 10 --verbose

# Filter by category
python benchmark/benchmark_ja.py --offline --category backend

# Enable term accuracy reporting (slower)
python benchmark/benchmark_ja.py --offline --term-acc

# ReazonSpeech model
python benchmark/benchmark_ja.py --offline --model-type reazonspeech-ja

# Save results to JSON
python benchmark/benchmark_ja.py --offline --output benchmark/results_ja.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir PATH` | `models/parakeet-ctc-ja-int8` | Model directory |
| `--model-type TYPE` | `parakeet-ctc-ja` | See table below |
| `--offline` | streaming | Use offline pipeline |
| `--category CAT` | all | Filter: `backend`, `cli`, `concept`, `frontend`, `infra`, `mixed` |
| `--speaker SPK` | all | Filter: `spk-01`, `spk-02`, `spk-03` |
| `--term-acc` | off | Compute Term Accuracy and Adlib composite score |
| `--language LANG` | `ja` | Language code |
| `--threads N` | `4` | ONNX runtime thread count |
| `--max-utts N` | all | Limit to first N utterances |
| `--verbose` | off | Print REF/HYP per utterance |
| `--output FILE` | — | Save full results to JSON |
| `--adlib-cases FILE` | *(auto)* | Path to cached `devterm_test_cases.jsonl` |

### Supported model types

| `--model-type` | Engine mapping | Notes |
|----------------|----------------|-------|
| `parakeet-ctc-ja` | `nemo_ctc` | Default; auto-downloaded |
| `reazonspeech-ja` | transducer | Auto-downloaded |
| `reazonspeech-ja-en` | transducer | Bilingual; auto-downloaded |
| `reazonspeech-ja-en-mls-5k` | transducer | Bilingual + MLS; auto-downloaded |

### Extra metrics (`--term-acc`)

When enabled, also reports:

| Metric | Description |
|--------|-------------|
| **Term Accuracy** | % of annotated IT terms correctly present in hypothesis |
| **Exact Term Acc** | Exact-match terms only |
| **Flexible Term Acc** | Flexible terms (katakana ↔ English equivalents accepted) |
| **Adlib Score** | `0.4 × (1 − CER) + 0.6 × TermAcc` — higher is better |

---

## benchmark_jvnv.py — Japanese / JVNV Emotional Speech

Evaluates Japanese ASR on the [JVNV v1](https://arxiv.org/abs/2310.06072) corpus — 1,615 utterances of expressive speech with embedded nonverbal vocalizations (laughter, sobbing, etc.). Tests robustness to emotional speech and non-standard prosody.

Primary metric: **KER** (Kana Error Rate). CER is also reported.

- **Speakers**: F1, F2 (female); M1, M2 (male)
- **Emotions**: anger, disgust, fear, happy, sad, surprise
- **Sessions**: `regular` (NV phrase designated) / `free` (speaker-chosen)
- **Audio**: 48 kHz mono WAV, resampled to 16 kHz for ASR

### Data

Download JVNV v1 from the [official distribution page](https://ss-takashi.sakura.ne.jp/corpus/jvnv/) and extract it. The expected directory structure:

```
jvnv_v1/
  transcription.csv
  F1/
    anger/
      regular/  F1_anger_regular_01.wav ...
      free/     F1_anger_free_01.wav ...
    disgust/ fear/ happy/ sad/ surprise/
  F2/  M1/  M2/
```

### Usage

```bash
# Full benchmark (all 1,615 utterances)
python benchmark/benchmark_jvnv.py --jvnv-dir /data/jvnv_v1 --offline

# Smoke test: first 20 utterances, verbose
python benchmark/benchmark_jvnv.py --jvnv-dir /data/jvnv_v1 --offline \
  --max-utts 20 --verbose

# Filter by emotion
python benchmark/benchmark_jvnv.py --jvnv-dir /data/jvnv_v1 --offline \
  --emotion happy

# Filter by speaker and session
python benchmark/benchmark_jvnv.py --jvnv-dir /data/jvnv_v1 --offline \
  --speaker F1 --session regular

# ReazonSpeech model
python benchmark/benchmark_jvnv.py --jvnv-dir /data/jvnv_v1 --offline \
  --model-type reazonspeech-ja

# Save results to JSON
python benchmark/benchmark_jvnv.py --jvnv-dir /data/jvnv_v1 --offline \
  --output benchmark/results_jvnv.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--jvnv-dir PATH` | `/data/jvnv_v1` | Root of the extracted JVNV corpus |
| `--model-dir PATH` | `models/parakeet-ctc-ja-int8` | Model directory |
| `--model-type TYPE` | `parakeet-ctc-ja` | Same aliases as `benchmark_ja.py` |
| `--offline` | streaming | Use offline pipeline |
| `--speaker SPK` | all | Filter: `F1`, `F2`, `M1`, `M2` |
| `--emotion EMO` | all | Filter: `anger`, `disgust`, `fear`, `happy`, `sad`, `surprise` |
| `--session SES` | all | Filter: `regular`, `free` |
| `--language LANG` | `ja` | Language code |
| `--threads N` | `4` | ONNX runtime thread count |
| `--max-utts N` | all | Limit to first N utterances |
| `--verbose` | off | Print REF/HYP per utterance |
| `--output FILE` | — | Save full results to JSON |

### Output breakdown

The summary table shows CER and KER broken down along three dimensions:

```
── CER/KER by Speaker ──
  Group                    N      CER      KER
  -----------------------------------------------
  F1                     356   xx.xx%   xx.xx%
  F2                     477   xx.xx%   xx.xx%
  M1                     356   xx.xx%   xx.xx%
  M2                     426   xx.xx%   xx.xx%

── CER/KER by Emotion ──
  Group                    N      CER      KER
  -----------------------------------------------
  anger                  xxx   xx.xx%   xx.xx%
  ...

── CER/KER by Session ──
  Group                    N      CER      KER
  -----------------------------------------------
  free                   240   xx.xx%   xx.xx%
  regular               1375   xx.xx%   xx.xx%
```

---

## benchmark_jsut.py — Japanese / JSUT Basic5000

Evaluates Japanese ASR on the [JSUT Basic5000](https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000) dataset — 5,000 utterances from a single female speaker covering a broad range of Japanese sentence patterns. The dataset is streamed from HuggingFace.

Primary metric: **KER** (Kana Error Rate). CER is also reported.

### Data

Streamed automatically from HuggingFace on first run — no manual download needed.

### Usage

```bash
# Full benchmark with default model
python benchmark/benchmark_jsut.py --offline

# Smoke test: first 20 utterances, verbose
python benchmark/benchmark_jsut.py --offline --max-utts 20 --verbose

# ReazonSpeech model
python benchmark/benchmark_jsut.py --offline --model-type reazonspeech-ja

# Save results to JSON
python benchmark/benchmark_jsut.py --offline --output benchmark/results_jsut.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset REPO` | `japanese-asr/ja_asr.jsut_basic5000` | HuggingFace dataset repository |
| `--split SPLIT` | `test` | Dataset split |
| `--model-dir PATH` | `models/parakeet-ctc-ja-int8` | Model directory |
| `--model-type TYPE` | `parakeet-ctc-ja` | See supported types in `benchmark_ja.py` |
| `--offline` | streaming | Use offline pipeline |
| `--language LANG` | `ja` | Language code |
| `--threads N` | `4` | ONNX runtime thread count |
| `--sample-rate HZ` | `16000` | Target sample rate |
| `--chunk-size SEC` | `0.1` | Streaming chunk size (online mode) |
| `--max-utts N` | all | Limit to first N utterances |
| `--verbose` | off | Print REF/HYP per utterance |
| `--output FILE` | — | Save full results to JSON |

---

## benchmark_jvs.py — Japanese / JVS Corpus

Evaluates Japanese ASR on the [JVS (Japanese Versatile Speech) corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) — 100 speakers (50 male, 50 female) each reading 100 sentences (`parallel100`), plus nonparallel, whispered, and falsetto subsets (~30 hours total). Supports per-speaker, subset, and gender filtering.

Primary metric: **KER** (Kana Error Rate). CER is also reported.

- **Speakers**: `jvs001`–`jvs100` (female: `jvs001`–`jvs050`; male: `jvs051`–`jvs100`)
- **Subsets**: `parallel100`, `nonpara30`, `whisper10`, `falsetto10`
- **Audio**: 24 kHz mono WAV, resampled to 16 kHz for ASR

### Data

Download the JVS corpus from the [official page](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) and extract it. Expected directory structure:

```
jvs_ver1/
  jvs001/
    parallel100/
      wav24kHz16bit/
        VOICEACTRESS100_001.wav  ...
      transcripts_utf8.txt
    nonpara30/
      wav24kHz16bit/  ...
      transcripts_utf8.txt
    whisper10/  falsetto10/  ...
  jvs002/  ...
```

Transcripts can use either colon-delimited (`VOICEACTRESS100_001:text`) or whitespace-delimited (`VOICEACTRESS100_001 text`) format.

### Usage

```bash
# Full benchmark, all 100 speakers, parallel100 subset
python benchmark/benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline

# Smoke test: first 20 utterances, verbose
python benchmark/benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline \
  --max-utts 20 --verbose

# Filter by gender
python benchmark/benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline --gender female

# Filter by specific speakers
python benchmark/benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline \
  --speaker jvs001 jvs002

# Evaluate a specific subset
python benchmark/benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline \
  --subset nonpara30

# ReazonSpeech model
python benchmark/benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline \
  --model-type reazonspeech-ja

# Save results to JSON
python benchmark/benchmark_jvs.py --jvs-dir /data/jvs_ver1 --offline \
  --output benchmark/results_jvs.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--jvs-dir PATH` | `/data/jvs_ver1` | Root of the extracted JVS corpus |
| `--subset SUBSET [...]` | `parallel100` | Subset(s): `parallel100`, `nonpara30`, `whisper10`, `falsetto10` |
| `--speaker SPK [...]` | all | Restrict to specific speaker IDs (e.g. `jvs001 jvs002`) |
| `--gender {male,female}` | all | Restrict to male (`jvs051`–`jvs100`) or female (`jvs001`–`jvs050`) speakers |
| `--model-dir PATH` | `models/parakeet-ctc-ja-int8` | Model directory |
| `--model-type TYPE` | `parakeet-ctc-ja` | See supported types in `benchmark_ja.py` |
| `--offline` | streaming | Use offline pipeline |
| `--language LANG` | `ja` | Language code |
| `--threads N` | `4` | ONNX runtime thread count |
| `--sample-rate HZ` | `16000` | Target sample rate |
| `--chunk-size SEC` | `0.1` | Streaming chunk size (online mode) |
| `--max-utts N` | all | Limit to first N utterances |
| `--verbose` | off | Print REF/HYP per utterance |
| `--output FILE` | — | Save full results to JSON |

---

## bench_resample.py — Resampler Quality

Not an ASR benchmark. Compares resampler implementations on signal quality and speed when downsampling from 48 kHz (JVNV native rate) to 16 kHz (ASR input rate).

```bash
python benchmark/bench_resample.py
```

Reports SNR (band-limited), edge SNR, passband ripple, and processing time for each resampler (soxr HQ/VHQ, scipy polyphase with different Kaiser windows, scipy FFT).

---

## Updating the Japanese ASR leaderboard

[`ASR_LEADERBOARD_JP.md`](ASR_LEADERBOARD_JP.md) is **derived** from the result JSON files
produced by `benchmark_ja.py`, `benchmark_jvnv.py`, `benchmark_jsut.py`, and `benchmark_jvs.py`.
Do not edit it by hand — it gets regenerated.

The leaderboard ranks models by **average KER** across all available datasets. CER, RTF, and per-dataset columns are also shown.

### Workflow

1. Run the Japanese benchmarks for the model under test, passing `--output` so the
   aggregate metrics land in a JSON file:

    ```bash
    MODEL=parakeet-ctc-ja   # or whisper, sense_voice, reazonspeech-ja, ...

    python benchmark/benchmark_ja.py --offline --model-type "$MODEL" \
      --output "benchmark/results_${MODEL}_adlib.json"

    python benchmark/benchmark_jvnv.py --offline --jvnv-dir /data/jvnv_v1 \
      --model-type "$MODEL" \
      --output "benchmark/results_${MODEL}_jvnv.json"

    python benchmark/benchmark_jsut.py --offline --model-type "$MODEL" \
      --output "benchmark/results_${MODEL}_jsut.json"

    python benchmark/benchmark_jvs.py --offline --jvs-dir /data/jvs_ver1 \
      --model-type "$MODEL" \
      --output "benchmark/results_${MODEL}_jvs.json"
    ```

2. Regenerate the leaderboard:

    ```bash
    python benchmark/update_asr_leaderboard_jp.py
    ```

The script scans `benchmark/*.json` and `results*.json` in the repo root, groups results
by `model_dir`, computes per-model averages across all datasets, and rewrites
`benchmark/ASR_LEADERBOARD_JP.md` with rows sorted by Avg KER ascending.

### How models are grouped

- **Grouping key:** the basename of the JSON's `model_dir` field (e.g.
  `parakeet-ctc-ja-int8`). Filenames are ignored — two JSONs with different names but the
  same `model_dir` merge into a single row.
- **Dataset key:** the `dataset` field inside the JSON (`holotherapper/adlib-devterm`,
  `JVNV`, `japanese-asr/ja_asr.jsut_basic5000`, or `JVS`). Other datasets are skipped.
- **Display name:** mapped via the `MODEL_LABELS` dict at the top of
  [`update_asr_leaderboard_jp.py`](update_asr_leaderboard_jp.py); falls back to the
  `model_dir` basename for unknown models. To add a friendlier name for a new model, add
  one entry to that dict and re-run the script.

### Adding a new dataset

1. Add the dataset's `"dataset"` field value → display label in the `DATASETS` dict in
   `update_asr_leaderboard_jp.py`.
2. Append the display label to `DATASET_ORDER`.
3. Re-run the script — the new column appears automatically.

## Comparing results across benchmarks

All Japanese ASR benchmarks use the same KER/CER normalization pipeline and composite score formula, so their numbers are directly comparable.

| Benchmark | Domain | Style | Challenge |
|-----------|--------|-------|-----------|
| adlib-devterm | IT / tech terms | Read speech | Katakana/English term accuracy |
| JVNV | General / emotional | Expressive + nonverbal | Prosodic variation, NV sounds |
| JSUT Basic5000 | General | Clean read speech | Standard Japanese read-speech baseline |
| JVS | General | Read speech, whispered, falsetto | Speaker diversity, speech style variation |
| LibriSpeech | Audiobook | Clean read speech | English baseline accuracy |

A model with low KER on JSUT/LibriSpeech but high KER on JVNV likely struggles with non-neutral prosody. High KER on adlib despite low JVNV KER suggests poor IT-domain vocabulary coverage. The `--gender`, `--speaker`, and `--subset` filters in `benchmark_jvs.py` allow fine-grained analysis of performance across speaker demographics and speech styles.

Use `--output FILE` on each benchmark to save JSON results, then compare `aggregate.ker`, `aggregate.cer`, and `aggregate.composite_score` across runs.
