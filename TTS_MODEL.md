# Supported TTS Models

`sherox.tts` includes a small built-in registry of language codes and backends. Models are auto-resolved from `--lang` and downloaded on first use when the backend is `sherpa-onnx`-based (`sherpa_onnx`, `kitten`, or `supertonic`).

## Built-In Languages (dedicated models)

| `--lang` | Backend | Model / Voice | Sample Rate | Notes |
|---------|---------|---------------|-------------|-------|
| `eng` | `sherpa_onnx` | `vits-piper-en_US-amy-medium` | 22050 Hz | English US, Piper VITS, Amy, medium quality |
| `eng-kitten` | `kitten` | `kitten-nano-en-v0_8-int8` | 24000 Hz | English, Kitten TTS Nano v0.8, quantized, 8 speakers |
| `deu` | `sherpa_onnx` | `vits-piper-de_DE-thorsten-medium` | 22050 Hz | German, Piper VITS, Thorsten, medium quality |
| `fra` | `sherpa_onnx` | `vits-piper-fr_FR-upmc-medium` | 22050 Hz | French, Piper VITS, UPMC, medium quality |
| `spa` | `sherpa_onnx` | `vits-piper-es_ES-mls_10246-medium` | 22050 Hz | Spanish, Piper VITS, MLS, medium quality |
| `ind` | `sherpa_onnx` | `vits-piper-id_ID-news_tts-medium` | 22050 Hz | Indonesian, Piper VITS, medium quality |
| `zho` | `sherpa_onnx` | `vits-icefall-zh-aishell3` | 8000 Hz | Chinese Mandarin, VITS, AiShell3, 174 speakers |
| `jpn` | `piper_plus` | `ja_JP-tsukuyomi-chan-medium` | 22050 Hz | Japanese via Piper Plus Tsukuyomi |

## Supertonic-3 (shared multi-language model)

A single shared model (`sherpa-onnx-supertonic-3-tts-int8-2026-05-11`, ~120 MB, 24000 Hz, 10 speakers) covers 25 additional languages plus an alternate Indonesian voice. Select a speaker with `--speaker-id 0-9`.

| `--lang` | ISO 639-1 | Description |
|---------|-----------|--------------|
| `kor` | `ko` | Korean |
| `ara` | `ar` | Arabic |
| `bul` | `bg` | Bulgarian |
| `ces` | `cs` | Czech |
| `dan` | `da` | Danish |
| `ell` | `el` | Greek |
| `est` | `et` | Estonian |
| `fin` | `fi` | Finnish |
| `hin` | `hi` | Hindi |
| `hrv` | `hr` | Croatian |
| `hun` | `hu` | Hungarian |
| `ita` | `it` | Italian |
| `lit` | `lt` | Lithuanian |
| `lav` | `lv` | Latvian |
| `nld` | `nl` | Dutch |
| `pol` | `pl` | Polish |
| `por` | `pt` | Portuguese |
| `ron` | `ro` | Romanian |
| `rus` | `ru` | Russian |
| `slk` | `sk` | Slovak |
| `slv` | `sl` | Slovenian |
| `swe` | `sv` | Swedish |
| `tur` | `tr` | Turkish |
| `ukr` | `uk` | Ukrainian |
| `vie` | `vi` | Vietnamese |
| `ind-supertonic` | `id` | Indonesian (alternative to the dedicated `ind` Piper model) |

Languages that already have a dedicated model above (`eng`, `deu`, `fra`, `spa`, `ind`, `zho`, `jpn`) keep their existing default and are **not** routed through Supertonic-3.

Example:

```bash
sherox.tts --text "Привет, как дела?" --lang rus --speaker-id 3
```

## Auto-Downloaded Models

These are downloaded automatically on first use.

### VITS (Piper / Icefall) — `models/<model-dir>/`

| `--lang` | Archive | Extracted Directory | Main Model File | Extra Files |
|---------|---------|---------------------|-----------------|------------|
| `eng` | `vits-piper-en_US-amy-medium.tar.bz2` | `vits-piper-en_US-amy-medium` | `en_US-amy-medium.onnx` | `tokens.txt`, `espeak-ng-data/` |
| `deu` | `vits-piper-de_DE-thorsten-medium.tar.bz2` | `vits-piper-de_DE-thorsten-medium` | `de_DE-thorsten-medium.onnx` | `tokens.txt`, `espeak-ng-data/` |
| `fra` | `vits-piper-fr_FR-upmc-medium.tar.bz2` | `vits-piper-fr_FR-upmc-medium` | `fr_FR-upmc-medium.onnx` | `tokens.txt`, `espeak-ng-data/` |
| `spa` | `vits-piper-es_ES-mls_10246-medium.tar.bz2` | `vits-piper-es_ES-mls_10246-medium` | `es_ES-mls_10246-medium.onnx` | `tokens.txt`, `espeak-ng-data/` |
| `ind` | `vits-piper-id_ID-news_tts-medium.tar.bz2` | `vits-piper-id_ID-news_tts-medium` | `id_ID-news_tts-medium.onnx` | `tokens.txt`, `espeak-ng-data/` |
| `zho` | `vits-icefall-zh-aishell3.tar.bz2` | `vits-icefall-zh-aishell3` | `model.onnx` | `tokens.txt`, `lexicon.txt` (no `espeak-ng-data/`) |

### Kitten TTS — `models/kitten/`

| `--lang` | Archive | Extracted Directory | Main Model File | Extra Files |
|---------|---------|---------------------|-----------------|------------|
| `eng-kitten` | `kitten-nano-en-v0_8-int8.tar.bz2` | `kitten-nano-en-v0_8-int8` | `model.int8.onnx` | `voices.bin`, `tokens.txt`, `espeak-ng-data/` |

### Supertonic-3 — `models/supertonic/`

One shared archive backs all 25 languages listed above plus `ind-supertonic`.

| Archive | Extracted Directory | Files |
|---------|---------------------|-------|
| `sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2` | `sherpa-onnx-supertonic-3-tts-int8-2026-05-11` | `duration_predictor.int8.onnx`, `text_encoder.int8.onnx`, `vector_estimator.int8.onnx`, `vocoder.int8.onnx`, `tts.json`, `unicode_indexer.bin`, `voice.bin` |

## Non-Sherpa Backends

### Piper Plus

`jpn` uses the `piper-plus` backend rather than a model directory managed by this repo.

Requirements:

```bash
pip install -e '.[tts-ja]'
```

Example:

```bash
sherox.tts --text "こんにちは、今日は良い天気ですね。" --lang jpn
```

## Playback and Saving

By default, `sherox.tts` writes `output.wav`. Add `--play` to play the synthesized audio after saving it:

```bash
sherox.tts --text "Halo dunia" --play
```

For playback-only output, use either `--play --no-save` or `--play --output none`:

```bash
sherox.tts --text "Halo dunia" --play --no-save
sherox.tts --text "Halo dunia" --play --output none
```

## Language Aliases

These aliases are normalized internally before lookup:

- `en` / `en-us` / `en-gb` -> `eng`
- `id` / `id-id` -> `ind`
- `ja` / `jp` / `ja-jp` -> `jpn`
- `zh` / `zh-cn` / `zh-tw` / `cmn` / `chi` -> `zho`
- `ko` -> `kor`
- `ar` -> `ara`
- `bg` -> `bul`
- `cs` -> `ces`
- `da` -> `dan`
- `de` / `ger` / `de-de` -> `deu`
- `el` -> `ell`
- `es` / `es-es` -> `spa`
- `et` -> `est`
- `fi` -> `fin`
- `fr` / `fre` / `fr-fr` -> `fra`
- `hi` -> `hin`
- `hr` -> `hrv`
- `hu` -> `hun`
- `it` -> `ita`
- `lt` -> `lit`
- `lv` -> `lav`
- `nl` / `dut` -> `nld`
- `pl` -> `pol`
- `pt` -> `por`
- `ro` / `rum` -> `ron`
- `ru` -> `rus`
- `sk` -> `slk`
- `sl` -> `slv`
- `sv` -> `swe`
- `tr` -> `tur`
- `uk` -> `ukr`
- `vi` -> `vie`

Notes:
- Chinese (`zho`): input must be plain Simplified Chinese text; numbers and mixed scripts may not normalize well.
- `eng-kitten` is a quantized model intended for fast, low-resource synthesis; quality is lower than the medium-quality Piper voices.
