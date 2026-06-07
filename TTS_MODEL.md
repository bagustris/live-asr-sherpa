# Supported TTS Models

`sherox.tts` includes a small built-in registry of language codes and backends. Models are auto-resolved from `--lang` and downloaded on first use when the backend is `sherpa-onnx`.

## Built-In Languages

| `--lang` | Backend | Model / Voice | Sample Rate | Notes |
|---------|---------|---------------|-------------|-------|
| `eng` | `sherpa_onnx` | `vits-piper-en_US-amy-medium` | 22050 Hz | English US, Piper VITS, Amy, medium quality |
| `deu` | `sherpa_onnx` | `vits-piper-de_DE-thorsten-medium` | 22050 Hz | German, Piper VITS, Thorsten, medium quality |
| `fra` | `sherpa_onnx` | `vits-piper-fr_FR-upmc-medium` | 22050 Hz | French, Piper VITS, UPMC, medium quality |
| `spa` | `sherpa_onnx` | `vits-piper-es_ES-mls_10246-medium` | 22050 Hz | Spanish, Piper VITS, MLS, medium quality |
| `ind` | `sherpa_onnx` | `vits-piper-id_ID-news_tts-medium` | 22050 Hz | Indonesian, Piper VITS, medium quality |
| `jpn` | `piper_plus` | `ja_JP-tsukuyomi-chan-medium` | 22050 Hz | Japanese via Piper Plus Tsukuyomi |
| `jpn-sarashina` | `sarashina` | `sbintuitions/Sarashina-TTS` | 24000 Hz | Japanese Sarashina2.2-TTS with zero-shot voice cloning support |

## Auto-Downloaded Sherpa-ONNX TTS Models

These are downloaded into `models/<model-dir>/` on first use.

| `--lang` | Archive | Extracted Directory | Main Model File | Extra Files |
|---------|---------|---------------------|-----------------|------------|
| `eng` | `vits-piper-en_US-amy-medium.tar.bz2` | `vits-piper-en_US-amy-medium` | `en_US-amy-medium.onnx` | `tokens.txt`, `espeak-ng-data/` |
| `deu` | `vits-piper-de_DE-thorsten-medium.tar.bz2` | `vits-piper-de_DE-thorsten-medium` | `de_DE-thorsten-medium.onnx` | `tokens.txt`, `espeak-ng-data/` |
| `fra` | `vits-piper-fr_FR-upmc-medium.tar.bz2` | `vits-piper-fr_FR-upmc-medium` | `fr_FR-upmc-medium.onnx` | `tokens.txt`, `espeak-ng-data/` |
| `spa` | `vits-piper-es_ES-mls_10246-medium.tar.bz2` | `vits-piper-es_ES-mls_10246-medium` | `es_ES-mls_10246-medium.onnx` | `tokens.txt`, `espeak-ng-data/` |
| `ind` | `vits-piper-id_ID-news_tts-medium.tar.bz2` | `vits-piper-id_ID-news_tts-medium` | `id_ID-news_tts-medium.onnx` | `tokens.txt`, `espeak-ng-data/` |

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

### Sarashina

`jpn-sarashina` uses the `sarashina-tts` backend and supports optional voice cloning through `--audio-prompt` and `--audio-prompt-text`.

Example:

```bash
sherox.tts --text "こんにちは。" --lang jpn-sarashina \
  --audio-prompt prompt.wav --audio-prompt-text "プロンプトの文章。"
```

## Language Aliases

These aliases are normalized internally before lookup:

- `id` -> `ind`
- `id-id` -> `ind`
- `ja` -> `jpn`
- `jp` -> `jpn`
- `ja-jp` -> `jpn`
- `sarashina` -> `jpn-sarashina`
- `jpn_sarashina` -> `jpn-sarashina`
