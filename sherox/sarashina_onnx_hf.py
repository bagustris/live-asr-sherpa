"""Package the exported Sarashina ONNX artifacts into a Hugging Face upload directory.

Produces a self-contained folder with the ONNX models, the mandatory license
files, and a model card (README.md), laid out so it can be pushed straight to a
Hugging Face model repo::

    Sarashina2.2-TTS-ONNX/
    ├── README.md            model card (YAML frontmatter + docs)
    ├── LICENSE              verbatim Sarashina Model NonCommercial License
    ├── NOTICE               required attribution notice (Sarashina + CosyVoice2)
    ├── meta.json            runtime constants
    ├── flow_encoder.onnx
    ├── flow_estimator.onnx
    ├── hift.onnx
    ├── campplus.onnx        speaker encoder, for zero-shot voice cloning
    ├── s3_tokenizer.onnx    semantic tokenizer, for zero-shot voice cloning
    ├── s3_mel_filters.npz   mel filterbank asset for the semantic tokenizer
    └── llm/                 onnxruntime-genai model (model.onnx + .data + tokenizer)

LICENSE COMPLIANCE — two different upstream licenses apply here:

1. Everything derived from the Sarashina checkpoint itself (the LLM, flow,
   HiFT, and CAMPPlus) is released under the *Sarashina Model NonCommercial
   License*, which requires that any redistributed derivative (this ONNX
   export is one) (a) ship a copy of the license, (b) carry the exact
   attribution notice, (c) keep a name beginning with "Sarashina", and (d)
   state "Built with Sarashina". This packager writes all of those; do not
   strip them.
2. s3_tokenizer.onnx is a *third-party* component from FunAudioLLM/CosyVoice2
   (Apache 2.0, https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) — separate
   from the Sarashina checkpoint and NOT covered by the NonCommercial terms.
   It's attributed in NOTICE and the model card.

Usage::

    python -m sherox.sarashina_onnx_hf \\
        --onnx-dir models/sarashina-onnx \\
        --checkpoint-dir models/sarashina \\
        --out-dir upload/Sarashina2.2-TTS-ONNX \\
        --repo-id <your-username>/Sarashina2.2-TTS-ONNX
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Redistribution requirement (b) of the Sarashina Model NonCommercial License.
_SARASHINA_ATTRIBUTION = (
    "Sarashina is licensed under the Sarashina Model NonCommercial License "
    "Agreement, Copyright ©SB Intuitions Corp. All Rights Reserved."
)
# s3_tokenizer.onnx is a third-party component, not part of the Sarashina
# checkpoint — attributed separately under its own (Apache 2.0) license.
_COSYVOICE2_ATTRIBUTION = (
    "s3_tokenizer.onnx is redistributed from FunAudioLLM/CosyVoice2-0.5B "
    "(https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B), licensed under "
    "Apache License 2.0, Copyright FunAudioLLM. Not covered by the Sarashina "
    "NonCommercial License above."
)
# default_prompt.wav is sourced from sbintuitions's own official demo samples,
# not a third-party project — same NonCommercial terms as the Sarashina model.
_DEFAULT_PROMPT_ATTRIBUTION = (
    "default_prompt.wav is redistributed from sbintuitions/sarashina2.2-tts's "
    "own official demo samples (samples/samples/prompt_C.wav), under the same "
    "Sarashina Model NonCommercial License above."
)
_ATTRIBUTION_NOTICE = "\n\n".join(
    [_SARASHINA_ATTRIBUTION, _COSYVOICE2_ATTRIBUTION, _DEFAULT_PROMPT_ATTRIBUTION]
)

_BASE_MODEL = "sbintuitions/sarashina2.2-tts"
_LICENSE_NAME = "sarashina-model-noncommercial-license"
_LICENSE_LINK = "https://huggingface.co/sbintuitions/sarashina2.2-tts/blob/main/LICENSE"

_ARTIFACTS = [
    "flow_encoder.onnx", "flow_estimator.onnx", "hift.onnx",
    "campplus.onnx", "s3_tokenizer.onnx", "s3_mel_filters.npz",
    "flow_rand_noise.npy", "default_prompt.wav", "meta.json",
]


def _model_card(repo_id: str, meta: dict) -> str:
    """Return the README.md model-card text (YAML frontmatter + body)."""
    model_name = repo_id.split("/")[-1]
    precision = meta.get("precision", "fp32")
    # int4 is a real precision-loss quantization; fp32/fp16 aren't "quantized"
    # in the sense that tag implies, so only set it when it's actually true.
    base_model_relation_line = "\nbase_model_relation: quantized" if precision == "int4" else ""
    precision_note = (
        "quantized to `int4`" if precision == "int4"
        else f"exported at `{precision}` (not quantized — full precision, chosen for "
        "content accuracy; see below)"
    )
    frontmatter = f"""---
license: other
license_name: {_LICENSE_NAME}
license_link: LICENSE
base_model: {_BASE_MODEL}{base_model_relation_line}
language:
- ja
- en
pipeline_tag: text-to-speech
library_name: onnx
tags:
- text-to-speech
- tts
- japanese-tts
- onnx
- onnxruntime
- sarashina
- voice-cloning
---"""

    body = f"""

# {model_name}

**Built with Sarashina.**

ONNX Runtime export of [`{_BASE_MODEL}`](https://huggingface.co/{_BASE_MODEL}),
a Japanese-centric zero-shot voice-cloning TTS model built on a 0.5B-parameter
Llama backbone plus a CosyVoice-style flow-matching decoder and HiFT vocoder.

This derivative runs the **entire inference pipeline in ONNX Runtime, including
zero-shot voice cloning** — no PyTorch, no `transformers`, no CUDA required —
making it light enough for CPU-only local and server deployment. The LLM stage
is {precision_note}. `int4` quantization was found to measurably degrade
content accuracy versus `fp32` on an otherwise identical pipeline — see
sherox's `sarashina_onnx_export.py` docstring for the measurements. (`fp16`
isn't an option here: onnxruntime-genai only supports it with CUDA/DML, not
the CPU execution provider this runtime targets.)

> ⚠️ **License: NonCommercial.** This is a derivative of a model released under
> the Sarashina Model NonCommercial License Agreement. Commercial use is **not**
> permitted. See [`LICENSE`](LICENSE) and the attribution in [`NOTICE`](NOTICE).
> `s3_tokenizer.onnx` is a third-party file under a separate Apache 2.0 license
> (see `NOTICE`) — not covered by the NonCommercial terms.

## Pipeline

```
text ─▶ LLM (onnxruntime-genai, {precision}) ─▶ semantic tokens
     ─▶ flow_encoder.onnx    ─▶ mu / mask / speaker / cond
     ─▶ flow_estimator.onnx  ─▶ (Euler ODE loop, {meta.get("n_timesteps", 10)} steps) ─▶ mel
     ─▶ hift.onnx            ─▶ waveform ({meta.get("sample_rate", 24000)} Hz)

--audio-prompt reference.wav (zero-shot voice cloning)
     ─▶ numpy DSP ─▶ s3_tokenizer.onnx  ─▶ semantic tokens (LLM few-shot prime)
                  └▶ campplus.onnx      ─▶ speaker embedding
```

`torch.stft` / `torch.istft` in the vocoder are replaced with equivalent
real-valued conv/matmul implementations, since the ONNX exporter cannot handle
complex tensors. The reference-audio feature extraction for voice cloning
(mel/fbank computation) is reimplemented in pure numpy. Outputs were validated
against the original PyTorch model on real speech: vocoder mean abs diff ~7e-4,
flow stages ~1e-6, speaker embedding cosine similarity ~0.998, semantic tokens
match exactly in the large majority of cases (a resampling-algorithm
difference can very occasionally flip one token to an acoustically adjacent
codebook entry).

## Files

| Path | Description |
|------|-------------|
| `llm/` | onnxruntime-genai model: text → semantic speech tokens |
| `flow_encoder.onnx` | tokens + prompt → conditioning tensors |
| `flow_estimator.onnx` | one flow-matching velocity step (driven by an Euler loop) |
| `hift.onnx` | mel spectrogram → waveform |
| `campplus.onnx` | speaker encoder, for zero-shot voice cloning |
| `s3_tokenizer.onnx` | semantic tokenizer for the reference wav, for zero-shot voice cloning (third-party, Apache 2.0 — see NOTICE) |
| `s3_mel_filters.npz` | mel filterbank asset needed by `s3_tokenizer.onnx` |
| `flow_rand_noise.npy` | fixed flow-matching ODE initial noise, extracted from the reference model |
| `default_prompt.wav` | bundled reference voice used when no `--audio-prompt` is given (see below; third-party, same NonCommercial license — see NOTICE) |
| `meta.json` | sample rate, mel channels, ODE steps, and other runtime constants |

## Usage

This model is designed to run through the `jpn-sarashina-onnx` backend of
[sherox](https://github.com/bagustris/sherox):

```bash
pip install 'sherox[tts-ja-sarashina-onnx]'

# download this repo into models/sarashina-onnx/, then:
sherox.tts --lang jpn-sarashina-onnx --text "こんにちは。" --output out.wav

# zero-shot voice cloning — also torch-free, no extra install needed
sherox.tts --lang jpn-sarashina-onnx \\
    --text "明日は友達と映画を見に行きます。" \\
    --audio-prompt reference.wav --audio-prompt-text "参照音声の書き起こし。" \\
    --output cloned.wav

# --seed lets you work around a bad sample for a specific phrase (see below)
sherox.tts --lang jpn-sarashina-onnx --text "お元気ですか。" --output out.wav --seed 2
```

The runtime uses `repetition_penalty=1.3` by default on the LLM stage; without
it the model tends to get stuck repeating a single semantic token at the start
of generation (a property of the base model, independent of the ONNX export).

### Default voice (no `--audio-prompt`)

The reference model's own [prompting guide](https://github.com/sbintuitions/sarashina2.2-tts#prompting-guide)
states that generation quality depends heavily on having a real audio prompt —
it describes no supported "no prompt" mode. A zero speaker-embedding fallback
is therefore an unsupported configuration, not just a degraded one (verified:
the original PyTorch backend is also unstable there — e.g. 5.7s of near-silence
for a 5-character greeting with certain seeds).

When `--audio-prompt` is omitted, this model bundles `default_prompt.wav`
(sourced from `sbintuitions/sarashina2.2-tts`'s own official demo samples,
under the same NonCommercial license — see NOTICE) and uses it automatically
instead of a zero embedding. This is a real but partial improvement, not a
full fix for content accuracy: some phrases that were wrong with the old
zero-embedding fallback now decode correctly; others that were fine with it
can come out wrong with the bundled prompt instead (a longer reference can
suppress/truncate the LLM's generated token count for the target text,
sometimes severely — shorter references were empirically better here). If a
specific phrase comes out wrong, try a different `--seed`; there is no
evidence of one seed that's reliably correct across every phrase.

## Regenerating these artifacts

```bash
pip install 'sherox[tts-ja-sarashina-onnx-export]'
python -m sherox.sarashina_onnx_export \\
    --model-dir <sarashina2.2-tts checkpoint> --out-dir models/sarashina-onnx
    # add --precision int4 for a smaller/faster but less accurate LLM
```

## Attribution & License

{_ATTRIBUTION_NOTICE}

Original model: [`{_BASE_MODEL}`](https://huggingface.co/{_BASE_MODEL}) by SB Intuitions Corp.
Redistributed under the Sarashina Model NonCommercial License Agreement — see
[`LICENSE`](LICENSE). Commercial use requires a separate agreement with SB Intuitions.
"""
    return frontmatter + body


def package(onnx_dir: str, checkpoint_dir: str, out_dir: str, repo_id: str) -> Path:
    """Assemble a Hugging-Face-ready upload directory. Returns its path."""
    onnx = Path(onnx_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta_path = onnx / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{meta_path} not found — run `python -m sherox.sarashina_onnx_export` first."
        )
    meta = json.loads(meta_path.read_text())

    # Copy ONNX artifacts.
    for name in _ARTIFACTS:
        src = onnx / name
        if not src.is_file():
            raise FileNotFoundError(f"Expected artifact missing: {src}")
        shutil.copy2(src, out / name)

    # Copy the LLM subdirectory (model.onnx + external data + tokenizer files).
    llm_src = onnx / "llm"
    if not llm_src.is_dir():
        raise FileNotFoundError(f"Expected LLM directory missing: {llm_src}")
    shutil.copytree(llm_src, out / "llm", dirs_exist_ok=True)

    # License compliance: copy LICENSE from the source checkpoint if present,
    # otherwise leave a clear placeholder the uploader must fill from the source repo.
    license_src = Path(checkpoint_dir) / "LICENSE"
    license_dst = out / "LICENSE"
    if license_src.is_file():
        shutil.copy2(license_src, license_dst)
    else:
        license_dst.write_text(
            "Place the verbatim Sarashina Model NonCommercial License here.\n"
            f"Download it from: {_LICENSE_LINK}\n\n" + _ATTRIBUTION_NOTICE + "\n"
        )
        print(
            f"[hf-package] WARNING: {license_src} not found. Wrote a LICENSE placeholder — "
            "you MUST replace it with the full license text before uploading."
        )

    # Required attribution notice (redistribution condition 3(b)).
    (out / "NOTICE").write_text(_ATTRIBUTION_NOTICE + "\n")

    # Track the large ONNX binaries with Git LFS (Hugging Face requirement).
    (out / ".gitattributes").write_text(
        "*.onnx filter=lfs diff=lfs merge=lfs -text\n"
        "*.onnx.data filter=lfs diff=lfs merge=lfs -text\n"
    )

    # Model card.
    (out / "README.md").write_text(_model_card(repo_id, meta))

    print(f"[hf-package] Upload directory ready at {out}")
    print("[hf-package] Push with:")
    print(f"    huggingface-cli upload {repo_id} {out} . --repo-type model")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package exported Sarashina ONNX artifacts for Hugging Face upload."
    )
    parser.add_argument("--onnx-dir", required=True, help="Directory of exported ONNX artifacts (from sarashina_onnx_export)")
    parser.add_argument("--checkpoint-dir", default="", help="Original Sarashina checkpoint dir (to copy its LICENSE)")
    parser.add_argument("--out-dir", required=True, help="Output directory to assemble for upload")
    parser.add_argument(
        "--repo-id", required=True,
        help="Target HF repo id, e.g. <username>/Sarashina2.2-TTS-ONNX (name MUST start with 'Sarashina')",
    )
    args = parser.parse_args()

    repo_name = args.repo_id.split("/")[-1]
    if not repo_name.lower().startswith("sarashina"):
        parser.error(
            "The Sarashina license requires derivative model names to begin with "
            f"'Sarashina'. Got repo name '{repo_name}'."
        )
    package(args.onnx_dir, args.checkpoint_dir, args.out_dir, args.repo_id)


if __name__ == "__main__":  # pragma: no cover
    main()
