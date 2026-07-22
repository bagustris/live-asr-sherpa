"""Export the Sarashina2.2-TTS PyTorch checkpoint to ONNX for the torch-free
`jpn-sarashina-onnx` runtime backend.

This is a one-time, offline step. It needs the heavy dependencies (torch,
transformers, the sarashina2.2-tts package, onnx, onnxscript) that the runtime
backend deliberately avoids — install them with::

    pip install 'sherox[tts-ja-sarashina-onnx-export]'

It produces, under ``<out_dir>``::

    llm/                    onnxruntime-genai int4 model (LLM: text -> semantic tokens)
    flow_encoder.onnx       tokens+prompt -> (mu, mask, spks, cond)
    flow_estimator.onnx     one flow-matching velocity step (driven by an Euler loop)
    hift.onnx               mel -> waveform (with manual STFT/ISTFT, no torch.istft)
    campplus.onnx           speaker encoder (zero-shot voice cloning)
    s3_tokenizer.onnx       semantic tokenizer for the --audio-prompt reference wav
    s3_mel_filters.npz      mel filterbank asset needed by the semantic tokenizer
    meta.json               shapes / constants the runtime needs

The flow estimator and HiFT vocoder use ``torch.istft``/``torch.stft`` internally,
which the ONNX exporter cannot handle; this module substitutes equivalent
real-valued matmul/conv implementations (:class:`_ManualSTFT`, :class:`_ManualISTFT`)
that were validated to match the originals to <1e-3 on real mel spectrograms.

campplus.onnx and s3_tokenizer.onnx, together with sherox.sarashina_audio_frontend's
pure-numpy DSP, are what let zero-shot voice cloning run without torch too — see
sherox.sarashina_onnx.extract_prompt_features.

Run as::

    python -m sherox.sarashina_onnx_export --model-dir models/sarashina --out-dir models/sarashina-onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# NOTE: torch and the sarashina_tts package are imported lazily inside functions
# so that merely importing this module (e.g. for --help) doesn't require them.

_SEMANTIC_VOCAB_SIZE = 6561  # speech_tokenizer_v2_25hz codebook size
_MEL_CHANNELS = 80
_SAMPLE_RATE = 24000
_N_TIMESTEPS = 10  # flow-matching Euler steps (hardcoded in the reference model)
_INFERENCE_CFG_RATE = 0.7
_NUM_LATENCY_TOKENS = 1


# ── Manual STFT / ISTFT (ONNX-exportable, no complex tensors) ─────────────────

def _build_manual_stft(torch, nn, F):
    class _ManualSTFT(nn.Module):
        """torch.stft(center=True, return_complex=True) + view_as_real, via Conv1d."""

        def __init__(self, n_fft, hop_len, window):
            super().__init__()
            self.n_fft = int(n_fft)
            self.hop_len = int(hop_len)
            self.pad = self.n_fft // 2
            n_freq = self.n_fft // 2 + 1
            k = torch.arange(n_freq, dtype=window.dtype).unsqueeze(1)
            n = torch.arange(self.n_fft, dtype=window.dtype).unsqueeze(0)
            ang = 2 * torch.pi * k * n / self.n_fft
            self.register_buffer("cos_kernel", (torch.cos(ang) * window.unsqueeze(0)).unsqueeze(1))
            self.register_buffer("sin_kernel", (-torch.sin(ang) * window.unsqueeze(0)).unsqueeze(1))

        def forward(self, x):
            x = x.unsqueeze(1)
            x = F.pad(x, (self.pad, self.pad), mode="reflect")
            real = F.conv1d(x, self.cos_kernel.to(x.dtype), stride=self.hop_len)
            imag = F.conv1d(x, self.sin_kernel.to(x.dtype), stride=self.hop_len)
            return real, imag

    return _ManualSTFT


def _build_manual_istft(torch, nn):
    class _ManualISTFT(nn.Module):
        """torch.istft(center=True) via inverse-DFT matmul + windowed overlap-add."""

        def __init__(self, n_fft, hop_len, window, center=True):
            super().__init__()
            self.n_fft = int(n_fft)
            self.hop_len = int(hop_len)
            self.center = bool(center)
            self.register_buffer("window", window)
            n_freq = self.n_fft // 2 + 1
            k = torch.arange(0, n_freq, dtype=window.dtype)
            n = torch.arange(0, self.n_fft, dtype=window.dtype)
            ang = 2 * torch.pi / self.n_fft * k.unsqueeze(1) * n.unsqueeze(0)
            cos_kn = torch.cos(ang)
            sin_kn = torch.sin(ang)
            idft_real = cos_kn.clone()
            idft_imag = torch.zeros_like(sin_kn)
            if self.n_fft > 2:
                idft_real[1:-1] = 2.0 * cos_kn[1:-1]
                idft_imag[1:-1] = -2.0 * sin_kn[1:-1]
            scale = 1.0 / self.n_fft
            self.register_buffer("idft_real_part", idft_real * scale)
            self.register_buffer("idft_imag_part", idft_imag * scale)

        def forward(self, real, imag):
            _, _, T = real.shape
            idft_r = self.idft_real_part.to(real.dtype)
            idft_i = self.idft_imag_part.to(real.dtype)
            window = self.window.to(real.dtype)
            frames = torch.matmul(real.transpose(1, 2), idft_r) + torch.matmul(imag.transpose(1, 2), idft_i)
            frames = frames * window
            B = real.shape[0]
            hop, n_fft = self.hop_len, self.n_fft
            out_len_full = (T - 1) * hop + n_fft
            device = real.device
            t_idx = torch.arange(T, device=device) * hop
            n_idx = torch.arange(n_fft, device=device)
            idx = (t_idx[:, None] + n_idx[None, :]).reshape(1, -1).expand(B, -1)
            y = torch.zeros(B, out_len_full, dtype=frames.dtype, device=device)
            y = y.scatter_add(1, idx, frames.reshape(B, -1))
            w_sq = window ** 2
            denom = torch.zeros(out_len_full, dtype=frames.dtype, device=device)
            denom = denom.scatter_add(
                0, (t_idx[:, None] + n_idx[None, :]).reshape(-1),
                w_sq.unsqueeze(0).expand(T, -1).reshape(-1),
            )
            y = y / denom.clamp_min(1e-8)
            if self.center:
                pad = n_fft // 2
                y = y[:, pad:out_len_full - pad]
            return y

    return _ManualISTFT


# ── Wrapper modules for export ────────────────────────────────────────────────

def _build_flow_encoder_wrapper(torch, nn, F):
    class _FlowEncoderWrapper(nn.Module):
        """CausalMaskedDiffWithXvec.forward() up to (not including) the decoder ODE."""

        def __init__(self, flow):
            super().__init__()
            self.flow = flow

        def forward(self, token, token_len, prompt_feat, prompt_feat_len, embedding):
            from sarashina_tts.flow_matching.upsample_encoder import make_pad_mask  # noqa: PLC0415
            token_len = token_len.long()
            prompt_feat_len = prompt_feat_len.long()
            embedding = F.normalize(embedding, dim=1)
            embedding = self.flow.spk_embed_affine_layer(embedding)
            mask = (~make_pad_mask(token_len, max_len=token.shape[1])).unsqueeze(-1).to(embedding)
            token_emb = self.flow.input_embedding(torch.clamp(token, min=0)) * mask
            h, h_lengths = self.flow.encoder(token_emb, token_len, streaming=False)
            h = self.flow.encoder_proj(h)
            conds = torch.zeros_like(h, device=token.device)
            for i, j in enumerate(prompt_feat_len):
                conds[i, :j] = prompt_feat[i, :j]
            conds = conds.transpose(1, 2)
            h_lengths = h_lengths.sum(dim=-1).squeeze(dim=1)
            out_mask = (~make_pad_mask(h_lengths, max_len=h.shape[1])).to(h)
            mu = h.transpose(1, 2).contiguous()
            return mu, out_mask.unsqueeze(1), embedding, conds

    return _FlowEncoderWrapper


def _build_hift_wrapper(torch, nn, F, hift, ManualSTFT, ManualISTFT):
    class _ExportableHiFT(nn.Module):
        """HiFTGenerator.forward but routing STFT/ISTFT through the manual modules."""

        def __init__(self, hift):
            super().__init__()
            self.hift = hift
            self.istft = ManualISTFT(hift.istft_params["n_fft"], hift.istft_params["hop_len"], hift.stft_window)
            self.stft = ManualSTFT(hift.istft_params["n_fft"], hift.istft_params["hop_len"], hift.stft_window)

        def forward(self, speech_feat):
            f0 = self.hift.f0_predictor(speech_feat)
            s = self.hift.f0_upsamp(f0[:, None]).transpose(1, 2)
            s, _, _ = self.hift.m_source(s)
            s = s.transpose(1, 2)
            s_stft_real, s_stft_imag = self.stft(s.squeeze(1))
            s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
            x = self.hift.conv_pre(speech_feat)
            for i in range(self.hift.num_upsamples):
                x = F.leaky_relu(x, self.hift.lrelu_slope)
                x = self.hift.ups[i](x)
                if i == self.hift.num_upsamples - 1:
                    x = self.hift.reflection_pad(x)
                si = self.hift.source_downs[i](s_stft)
                si = self.hift.source_resblocks[i](si)
                x = x + si
                xs = None
                for j in range(self.hift.num_kernels):
                    out = self.hift.resblocks[i * self.hift.num_kernels + j](x)
                    xs = out if xs is None else xs + out
                x = xs / self.hift.num_kernels
            x = F.leaky_relu(x)
            x = self.hift.conv_post(x)
            n_fft = self.hift.istft_params["n_fft"]
            magnitude = torch.clip(torch.exp(x[:, :n_fft // 2 + 1, :]), max=1e2)
            phase = torch.sin(x[:, n_fft // 2 + 1:, :])
            real = magnitude * torch.cos(phase)
            imag = magnitude * torch.sin(phase)
            wav = self.istft(real, imag)
            return torch.clamp(wav, -self.hift.audio_limit, self.hift.audio_limit)

    return _ExportableHiFT(hift)


# ── Export driver ─────────────────────────────────────────────────────────────

def export(model_dir: str, out_dir: str, precision: str = "int4") -> None:
    """Export all ONNX artifacts from the Sarashina checkpoint at *model_dir* into *out_dir*."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- LLM via onnxruntime-genai model builder ---
    print("[sarashina-onnx-export] Building LLM ONNX (onnxruntime-genai)…")
    from onnxruntime_genai.models.builder import create_model  # noqa: PLC0415
    llm_out = out / "llm"
    llm_out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "_hf_cache"
    create_model("", model_dir, str(llm_out), precision, "cpu", str(cache_dir), hf_remote="false")

    # --- Load the flow + HiFT modules from the checkpoint ---
    print("[sarashina-onnx-export] Loading flow + HiFT PyTorch modules…")
    from sarashina_tts.flow_matching.decoder import FlowDecoder  # noqa: PLC0415
    from sarashina_tts.flow_matching.flow import CausalMaskedDiffWithXvec  # noqa: PLC0415

    flow = CausalMaskedDiffWithXvec()
    flow.load_state_dict(torch.load(f"{model_dir}/flow.pt", map_location="cpu", weights_only=True), strict=True)
    flow.eval()

    decoder = FlowDecoder(model_dir, fp16=False, device="cpu")
    hift = decoder.hift  # weight_norm already removed inside FlowDecoder

    ManualSTFT = _build_manual_stft(torch, nn, F)
    ManualISTFT = _build_manual_istft(torch, nn)
    FlowEncoderWrapper = _build_flow_encoder_wrapper(torch, nn, F)

    # --- Export flow encoder ---
    print("[sarashina-onnx-export] Exporting flow encoder…")
    enc_wrapper = FlowEncoderWrapper(flow).eval()
    token = torch.randint(0, _SEMANTIC_VOCAB_SIZE, (1, 128), dtype=torch.int64)
    token_len = torch.tensor([128], dtype=torch.int32)
    prompt_feat = torch.randn(1, 40, _MEL_CHANNELS, dtype=torch.float32)
    prompt_feat_len = torch.tensor([40], dtype=torch.int32)
    embedding = torch.randn(1, 192, dtype=torch.float32)
    torch.onnx.export(
        enc_wrapper, (token, token_len, prompt_feat, prompt_feat_len, embedding),
        str(out / "flow_encoder.onnx"),
        input_names=["token", "token_len", "prompt_feat", "prompt_feat_len", "embedding"],
        output_names=["mu", "mask", "spks", "cond"],
        dynamic_axes={
            "token": {0: "batch", 1: "token_len"}, "token_len": {0: "batch"},
            "prompt_feat": {0: "batch", 1: "prompt_len"}, "prompt_feat_len": {0: "batch"},
            "embedding": {0: "batch"},
            "mu": {0: "batch", 2: "time"}, "mask": {0: "batch", 2: "time"},
            "spks": {0: "batch"}, "cond": {0: "batch", 2: "time"},
        },
        opset_version=17, dynamo=False,
    )

    # --- Export flow estimator (single velocity step) ---
    print("[sarashina-onnx-export] Exporting flow estimator…")
    estimator = flow.decoder.estimator.eval()
    seq = 158
    x_in = torch.randn(2, _MEL_CHANNELS, seq)
    mask_in = torch.ones(2, 1, seq)
    mu_in = torch.randn(2, _MEL_CHANNELS, seq)
    t_in = torch.full((2,), 0.3)
    spks_in = torch.randn(2, _MEL_CHANNELS)
    cond_in = torch.randn(2, _MEL_CHANNELS, seq)
    torch.onnx.export(
        estimator, (x_in, mask_in, mu_in, t_in, spks_in, cond_in, False),
        str(out / "flow_estimator.onnx"),
        input_names=["x", "mask", "mu", "t", "spks", "cond"],
        output_names=["dphi_dt"],
        dynamic_axes={
            "x": {0: "batch2", 2: "time"}, "mask": {0: "batch2", 2: "time"},
            "mu": {0: "batch2", 2: "time"}, "t": {0: "batch2"},
            "spks": {0: "batch2"}, "cond": {0: "batch2", 2: "time"},
            "dphi_dt": {0: "batch2", 2: "time"},
        },
        opset_version=17, dynamo=False,
    )

    # --- Export HiFT vocoder ---
    print("[sarashina-onnx-export] Exporting HiFT vocoder…")
    hift_wrapper = _build_hift_wrapper(torch, nn, F, hift, ManualSTFT, ManualISTFT).eval()
    speech_feat = torch.randn(1, _MEL_CHANNELS, 300, dtype=torch.float32)
    torch.onnx.export(
        hift_wrapper, (speech_feat,), str(out / "hift.onnx"),
        input_names=["speech_feat"], output_names=["generated_speech"],
        dynamic_axes={"speech_feat": {0: "batch", 2: "seq_len"}, "generated_speech": {0: "batch", 1: "wav_len"}},
        opset_version=17, dynamo=False,
    )

    # --- Export CAMPPlus speaker encoder (for zero-shot voice cloning) ---
    print("[sarashina-onnx-export] Exporting CAMPPlus speaker encoder…")
    from sarashina_tts.speech_encoder.speech_encoder import SpeechEncoder  # noqa: PLC0415

    speech_encoder = SpeechEncoder(f"{model_dir}/campplus_cn_common.bin", device="cpu")
    campplus = speech_encoder.model.eval()
    dummy_fbank = torch.randn(1, 200, 80, dtype=torch.float32)
    torch.onnx.export(
        campplus, (dummy_fbank,), str(out / "campplus.onnx"),
        input_names=["fbank"], output_names=["embedding"],
        dynamic_axes={"fbank": {1: "time"}},
        opset_version=17, dynamo=False,
    )

    # --- Bundle the S3 semantic tokenizer (already ships as ONNX upstream — no
    # export needed, just copy the file s3tokenizer already downloaded/cached) ---
    print("[sarashina-onnx-export] Bundling S3 semantic tokenizer…")
    import os  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import s3tokenizer  # noqa: PLC0415

    cache_default = os.path.join(os.path.expanduser("~"), ".cache")
    s3_cache_dir = os.path.join(os.getenv("XDG_CACHE_HOME", cache_default), "s3tokenizer")
    s3_onnx_path = s3tokenizer._download("speech_tokenizer_v2_25hz", s3_cache_dir)  # noqa: SLF001
    shutil.copy2(s3_onnx_path, out / "s3_tokenizer.onnx")
    mel_filters_asset = Path(s3tokenizer.__file__).parent / "assets" / "mel_filters.npz"
    shutil.copy2(mel_filters_asset, out / "s3_mel_filters.npz")

    meta = {
        "sample_rate": _SAMPLE_RATE,
        "mel_channels": _MEL_CHANNELS,
        "semantic_vocab_size": _SEMANTIC_VOCAB_SIZE,
        "n_timesteps": _N_TIMESTEPS,
        "inference_cfg_rate": _INFERENCE_CFG_RATE,
        "num_latency_tokens": _NUM_LATENCY_TOKENS,
        "precision": precision,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[sarashina-onnx-export] Done. Artifacts written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Sarashina2.2-TTS to ONNX for the torch-free runtime backend.")
    parser.add_argument("--model-dir", required=True, help="Path to the Sarashina checkpoint directory (flow.pt, hift.pt, config.json, …)")
    parser.add_argument("--out-dir", required=True, help="Output directory for ONNX artifacts")
    parser.add_argument("--precision", default="int4", choices=["int4", "fp16", "fp32"], help="LLM quantization precision")
    args = parser.parse_args()
    export(args.model_dir, args.out_dir, args.precision)


if __name__ == "__main__":  # pragma: no cover
    main()
