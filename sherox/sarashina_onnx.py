"""Torch-free ONNX runtime for the Sarashina2.2-TTS pipeline.

Drives the ONNX artifacts produced by :mod:`sherox.sarashina_onnx_export`:

    LLM (onnxruntime-genai) -> semantic tokens
    flow encoder (onnxruntime) -> mu, mask, spks, cond
    flow estimator + Euler ODE loop (onnxruntime) -> mel
    HiFT vocoder (onnxruntime) -> waveform

The runtime depends only on ``onnxruntime`` + ``onnxruntime-genai`` + ``numpy``
(install with ``pip install 'sherox[tts-ja-sarashina-onnx]'``). It never imports
torch.

Zero-shot voice cloning still needs reference-audio features (semantic tokens,
speaker embedding, prompt mel). Extracting those currently reuses the original
torch-based ``sarashina_tts`` extractors (see :func:`extract_prompt_features`),
so cloning mode requires the ``tts-ja-sarashina`` extra to also be installed.
Default-voice synthesis (no ``--audio-prompt``) is fully torch-free.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

# Token constants (mirrors sarashina_tts.additional_tokens; duplicated here so the
# runtime doesn't import the heavy torch-based package).
_SPEECH_START_TOKEN = "<|speech_start|>"
_SEMANTIC_TOKEN_TEMPLATE = "<|semantic_{}|>"
_SEMANTIC_RE = re.compile(r"<\|semantic_(\d+)\|>")

# LLM sampling defaults. repetition_penalty is deliberately > 1.0: without it the
# model reliably gets stuck emitting one semantic token dozens–hundreds of times
# at the prompt->content handoff (verified across seeds, and present in the
# original torch pipeline too, whose HuggingFace path hardcodes penalty=1.0).
_DEFAULT_REPETITION_PENALTY = 1.3
_DEFAULT_TEMPERATURE = 0.9
_DEFAULT_TOP_P = 0.95
_DEFAULT_MAX_NEW_TOKENS = 2048


def _semantic_ids_to_str(ids) -> str:
    return "".join(_SEMANTIC_TOKEN_TEMPLATE.format(int(t)) for t in ids)


def _parse_semantic_ids(text: str) -> list[int]:
    return [int(m) for m in _SEMANTIC_RE.findall(text)]


class SarashinaOnnxRuntime:
    """Loads the ONNX artifacts once and synthesises waveforms from text."""

    def __init__(self, model_dir: str, *, num_threads: int = 4):
        import onnxruntime as ort  # noqa: PLC0415
        import onnxruntime_genai as og  # noqa: PLC0415

        self._dir = Path(model_dir)
        meta_path = self._dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"meta.json not found in {model_dir}. Run "
                "`python -m sherox.sarashina_onnx_export` to produce the ONNX artifacts."
            )
        self.meta = json.loads(meta_path.read_text())

        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads

        self._og = og
        self._llm = og.Model(str(self._dir / "llm"))
        self._tokenizer = og.Tokenizer(self._llm)
        self._enc = ort.InferenceSession(str(self._dir / "flow_encoder.onnx"), so, providers=["CPUExecutionProvider"])
        self._est = ort.InferenceSession(str(self._dir / "flow_estimator.onnx"), so, providers=["CPUExecutionProvider"])
        self._hift = ort.InferenceSession(str(self._dir / "hift.onnx"), so, providers=["CPUExecutionProvider"])

    # -- Stage 1: LLM (text -> semantic token ids) --------------------------------
    def _run_llm(
        self,
        text: str,
        audio_prompt_text: str,
        audio_prompt_tokens: Optional[list[int]],
        *,
        seed: int,
        repetition_penalty: float,
        max_new_tokens: int,
    ) -> list[int]:
        if audio_prompt_tokens is not None:
            token_suffix = _semantic_ids_to_str(audio_prompt_tokens)
            prompt = f"{audio_prompt_text}{text}{_SPEECH_START_TOKEN}{token_suffix}"
        else:
            prompt = f"{text}{_SPEECH_START_TOKEN}"

        input_tokens = self._tokenizer.encode(prompt)
        params = self._og.GeneratorParams(self._llm)
        params.set_search_options(
            max_length=len(input_tokens) + max_new_tokens,
            do_sample=True,
            temperature=_DEFAULT_TEMPERATURE,
            top_p=_DEFAULT_TOP_P,
            repetition_penalty=repetition_penalty,
            random_seed=seed,
        )
        generator = self._og.Generator(self._llm, params)
        generator.append_tokens(input_tokens)
        new_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens.append(generator.get_next_tokens()[0])
        decoded = self._tokenizer.decode(new_tokens)
        return _parse_semantic_ids(decoded)

    # -- Stage 2+3: flow encoder + Euler ODE loop (semantic tokens -> mel) --------
    def _run_flow(
        self,
        semantic_ids: list[int],
        audio_prompt_tokens: Optional[list[int]],
        flow_embedding: np.ndarray,
        prompt_feat: np.ndarray,
    ) -> np.ndarray:
        num_latency = self.meta["num_latency_tokens"]
        gen_ids = semantic_ids[num_latency:] if num_latency else semantic_ids
        prompt_ids = audio_prompt_tokens or []
        combined = np.array(prompt_ids + gen_ids, dtype=np.int64)[None, :]
        token_len = np.array([combined.shape[1]], dtype=np.int32)
        prompt_feat = prompt_feat.astype(np.float32)
        embedding = flow_embedding.astype(np.float32).reshape(1, -1)

        # Number of mel frames to drop later (= prompt-reconstruction portion).
        prompt_mel_len = prompt_feat.shape[1]

        # The exported flow encoder can't handle a zero-length prompt_feat (its
        # traced prompt-fill loop reshapes to a fixed rank and crashes on {0, 80}).
        # For the default voice, feed a 1-frame zero buffer instead: it fills
        # conds[:, :1] with zeros — identical to the empty case, since conds is
        # zero-initialized — and leaves mu/mask/spks untouched. We still drop 0
        # frames below because the real prompt length is 0.
        enc_prompt_feat = prompt_feat
        if prompt_mel_len == 0:
            enc_prompt_feat = np.zeros((1, 1, self.meta["mel_channels"]), dtype=np.float32)
        enc_prompt_feat_len = np.array([enc_prompt_feat.shape[1]], dtype=np.int32)

        mu, mask, spks, cond = self._enc.run(None, {
            "token": combined, "token_len": token_len,
            "prompt_feat": enc_prompt_feat, "prompt_feat_len": enc_prompt_feat_len,
            "embedding": embedding,
        })

        n_steps = self.meta["n_timesteps"]
        cfg_rate = self.meta["inference_cfg_rate"]
        t_span = np.linspace(0, 1, n_steps + 1).astype(np.float32)
        t_span = 1 - np.cos(t_span * 0.5 * np.pi)  # cosine scheduler
        rng = np.random.RandomState(0)  # matches the reference model's fixed inference noise
        x = rng.randn(1, self.meta["mel_channels"], mu.shape[2]).astype(np.float32)

        t = float(t_span[0])
        dt = float(t_span[1] - t_span[0])
        for step in range(1, len(t_span)):
            x_in = np.concatenate([x, x], axis=0)
            mask_in = np.concatenate([mask, mask], axis=0).astype(np.float32)
            mu_in = np.zeros((2, *mu.shape[1:]), dtype=np.float32)
            mu_in[:1] = mu
            t_in = np.full((2,), t, dtype=np.float32)
            spks_in = np.zeros((2, *spks.shape[1:]), dtype=np.float32)
            spks_in[:1] = spks
            cond_in = np.zeros((2, *cond.shape[1:]), dtype=np.float32)
            cond_in[:1] = cond

            dphi = self._est.run(None, {
                "x": x_in, "mask": mask_in, "mu": mu_in,
                "t": t_in, "spks": spks_in, "cond": cond_in,
            })[0]
            dphi_cond, dphi_uncond = dphi[:1], dphi[1:]
            dphi_final = (1.0 + cfg_rate) * dphi_cond - cfg_rate * dphi_uncond
            x = x + dt * dphi_final
            t = t + dt
            if step < len(t_span) - 1:
                dt = float(t_span[step + 1]) - t

        # Drop the prompt-reconstruction portion of the mel (matches the reference
        # FlowDecoder.token2wav, which slices mel_out[:, :, prompt_mel_len:]).
        return x[:, :, prompt_mel_len:]

    # -- Stage 4: HiFT vocoder (mel -> waveform) ---------------------------------
    def _run_hift(self, mel: np.ndarray) -> np.ndarray:
        wav = self._hift.run(None, {"speech_feat": mel.astype(np.float32)})[0]
        return wav[0]  # (1, wav_len) -> (wav_len,)

    def synthesise(
        self,
        text: str,
        *,
        audio_prompt_text: str = "",
        audio_prompt_tokens: Optional[list[int]] = None,
        flow_embedding: Optional[np.ndarray] = None,
        prompt_feat: Optional[np.ndarray] = None,
        seed: int = 0,
        repetition_penalty: float = _DEFAULT_REPETITION_PENALTY,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    ) -> tuple[np.ndarray, int]:
        """Return (samples, sample_rate) for *text*.

        For zero-shot voice cloning, pass all of ``audio_prompt_tokens``,
        ``flow_embedding`` and ``prompt_feat`` (see :func:`extract_prompt_features`).
        For the default voice, leave them ``None``.
        """
        cloning = audio_prompt_tokens is not None
        if cloning and (flow_embedding is None or prompt_feat is None):
            raise ValueError("Voice cloning requires audio_prompt_tokens, flow_embedding and prompt_feat together.")
        if not cloning:
            flow_embedding = np.zeros(192, dtype=np.float32)
            prompt_feat = np.zeros((1, 0, self.meta["mel_channels"]), dtype=np.float32)

        semantic_ids = self._run_llm(
            text, audio_prompt_text, audio_prompt_tokens,
            seed=seed, repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
        )
        if not semantic_ids:
            raise RuntimeError("LLM produced no semantic tokens.")
        mel = self._run_flow(semantic_ids, audio_prompt_tokens, flow_embedding, prompt_feat)
        samples = self._run_hift(mel).astype(np.float32)
        return samples, self.meta["sample_rate"]


def extract_prompt_features(audio_prompt_path: str, model_dir: str):
    """Extract (semantic_tokens, flow_embedding, prompt_feat) from a reference wav.

    This currently reuses the torch-based ``sarashina_tts`` extractors, so it
    needs the ``tts-ja-sarashina`` extra installed. It runs once per distinct
    reference voice, so its cost is not on the per-utterance hot path.

    Returns
    -------
    (list[int], np.ndarray, np.ndarray)
        semantic token ids, (192,) speaker embedding, (1, T, 80) prompt mel.
    """
    try:
        from sarashina_tts.generate.generate import SarashinaTTSGenerator  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Zero-shot voice cloning currently needs the torch-based extractors. "
            "Install them with: pip install 'sherox[tts-ja-sarashina]'"
        ) from exc

    gen = SarashinaTTSGenerator(model_dir=model_dir, decoder_fp16=False, watermark=False, device="cpu")
    tokens = gen._extract_audio_prompt_tokens(audio_prompt_path)
    embedding = gen._extract_zero_shot_embedding(audio_prompt_path).cpu().numpy()
    feat = gen._extract_audio_prompt_feat(audio_prompt_path).cpu().numpy()
    return tokens, embedding, feat
