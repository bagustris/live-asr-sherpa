"""Tests for the torch-free Sarashina ONNX runtime and its HF packaging.

These avoid the heavy onnxruntime / onnxruntime-genai / torch dependencies by
exercising the pure-Python helpers and mocking the ONNX sessions.
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sherox.sarashina_onnx as rt
import sherox.sarashina_onnx_hf as hf


# ---------------------------------------------------------------------------
# semantic-token helpers
# ---------------------------------------------------------------------------

class TestSemanticTokenHelpers:
    def test_roundtrip(self):
        ids = [0, 5, 6560]
        s = rt._semantic_ids_to_str(ids)
        assert s == "<|semantic_0|><|semantic_5|><|semantic_6560|>"
        assert rt._parse_semantic_ids(s) == ids

    def test_parse_ignores_other_text(self):
        assert rt._parse_semantic_ids("foo<|semantic_3|>bar<|speech_start|>") == [3]

    def test_parse_empty(self):
        assert rt._parse_semantic_ids("no tokens here") == []


# ---------------------------------------------------------------------------
# SarashinaOnnxRuntime — construction guards + flow ODE glue (sessions mocked)
# ---------------------------------------------------------------------------

class TestRuntimeConstruction:
    def test_missing_meta_raises(self, tmp_path):
        # onnxruntime / genai are imported first; skip cleanly if unavailable.
        pytest.importorskip("onnxruntime")
        pytest.importorskip("onnxruntime_genai")
        with pytest.raises(FileNotFoundError):
            rt.SarashinaOnnxRuntime(str(tmp_path))


class TestRuntimeFlow:
    def _make_runtime(self, tmp_path):
        """Build a runtime with all ONNX sessions replaced by mocks."""
        (tmp_path / "meta.json").write_text(json.dumps({
            "sample_rate": 24000, "mel_channels": 80, "semantic_vocab_size": 6561,
            "n_timesteps": 2, "inference_cfg_rate": 0.7, "num_latency_tokens": 1,
            "precision": "int4",
        }))
        r = rt.SarashinaOnnxRuntime.__new__(rt.SarashinaOnnxRuntime)
        r.meta = json.loads((tmp_path / "meta.json").read_text())
        r._enc = MagicMock()
        r._est = MagicMock()
        r._hift = MagicMock()
        r._rand_noise = np.random.RandomState(0).randn(1, 80, 64).astype(np.float32)
        return r

    def test_run_flow_slices_fixed_noise_buffer_not_fresh_random(self, tmp_path):
        """Regression guard: the flow-matching ODE's initial noise must come
        from the fixed buffer extracted from the reference model (see
        sarashina_onnx_export.py), sliced [:, :, :T] — not freshly sampled
        with a different RNG. With only 10 Euler steps, using a different
        (even if equally N(0,1)) noise realization measurably changes the
        decoded mel enough to garble pronunciation — verified by feeding an
        independently-sampled same-shape buffer into the real reference
        decoder and diffing against the correct-buffer output (max abs diff
        ~3.3, vs ~3e-5 for a correctly-sliced buffer)."""
        r = self._make_runtime(tmp_path)
        T = 6
        r._rand_noise = np.arange(1 * 80 * 64, dtype=np.float32).reshape(1, 80, 64)
        r._enc.run.return_value = [
            np.zeros((1, 80, T), dtype=np.float32),
            np.ones((1, 1, T), dtype=np.float32),
            np.zeros((1, 80), dtype=np.float32),
            np.zeros((1, 80, T), dtype=np.float32),
        ]
        captured_x = {}
        def fake_est_run(_, feed):
            captured_x.setdefault("x", feed["x"].copy())
            return [np.zeros((2, 80, T), dtype=np.float32)]
        r._est.run.side_effect = fake_est_run

        prompt_feat = np.zeros((1, 0, 80), dtype=np.float32)
        r._run_flow([9, 1, 2, 3], None, np.zeros(192, dtype=np.float32), prompt_feat)

        expected_x0 = r._rand_noise[:, :, :T]
        np.testing.assert_array_equal(captured_x["x"][:1], expected_x0)

    def test_run_flow_drops_prompt_frames_and_latency_token(self, tmp_path):
        r = self._make_runtime(tmp_path)
        T = 10
        prompt_mel = 3
        r._enc.run.return_value = [
            np.zeros((1, 80, T), dtype=np.float32),   # mu
            np.ones((1, 1, T), dtype=np.float32),      # mask
            np.zeros((1, 80), dtype=np.float32),       # spks
            np.zeros((1, 80, T), dtype=np.float32),    # cond
        ]
        r._est.run.return_value = [np.zeros((2, 80, T), dtype=np.float32)]

        prompt_feat = np.zeros((1, prompt_mel, 80), dtype=np.float32)
        mel = r._run_flow([9, 1, 2, 3], [7, 8], np.zeros(192, dtype=np.float32), prompt_feat)

        # Encoder token input = prompt_tokens + gen_ids[num_latency:] = [7,8] + [1,2,3]
        enc_feed = r._enc.run.call_args[0][1]
        assert enc_feed["token"].tolist() == [[7, 8, 1, 2, 3]]
        assert enc_feed["prompt_feat_len"].tolist() == [prompt_mel]
        # Prompt-reconstruction frames are trimmed off the front of the mel.
        assert mel.shape == (1, 80, T - prompt_mel)

    def test_run_flow_default_voice_substitutes_nonempty_prompt_buffer(self, tmp_path):
        """Empty prompt_feat (default voice) must be padded to 1 frame for the
        encoder — which can't run on a zero-length prompt — while still dropping
        0 mel frames."""
        r = self._make_runtime(tmp_path)
        T = 8
        r._enc.run.return_value = [
            np.zeros((1, 80, T), dtype=np.float32),
            np.ones((1, 1, T), dtype=np.float32),
            np.zeros((1, 80), dtype=np.float32),
            np.zeros((1, 80, T), dtype=np.float32),
        ]
        r._est.run.return_value = [np.zeros((2, 80, T), dtype=np.float32)]

        empty_prompt = np.zeros((1, 0, 80), dtype=np.float32)
        mel = r._run_flow([9, 1, 2, 3], None, np.zeros(192, dtype=np.float32), empty_prompt)

        enc_feed = r._enc.run.call_args[0][1]
        # Zero-length prompt is padded to a single (zero) frame for the encoder.
        assert enc_feed["prompt_feat"].shape == (1, 1, 80)
        assert enc_feed["prompt_feat_len"].tolist() == [1]
        # combined tokens = gen_ids[num_latency:] only (no prompt tokens prepended).
        assert enc_feed["token"].tolist() == [[1, 2, 3]]
        # No frames dropped — the whole mel is generated content.
        assert mel.shape == (1, 80, T)

    def test_synthesise_requires_all_cloning_inputs(self, tmp_path):
        r = self._make_runtime(tmp_path)
        with pytest.raises(ValueError):
            r.synthesise("テスト", audio_prompt_tokens=[1, 2], flow_embedding=None, prompt_feat=None)


# ---------------------------------------------------------------------------
# extract_prompt_features — dependency guard
# ---------------------------------------------------------------------------

class TestExtractPromptFeatures:
    def _make_wav(self, tmp_path, sr=22050, seconds=1.0):
        import soundfile as sf
        wav_path = tmp_path / "prompt.wav"
        n = int(sr * seconds)
        audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype(np.float32)
        sf.write(str(wav_path), audio, sr)
        return wav_path

    def _make_mel_filters(self, tmp_path):
        # whisper_log_mel needs a (128, n_fft//2+1) = (128, 201) filterbank for n_fft=400.
        path = tmp_path / "s3_mel_filters.npz"
        np.savez(path, mel_128=np.random.RandomState(0).rand(128, 201).astype(np.float32))
        return path

    def test_extracts_tokens_embedding_and_feat(self, tmp_path):
        wav_path = self._make_wav(tmp_path)
        self._make_mel_filters(tmp_path)

        mock_s3_sess = MagicMock()
        mock_s3_sess.run.return_value = [np.array([[5, 6, 7]], dtype=np.int32)]
        mock_campplus_sess = MagicMock()
        mock_campplus_sess.run.return_value = [np.zeros((1, 192), dtype=np.float32)]

        def fake_session(path, providers=None):
            return mock_s3_sess if "s3_tokenizer" in path else mock_campplus_sess

        with patch("onnxruntime.InferenceSession", side_effect=fake_session):
            tokens, embedding, feat = rt.extract_prompt_features(str(wav_path), str(tmp_path))

        assert tokens == [5, 6, 7]
        assert embedding.shape == (1, 192)
        assert feat.ndim == 3
        assert feat.shape[0] == 1 and feat.shape[2] == 80

    def test_uses_correct_onnx_model_paths(self, tmp_path):
        wav_path = self._make_wav(tmp_path)
        self._make_mel_filters(tmp_path)

        mock_sess = MagicMock()
        mock_sess.run.return_value = [np.zeros((1, 1), dtype=np.int32)]
        seen_paths = []

        def fake_session(path, providers=None):
            seen_paths.append(path)
            return mock_sess

        with patch("onnxruntime.InferenceSession", side_effect=fake_session):
            rt.extract_prompt_features(str(wav_path), str(tmp_path))

        assert str(tmp_path / "s3_tokenizer.onnx") in seen_paths
        assert str(tmp_path / "campplus.onnx") in seen_paths


# ---------------------------------------------------------------------------
# HF packaging
# ---------------------------------------------------------------------------

class TestHfPackaging:
    def _make_onnx_dir(self, tmp_path):
        d = tmp_path / "onnx"
        (d / "llm").mkdir(parents=True)
        for name in ["flow_encoder.onnx", "flow_estimator.onnx", "hift.onnx",
                     "campplus.onnx", "s3_tokenizer.onnx", "s3_mel_filters.npz"]:
            (d / name).write_bytes(b"onnx")
        (d / "meta.json").write_text(json.dumps({"precision": "int4", "sample_rate": 24000, "n_timesteps": 10}))
        (d / "llm" / "model.onnx").write_bytes(b"llm")
        (d / "llm" / "genai_config.json").write_text("{}")
        return d

    def test_model_card_contains_required_attribution(self):
        card = hf._model_card("user/Sarashina2.2-TTS-ONNX", {"precision": "int4"})
        assert "Built with Sarashina" in card
        assert hf._ATTRIBUTION_NOTICE in card
        assert "license: other" in card
        assert "base_model: sbintuitions/sarashina2.2-tts" in card

    def test_attribution_covers_both_upstream_licenses(self):
        # s3_tokenizer.onnx is a third-party (Apache 2.0) file, separate from
        # the Sarashina checkpoint's NonCommercial terms — both must be named.
        assert "SB Intuitions" in hf._ATTRIBUTION_NOTICE
        assert "FunAudioLLM/CosyVoice2-0.5B" in hf._ATTRIBUTION_NOTICE
        assert "Apache License 2.0" in hf._ATTRIBUTION_NOTICE

    def test_package_assembles_all_files(self, tmp_path):
        onnx = self._make_onnx_dir(tmp_path)
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "LICENSE").write_text("SARASHINA LICENSE TEXT")
        out = tmp_path / "out"

        hf.package(str(onnx), str(ckpt), str(out), "user/Sarashina2.2-TTS-ONNX")

        for name in ["flow_encoder.onnx", "flow_estimator.onnx", "hift.onnx",
                     "campplus.onnx", "s3_tokenizer.onnx", "s3_mel_filters.npz",
                     "meta.json", "LICENSE", "NOTICE", "README.md", ".gitattributes"]:
            assert (out / name).is_file(), f"missing {name}"
        assert (out / "llm" / "model.onnx").is_file()
        assert (out / "LICENSE").read_text() == "SARASHINA LICENSE TEXT"
        assert hf._ATTRIBUTION_NOTICE in (out / "NOTICE").read_text()
        assert "lfs" in (out / ".gitattributes").read_text()

    def test_package_writes_license_placeholder_when_missing(self, tmp_path):
        onnx = self._make_onnx_dir(tmp_path)
        out = tmp_path / "out"
        hf.package(str(onnx), str(tmp_path / "nonexistent"), str(out), "user/Sarashina2.2-TTS-ONNX")
        placeholder = (out / "LICENSE").read_text()
        assert "Place the verbatim Sarashina" in placeholder

    def test_cli_rejects_non_sarashina_repo_name(self):
        with patch("sys.argv", ["prog", "--onnx-dir", "x", "--out-dir", "y", "--repo-id", "user/Whisper-ONNX"]):
            with pytest.raises(SystemExit):
                hf.main()
