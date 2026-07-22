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
        return r

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
    def test_missing_dependency_raises_importerror(self, tmp_path):
        with patch.dict("sys.modules", {"sarashina_tts.generate.generate": None}):
            with pytest.raises(ImportError):
                rt.extract_prompt_features(str(tmp_path / "x.wav"), str(tmp_path))


# ---------------------------------------------------------------------------
# HF packaging
# ---------------------------------------------------------------------------

class TestHfPackaging:
    def _make_onnx_dir(self, tmp_path):
        d = tmp_path / "onnx"
        (d / "llm").mkdir(parents=True)
        for name in ["flow_encoder.onnx", "flow_estimator.onnx", "hift.onnx"]:
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

    def test_package_assembles_all_files(self, tmp_path):
        onnx = self._make_onnx_dir(tmp_path)
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "LICENSE").write_text("SARASHINA LICENSE TEXT")
        out = tmp_path / "out"

        hf.package(str(onnx), str(ckpt), str(out), "user/Sarashina2.2-TTS-ONNX")

        for name in ["flow_encoder.onnx", "flow_estimator.onnx", "hift.onnx",
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
