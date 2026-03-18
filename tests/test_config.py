from dataclasses import fields

from sherox.config import Config, SegmentConfig, TtsConfig


def test_default_model_dir():
    assert Config().model_dir == "models/zipformer-en-2023"


def test_default_sample_rate():
    assert Config().sample_rate == 16000


def test_default_chunk_size():
    assert Config().chunk_size == 0.16


def test_default_num_threads():
    assert Config().num_threads == 4


def test_default_model_type_is_empty():
    assert Config().model_type == ""


def test_default_offline_is_false():
    assert Config().offline is False


def test_default_vad_model_is_empty():
    assert Config().vad_model == ""


def test_default_vad_type_is_silero():
    assert Config().vad_type == "silero"


def test_default_ten_vad_model_is_int8():
    assert Config().ten_vad_model == "ten-vad.int8.onnx"


def test_default_vad_threshold():
    assert Config().vad_threshold == 0.5


def test_default_vad_min_silence_duration():
    assert Config().vad_min_silence_duration == 0.5


def test_default_vad_min_speech_duration():
    assert Config().vad_min_speech_duration == 0.25


def test_default_language():
    assert Config().language == "en"


def test_default_show_mic_level_is_false():
    assert Config().show_mic_level is False


def test_custom_values():
    cfg = Config(
        model_dir="models/whisper",
        sample_rate=44100,
        chunk_size=0.2,
        num_threads=8,
        model_type="whisper",
        offline=True,
        vad_model="models/vad.onnx",
        vad_threshold=0.7,
        vad_min_silence_duration=1.0,
        vad_min_speech_duration=0.5,
        language="zh",
        show_mic_level=True,
    )
    assert cfg.model_dir == "models/whisper"
    assert cfg.sample_rate == 44100
    assert cfg.chunk_size == 0.2
    assert cfg.num_threads == 8
    assert cfg.model_type == "whisper"
    assert cfg.offline is True
    assert cfg.vad_model == "models/vad.onnx"
    assert cfg.vad_threshold == 0.7
    assert cfg.vad_min_silence_duration == 1.0
    assert cfg.vad_min_speech_duration == 0.5
    assert cfg.language == "zh"
    assert cfg.show_mic_level is True


def test_config_is_mutable_dataclass():
    cfg = Config()
    cfg.offline = True
    assert cfg.offline is True


def test_config_has_expected_fields():
    field_names = {f.name for f in fields(Config)}
    expected = {
        "model_dir", "sample_rate", "chunk_size", "num_threads",
        "model_type", "offline", "vad_model", "vad_type", "ten_vad_model",
        "vad_threshold", "vad_min_silence_duration", "vad_min_speech_duration",
        "language", "show_mic_level",
    }
    assert expected.issubset(field_names)


# ── SegmentConfig ─────────────────────────────────────────────────────────────

def test_segment_default_vad_type():
    assert SegmentConfig().vad_type == "silero"


def test_segment_default_ten_vad_model():
    assert SegmentConfig().ten_vad_model == "ten-vad.int8.onnx"


def test_segment_default_threshold():
    assert SegmentConfig().vad_threshold == 0.5


def test_segment_default_sample_rate():
    assert SegmentConfig().sample_rate == 16000


def test_segment_default_num_threads():
    assert SegmentConfig().num_threads == 4


def test_segment_default_show_timestamps():
    assert SegmentConfig().show_timestamps is True


def test_segment_default_output_dir_is_empty():
    assert SegmentConfig().output_dir == ""


def test_segment_custom_values():
    cfg = SegmentConfig(
        vad_type="ten-vad",
        vad_threshold=0.7,
        sample_rate=48000,
        output_dir="/tmp/segs",
    )
    assert cfg.vad_type == "ten-vad"
    assert cfg.vad_threshold == 0.7
    assert cfg.sample_rate == 48000
    assert cfg.output_dir == "/tmp/segs"


def test_segment_has_expected_fields():
    field_names = {f.name for f in fields(SegmentConfig)}
    expected = {
        "vad_type", "vad_model", "ten_vad_model",
        "vad_threshold", "vad_min_silence_duration", "vad_min_speech_duration",
        "sample_rate", "capture_rate", "num_threads", "chunk_size",
        "show_timestamps", "show_mic_level", "output_dir",
    }
    assert expected.issubset(field_names)


# ── TtsConfig ─────────────────────────────────────────────────────────────────

def test_tts_default_language():
    assert TtsConfig().language == "ind"


def test_tts_default_speaker_id():
    assert TtsConfig().speaker_id == 0


def test_tts_default_speed():
    assert TtsConfig().speed == 1.0


def test_tts_default_output():
    assert TtsConfig().output == "output.wav"


def test_tts_default_play_is_false():
    assert TtsConfig().play is False


def test_tts_default_num_threads():
    assert TtsConfig().num_threads == 4


def test_tts_default_model_dir_is_empty():
    assert TtsConfig().model_dir == ""


def test_tts_custom_values():
    cfg = TtsConfig(
        language="ind",
        speaker_id=1,
        speed=0.8,
        output="out.wav",
        play=True,
        num_threads=2,
    )
    assert cfg.language == "ind"
    assert cfg.speaker_id == 1
    assert cfg.speed == 0.8
    assert cfg.output == "out.wav"
    assert cfg.play is True
    assert cfg.num_threads == 2


def test_tts_has_expected_fields():
    field_names = {f.name for f in fields(TtsConfig)}
    expected = {"model_dir", "language", "speaker_id", "speed", "output", "play", "num_threads"}
    assert expected.issubset(field_names)
