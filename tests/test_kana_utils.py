"""Tests for benchmark/kana_utils.py."""

import pytest
from kana_utils import (
    _compute_ker,
    _katakana_to_hiragana,
    _levenshtein,
    _text_to_kana,
)


# ---------------------------------------------------------------------------
# _katakana_to_hiragana
# ---------------------------------------------------------------------------

def test_katakana_to_hiragana_basic():
    """Katakana characters are shifted to hiragana."""
    # ア → あ, イ → い, ウ → う, エ → え, オ → お
    assert _katakana_to_hiragana("アイウエオ") == "あいうえお"


def test_katakana_to_hiragana_long_vowel_preserved():
    """ー (U+30FC) is outside the katakana shift range and is kept as-is."""
    assert _katakana_to_hiragana("ラーメン") == "らーめん"


def test_katakana_to_hiragana_passthrough_hiragana():
    """Hiragana characters are returned unchanged."""
    text = "あいうえお"
    assert _katakana_to_hiragana(text) == text


def test_katakana_to_hiragana_passthrough_kanji():
    """Kanji and ASCII are returned unchanged."""
    assert _katakana_to_hiragana("東京Tokyo") == "東京Tokyo"


def test_katakana_to_hiragana_mixed():
    """Mixed string: only katakana gets converted."""
    # コンピュータ → こんぴゅーた (ー stays ー)
    result = _katakana_to_hiragana("コンピュータ")
    assert result == "こんぴゅーた"


def test_katakana_to_hiragana_empty():
    assert _katakana_to_hiragana("") == ""


# ---------------------------------------------------------------------------
# _text_to_kana  (without pyopenjtalk — fallback path only)
# ---------------------------------------------------------------------------

def test_text_to_kana_hiragana_passthrough():
    """Pure hiragana input is returned as-is (no kanji to convert)."""
    text = "あいうえお"
    result = _text_to_kana(text)
    assert result == text


def test_text_to_kana_katakana_converted():
    """Katakana is converted to hiragana."""
    result = _text_to_kana("アイウエオ")
    assert result == "あいうえお"


def test_text_to_kana_strips_ascii():
    """ASCII characters are stripped (fallback path drops non-kana)."""
    result = _text_to_kana("hello world")
    # All ASCII dropped → empty or only kana
    assert all(ord(c) >= 0x3041 for c in result if c != "ー")


def test_text_to_kana_strips_punctuation():
    """Japanese punctuation is not hiragana and should be dropped."""
    result = _text_to_kana("こんにちは。今日は晴れです！")
    # Punctuation should be removed; remaining chars should be hiragana
    for ch in result:
        assert "\u3041" <= ch <= "\u3096" or ch == "ー", f"Unexpected char: {ch!r}"


def test_text_to_kana_long_vowel_preserved_in_katakana():
    """ー should be preserved when present in katakana input."""
    result = _text_to_kana("ラーメン")
    assert "ー" in result


def test_text_to_kana_empty():
    assert _text_to_kana("") == ""


# ---------------------------------------------------------------------------
# _levenshtein
# ---------------------------------------------------------------------------

def test_levenshtein_identical():
    assert _levenshtein(list("abc"), list("abc")) == 0


def test_levenshtein_empty_left():
    assert _levenshtein([], list("abc")) == 3


def test_levenshtein_empty_right():
    assert _levenshtein(list("abc"), []) == 3


def test_levenshtein_one_substitution():
    assert _levenshtein(list("abc"), list("axc")) == 1


def test_levenshtein_one_insertion():
    assert _levenshtein(list("ac"), list("abc")) == 1


def test_levenshtein_one_deletion():
    assert _levenshtein(list("abc"), list("ac")) == 1


def test_levenshtein_both_empty():
    assert _levenshtein([], []) == 0


# ---------------------------------------------------------------------------
# _compute_ker
# ---------------------------------------------------------------------------

def test_compute_ker_identical_hiragana():
    """Identical hiragana strings → KER = 0."""
    ker, dist, ref_len = _compute_ker("あいうえお", "あいうえお")
    assert ker == 0.0
    assert dist == 0


def test_compute_ker_different_hiragana():
    """Different hiragana strings → KER > 0."""
    ker, dist, ref_len = _compute_ker("あいう", "えおか")
    assert ker > 0.0
    assert dist > 0


def test_compute_ker_empty_both():
    """Both empty → KER = 0, distance = 0."""
    ker, dist, ref_len = _compute_ker("", "")
    assert ker == 0.0
    assert dist == 0
    assert ref_len == 0


def test_compute_ker_empty_reference():
    """Empty reference with non-empty hypothesis → KER = 1.0."""
    ker, dist, ref_len = _compute_ker("", "あいう")
    assert ker == 1.0


def test_compute_ker_empty_hypothesis():
    """Non-empty reference with empty hypothesis → KER = 1.0."""
    ker, dist, ref_len = _compute_ker("あいう", "")
    assert ker == 1.0
    assert ref_len == 3


def test_compute_ker_returns_tuple():
    """_compute_ker returns a 3-tuple (ker, edit_dist, ref_len)."""
    result = _compute_ker("あいう", "あいう")
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_compute_ker_ker_capped_at_one():
    """KER should never exceed 1.0."""
    # Very long hypothesis vs short reference
    ker, dist, ref_len = _compute_ker("あ", "あいうえおかきくけこ")
    assert ker <= 1.0


def test_compute_ker_katakana_same_as_hiragana():
    """Katakana and hiragana representations of the same word should match."""
    # アイウ and あいう should produce the same kana → KER = 0
    ker, dist, ref_len = _compute_ker("アイウ", "あいう")
    assert ker == 0.0


def test_compute_ker_nonnegative_distance():
    """Edit distance is always non-negative."""
    _, dist, _ = _compute_ker("あいう", "うえお")
    assert dist >= 0


def test_compute_ker_ref_len_matches_kana_conversion():
    """Reference length should match the kana length of the reference."""
    from kana_utils import _text_to_kana
    ref = "あいうえお"
    _, _, ref_len = _compute_ker(ref, "")
    assert ref_len == len(_text_to_kana(ref))
