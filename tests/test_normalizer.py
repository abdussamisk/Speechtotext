"""
Unit tests for text normalization (English and Hindi).
"""

import pytest
from src.normalizer import (
    normalize_english,
    normalize_hindi,
    normalize_text,
    DEVANAGARI_DIGITS_MAP
)


def test_normalize_english_basic():
    text = "Hello, World! This is a Test."
    assert normalize_english(text) == "hello world this is a test"


def test_normalize_english_contractions():
    text = "She can't won't shouldn't don't"
    normalized = normalize_english(text)
    assert "cannot" in normalized
    assert "will not" in normalized


def test_normalize_english_whitespace():
    text = "  Multiple    spaces   and \n newlines.  "
    assert normalize_english(text) == "multiple spaces and newlines"


def test_normalize_hindi_danda():
    text = "नमस्ते दुनिया। यह एक परीक्षण है॥"
    # Danda (।) and double danda (॥) should be removed
    normalized = normalize_hindi(text)
    assert "।" not in normalized
    assert "॥" not in normalized
    assert normalized == "नमस्ते दुनिया यह एक परीक्षण है"


def test_normalize_hindi_digits():
    text = "मेरे पास ५ सेब और १२ संतरे हैं।"
    normalized = normalize_hindi(text, normalize_digits=True)
    assert "5" in normalized
    assert "12" in normalized
    assert "५" not in normalized
    assert "१" not in normalized


def test_normalize_hindi_zero_width():
    # Text containing ZWNJ (\u200c) or ZWJ (\u200d)
    text = "पु\u200Dस्तक और वि\u200Cद्यालय"
    normalized = normalize_hindi(text)
    assert "\u200d" not in normalized
    assert "\u200c" not in normalized


def test_normalize_text_router():
    assert normalize_text("Hello World!", language="en") == "hello world"
    assert normalize_text("नमस्ते भारत!", language="hi") == "नमस्ते भारत"
