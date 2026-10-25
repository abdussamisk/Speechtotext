"""
Text normalization utilities for ASR evaluation (WER/CER).
Handles English and Indic languages (specifically Hindi / Devanagari script).
"""

import re
import unicodedata
from typing import Optional


# Devanagari numerals to standard Arabic digits
DEVANAGARI_DIGITS_MAP = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"
}

# Devanagari punctuation and symbols
DEVANAGARI_PUNCTUATION = ["।", "॥", "॰", "ऽ"]

# Common English contractions
CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bn't\b": " not",
    r"\b're\b": " are",
    r"\b's\b": " is",
    r"\b'd\b": " would",
    r"\b'll\b": " will",
    r"\b've\b": " have",
    r"\b'm\b": " am"
}


def normalize_english(text: str, remove_punct: bool = True) -> str:
    """
    Standard text normalization for English transcription evaluation.
    
    Args:
        text: Input string.
        remove_punct: Whether to strip punctuation.
        
    Returns:
        Normalized lowercase string without redundant whitespace or punctuation.
    """
    if not text:
        return ""
    
    text = text.lower()
    
    # Expand common contractions
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)
        
    if remove_punct:
        # Replace hyphens with spaces
        text = text.replace("-", " ")
        # Strip all punctuation except alphanumeric and space
        text = re.sub(r"[^\w\s]", "", text)
        
    # Collapse consecutive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_hindi(text: str, remove_punct: bool = True, normalize_digits: bool = True) -> str:
    """
    Normalization for Hindi (Devanagari script) text.
    
    Handles:
    - Unicode normalization (NFC/NFKC)
    - Removal of Devanagari danda (।) and double danda (॥)
    - Removal of ASCII punctuation
    - Zero-Width Joiner (ZWJ \\u200D) and Zero-Width Non-Joiner (ZWNJ \\u200C)
    - Standardizing Devanagari numerals to Arabic digits
    - Normalization of nukta characters (e.g., क़ vs क)
    
    Args:
        text: Input Hindi string.
        remove_punct: Whether to strip punctuation.
        normalize_digits: Convert Devanagari numerals to standard digits.
        
    Returns:
        Cleaned, normalized Hindi string.
    """
    if not text:
        return ""
    
    # Unicode canonical decomposition and composition (NFC)
    text = unicodedata.normalize("NFC", text)
    
    # Remove zero-width characters (ZWJ, ZWNJ, soft hyphens)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    
    # Convert Devanagari numerals if requested
    if normalize_digits:
        for dev_digit, arab_digit in DEVANAGARI_DIGITS_MAP.items():
            text = text.replace(dev_digit, arab_digit)
            
    if remove_punct:
        # Remove Devanagari specific punctuation
        for p in DEVANAGARI_PUNCTUATION:
            text = text.replace(p, " ")
            
        # Remove standard ASCII and Latin punctuation
        text = re.sub(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]", " ", text)
        
    # Standardize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str, language: str = "en", remove_punct: bool = True) -> str:
    """
    Unified normalizer dispatching based on language code.
    
    Args:
        text: Input string.
        language: Language code ('en', 'hi', 'te', 'ta', etc.).
        remove_punct: Whether to remove punctuation.
        
    Returns:
        Normalized string.
    """
    lang = (language or "en").lower().strip()
    if lang in ["en", "english"]:
        return normalize_english(text, remove_punct=remove_punct)
    elif lang in ["hi", "hindi", "mr", "marathi", "sa", "sanskrit", "ne", "nepali"]:
        return normalize_hindi(text, remove_punct=remove_punct)
    else:
        # General multilingual fallback
        text = unicodedata.normalize("NFC", text)
        if remove_punct:
            text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip().lower()
