"""
Unit tests for audio utilities.
"""

import os
import numpy as np
import pytest
from src.audio_utils import (
    load_audio_wav,
    save_audio_wav,
    add_gaussian_noise,
    setup_ffmpeg_path
)


def test_setup_ffmpeg():
    ffmpeg_path = setup_ffmpeg_path()
    assert ffmpeg_path is not None
    assert os.path.exists(ffmpeg_path)


def test_audio_save_and_load(tmp_path):
    sr = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate 440 Hz sine wave
    sine_wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    test_wav = str(tmp_path / "test_sine.wav")
    save_audio_wav(test_wav, sine_wave, sr=sr)
    assert os.path.exists(test_wav)

    loaded_audio, loaded_sr = load_audio_wav(test_wav, target_sr=16000)
    assert loaded_sr == 16000
    assert len(loaded_audio) == len(sine_wave)
    # Check max difference due to 16-bit quantization
    assert np.max(np.abs(loaded_audio - sine_wave)) < 1e-3


def test_add_gaussian_noise():
    sr = 16000
    data = (0.5 * np.ones(sr, dtype=np.float32))
    noisy = add_gaussian_noise(data, snr_db=10.0)
    assert noisy.shape == data.shape
    assert not np.array_equal(noisy, data)
    # Energy should be higher due to added noise
    assert np.mean(noisy ** 2) > np.mean(data ** 2)
