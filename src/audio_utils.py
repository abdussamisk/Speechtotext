"""
Audio utilities: loading, resampling, format conversion,
SNR noise injection, and ffmpeg runtime resolution.
"""

import os
import sys
import numpy as np
from typing import Tuple, Optional


def setup_ffmpeg_path() -> Optional[str]:
    """
    Locates ffmpeg executable and injects it into os.environ['PATH']
    to ensure Whisper can locate ffmpeg on Windows seamlessly.
    """
    import shutil

    # 1. Check if already in PATH
    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        return which_ffmpeg

    # 2. Check imageio_ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(ffmpeg_exe):
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            target_alias = os.path.join(ffmpeg_dir, "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
            if not os.path.exists(target_alias):
                shutil.copyfile(ffmpeg_exe, target_alias)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            return target_alias
    except ImportError:
        pass

    return None


def load_audio_wav(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Reads a WAV file and converts to mono float32 normalized to [-1.0, 1.0],
    resampling if necessary using scipy.
    """
    from scipy.io import wavfile
    from scipy import signal

    sr, data = wavfile.read(file_path)

    # Convert integer types to float32 [-1.0, 1.0]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    # Convert multi-channel to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to target_sr if required
    if sr != target_sr:
        num_samples = int(len(data) * float(target_sr) / sr)
        data = signal.resample(data, num_samples)
        sr = target_sr

    return data, sr


def save_audio_wav(file_path: str, data: np.ndarray, sr: int = 16000) -> None:
    """
    Saves float32 audio data to 16-bit PCM WAV.
    """
    from scipy.io import wavfile

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    # Clip to [-1.0, 1.0]
    clipped = np.clip(data, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)
    wavfile.write(file_path, sr, int16_data)


def add_gaussian_noise(audio: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """
    Adds Gaussian white noise to an audio waveform at a designated Signal-to-Noise Ratio (SNR).
    Used for noise-robustness experiments.
    
    Args:
        audio: 1D numpy array of audio samples (float32).
        snr_db: Target SNR in decibels (e.g., 20dB = light noise, 10dB = moderate, 0dB = heavy).
        
    Returns:
        Noisy audio array.
    """
    sig_power = np.mean(audio ** 2)
    if sig_power <= 1e-10:
        return audio.copy()

    # SNR = 10 * log10(sig_power / noise_power)
    # noise_power = sig_power / (10 ** (snr_db / 10))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)

    noisy_audio = audio + noise
    return np.clip(noisy_audio, -1.0, 1.0)
