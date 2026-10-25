"""
Whisper Transcription Engine.
Wraps OpenAI Whisper for multilingual speech-to-text,
language identification, timestamp extraction, and performance profiling.
"""

import time
import os
import torch
from typing import Dict, Any, Optional, List
from src.audio_utils import setup_ffmpeg_path


class WhisperTranscriber:
    """
    Multilingual speech recognition transcriber utilizing OpenAI Whisper models.
    Supports model sizes: tiny, base, small, medium, large-v3, etc.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        download_root: Optional[str] = None
    ):
        setup_ffmpeg_path()
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Whisper model '{model_name}' on device: {self.device.upper()}...")
        start_time = time.time()
        import whisper
        self.model = whisper.load_model(model_name, device=self.device, download_root=download_root)
        self.load_time = time.time() - start_time
        print(f"Model '{model_name}' loaded in {self.load_time:.2f} seconds.")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        temperature: float = 0.0,
        beam_size: int = 5,
        best_of: int = 5
    ) -> Dict[str, Any]:
        """
        Transcribes the given audio file.
        
        Args:
            audio_path: Path to .wav, .mp3, or other audio file.
            language: ISO 639-1 language code (e.g. 'en', 'hi') or None for auto-detection.
            task: 'transcribe' or 'translate' (Whisper can also translate to English).
            temperature: Sampling temperature.
            beam_size: Beam search width.
            best_of: Number of candidate samples.
            
        Returns:
            Dictionary with transcription result, detected language, segments, latency, and RTF.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        import whisper

        # Check audio duration
        audio = whisper.load_audio(audio_path)
        duration = len(audio) / whisper.audio.SAMPLE_RATE

        fp16 = (self.device == "cuda")

        options = {
            "task": task,
            "temperature": temperature,
            "beam_size": beam_size,
            "best_of": best_of,
            "fp16": fp16
        }

        if language:
            options["language"] = language

        start_time = time.time()
        result = self.model.transcribe(audio_path, **options)
        latency = time.time() - start_time

        rtf = latency / duration if duration > 0 else 0.0

        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", language or "unknown"),
            "segments": [
                {
                    "id": seg.get("id"),
                    "start": round(seg.get("start", 0.0), 2),
                    "end": round(seg.get("end", 0.0), 2),
                    "text": seg.get("text", "").strip()
                }
                for seg in result.get("segments", [])
            ],
            "audio_duration_sec": round(duration, 2),
            "latency_sec": round(latency, 2),
            "real_time_factor": round(rtf, 3),
            "model": self.model_name,
            "device": self.device
        }

    def detect_language(self, audio_path: str) -> Dict[str, Any]:
        """
        Detects the spoken language from the initial 30 seconds of audio.
        
        Returns:
            Dict with 'detected_language', 'language_code', and top language probabilities.
        """
        import whisper

        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        _, probs = self.model.detect_language(mel)

        detected_lang = max(probs, key=probs.get)
        top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "detected_language": detected_lang,
            "confidence": round(probs[detected_lang], 4),
            "top_candidates": [(lang, round(prob, 4)) for lang, prob in top_probs]
        }
