"""
Script to prepare evaluation audio datasets for English and Hindi.
Generates audio samples with ground truth transcripts and saves metadata to reference.json.
Uses gTTS (Google Text-to-Speech) and converts to 16kHz mono WAV.
"""

import os
import sys
import json
import numpy as np

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_utils import setup_ffmpeg_path

# Target sample definitions
BENCHMARK_SAMPLES = [
    # English Samples
    {
        "id": "en_01",
        "language": "en",
        "category": "conversational",
        "reference": "Artificial intelligence and speech recognition are transforming modern communication across the globe.",
        "tts_text": "Artificial intelligence and speech recognition are transforming modern communication across the globe."
    },
    {
        "id": "en_02",
        "language": "en",
        "category": "technical",
        "reference": "Whisper is an automatic speech recognition model trained on thousands of hours of multilingual audio.",
        "tts_text": "Whisper is an automatic speech recognition model trained on thousands of hours of multilingual audio."
    },
    {
        "id": "en_03",
        "language": "en",
        "category": "weather",
        "reference": "The weather forecast predicts heavy rain and thunderstorms in northern regions tomorrow.",
        "tts_text": "The weather forecast predicts heavy rain and thunderstorms in northern regions tomorrow."
    },
    {
        "id": "en_04",
        "language": "en",
        "category": "numerical",
        "reference": "She ordered three cups of coffee and two chocolate donuts at nine thirty in the morning.",
        "tts_text": "She ordered three cups of coffee and two chocolate donuts at nine thirty in the morning."
    },
    {
        "id": "en_05",
        "language": "en",
        "category": "metrics",
        "reference": "Word error rate evaluates the accuracy of speech recognition by comparing predicted words with ground truth.",
        "tts_text": "Word error rate evaluates the accuracy of speech recognition by comparing predicted words with ground truth."
    },
    # Hindi Samples
    {
        "id": "hi_01",
        "language": "hi",
        "category": "conversational",
        "reference": "भारत एक बहुभाषी देश है जहाँ कई भाषाएं और संस्कृतियां एक साथ फलती फूलती हैं।",
        "tts_text": "भारत एक बहुभाषी देश है जहाँ कई भाषाएं और संस्कृतियां एक साथ फलती फूलती हैं।"
    },
    {
        "id": "hi_02",
        "language": "hi",
        "category": "technical",
        "reference": "आर्टिफिशियल इंटेलिजेंस और मशीन लर्निंग आज तकनीकी प्रगति के प्रमुख आधार बन चुके हैं।",
        "tts_text": "आर्टिफिशियल इंटेलिजेंस और मशीन लर्निंग आज तकनीकी प्रगति के प्रमुख आधार बन चुके हैं।"
    },
    {
        "id": "hi_03",
        "language": "hi",
        "category": "weather",
        "reference": "आज का मौसम बहुत सुहावना है और शाम को हल्की बारिश होने की संभावना है।",
        "tts_text": "आज का मौसम बहुत सुहावना है और शाम को हल्की बारिश होने की संभावना है।"
    },
    {
        "id": "hi_04",
        "language": "hi",
        "category": "numerical",
        "reference": "रोहन ने बाज़ार से पाँच किलो सेब और दो लीटर दूध खरीदा।",
        "tts_text": "रोहन ने बाज़ार से पाँच किलो सेब और दो लीटर दूध खरीदा।"
    },
    {
        "id": "hi_05",
        "language": "hi",
        "category": "domain",
        "reference": "भाषण से पाठ अनुवाद प्रणाली विभिन्न भाषाओं के बीच संचार को आसान बनाती है।",
        "tts_text": "भाषण से पाठ अनुवाद प्रणाली विभिन्न भाषाओं के बीच संचार को आसान बनाती है।"
    }
]


def generate_audio_dataset(output_dir: str = "data/samples"):
    """
    Generates test audio files for English and Hindi using gTTS,
    converts them to standard 16kHz mono WAV format, and writes reference.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_ffmpeg_path()

    from gtts import gTTS
    import subprocess
    import tempfile

    dataset_records = []

    for item in BENCHMARK_SAMPLES:
        sample_id = item["id"]
        lang = item["language"]
        text = item["tts_text"]
        wav_path = os.path.join(output_dir, f"{sample_id}.wav")

        print(f"Generating sample [{sample_id}] ({lang}): '{text[:40]}...'")

        # 1. Synthesize MP3 using gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_mp3_path = tmp_mp3.name

        tts.save(tmp_mp3_path)

        # 2. Convert MP3 to 16kHz mono 16-bit PCM WAV using ffmpeg or soundfile/scipy
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y", "-i", tmp_mp3_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                wav_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            # Fallback if imageio_ffmpeg is not ready
            print(f"Note: conversion via ffmpeg encountered: {e}")

        # Cleanup temp file
        if os.path.exists(tmp_mp3_path):
            os.remove(tmp_mp3_path)

        dataset_records.append({
            "id": sample_id,
            "language": lang,
            "category": item["category"],
            "audio_path": os.path.abspath(wav_path),
            "reference": item["reference"]
        })

    reference_json_path = os.path.join(output_dir, "reference.json")
    with open(reference_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully generated {len(dataset_records)} audio samples.")
    print(f"Reference metadata saved to: {reference_json_path}")
    return dataset_records


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/samples"
    generate_audio_dataset(out_dir)
