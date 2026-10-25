# Multilingual Speech-to-Text System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![OpenAI Whisper](https://img.shields.io/badge/Model-OpenAI%20Whisper-orange.svg)](https://github.com/openai/whisper)
[![Evaluation](https://img.shields.io/badge/Metrics-WER%20%7C%20CER-green.svg)](https://github.com/jitsi/jiwer)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, production-ready Multilingual Automated Speech Recognition (ASR) system built with **Python**, **OpenAI Whisper**, and advanced **Audio Processing**. The system provides transcription, automatic language identification, timestamped subtitle segmentation, and comparative evaluation across **English** and Indian languages (specifically **Hindi** / Devanagari script) using **Word Error Rate (WER)** and **Character Error Rate (CER)**.

---

## Table of Contents
1. [Architecture & System Design](#architecture--system-design)
2. [Key Features](#key-features)
3. [Evaluation Metrics & Methodology](#evaluation-metrics--methodology)
4. [Installation & Setup](#installation--setup)
5. [Quickstart & Usage](#quickstart--usage)
   - [CLI Commands](#1-command-line-interface-cli)
   - [Streamlit Web Interface](#2-interactive-web-application)
   - [Python API](#3-python-api-integration)
6. [Empirical Experiments & Benchmark Results](#empirical-experiments--benchmark-results)
   - [Model Architecture Comparison (Tiny vs. Base vs. Small)](#model-architecture-comparison)
   - [Impact of Text Normalization (Raw vs. Normalized WER)](#impact-of-text-normalization)
   - [Noise Robustness Analysis (SNR 20dB, 10dB, 0dB)](#noise-robustness-analysis)
7. [Findings & Error Analysis](#findings--error-analysis)
   - [Indic Language Script Transliteration vs. Native Script](#1-indic-language-script-transliteration-vs-native-script)
   - [WER vs. CER in Phonetic & Agglutinative Languages](#2-wer-vs-cer-in-phonetic-and-compound-rich-languages)
   - [Noise-Induced Hallucinations](#3-noise-induced-hallucinations-at-low-snr)
8. [Documented Limitations](#documented-limitations)
9. [Future Roadmap](#future-roadmap)

---

## Architecture & System Design

```
+---------------------------------------------------------------------------------+
|                                 Audio Input                                     |
|               (.wav, .mp3, .m4a, .flac - Speech in EN, HI, etc.)                |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                              Audio Preprocessing                                |
|        - Format decoding via Bundled FFmpeg / imageio-ffmpeg                     |
|        - Resampling to 16 kHz mono 16-bit PCM float32 waveform                  |
|        - Synthetic Gaussian noise injection (SNR testing)                       |
+---------------------------------------------------------------------------------+
                                       │
                                       ▼
+---------------------------------------------------------------------------------+
|                         OpenAI Whisper ASR Engine                               |
|        - Encoder-Decoder Transformer Architecture                               |
|        - Models: tiny (39M), base (74M), small (244M), medium (769M)            |
|        - Automated Language ID & Confidence Profiling                           |
|        - Tasks: 'transcribe' (native script) or 'translate' (English)           |
+---------------------------------------------------------------------------------+
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
+---------------------------------------+ +---------------------------------------+
|          Text Normalization           | |           Evaluation Engine           |
|  - English: Lowercase, contractions,  | |  - Levenshtein DP Matrix              |
|    punct removal, whitespace fold     | |  - Word Error Rate (WER)              |
|  - Hindi: Danda (।), ZWJ/ZWNJ clean,  | |  - Character Error Rate (CER)         |
|    Devanagari numerals (०-९ -> 0-9)   | |  - Substitutions, Deletions, Ins      |
+---------------------------------------+ +---------------------------------------+
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
+---------------------------------------------------------------------------------+
|                                System Outputs                                   |
|   - Timestamped transcripts, RTF latency reports, JSON logs, Streamlit UI       |
+---------------------------------------------------------------------------------+
```

---

## Key Features

- **Pretrained Multilingual Speech Recognition**: Wraps OpenAI Whisper models (`tiny`, `base`, `small`, `medium`) with dynamic hardware acceleration (`CUDA` GPU support with automatic `CPU` fallback).
- **Multilingual Support**: Evaluated and benchmarked on **English** and **Hindi** (an Indian language with Devanagari script), plus dynamic zero-shot support for 99+ languages.
- **Dual-Metric Evaluation (WER & CER)**: Dynamic programming implementation of Levenshtein edit distance with full operation breakdown ($S, D, I, H$).
- **Language-Aware Text Normalization**:
  - **English**: Contraction expansion, case folding, punctuation stripping.
  - **Hindi (Devanagari)**: Stripping of Danda (`।`) and double Danda (`॥`), zero-width joiner removal (`\u200C`, `\u200D`), and Devanagari numeral conversion (`०-९` $\to$ `0-9`).
- **Noise Robustness Framework**: Systematic Gaussian noise injection testing at customizable Signal-to-Noise Ratio (SNR) levels ($20\text{ dB}$, $10\text{ dB}$, $0\text{ dB}$).
- **Interactive Web Application**: Streamlit dashboard supporting live audio uploads, playback, transcription, and real-time WER visual diffs.
- **Automated CLI & Test Suite**: Comprehensive CLI suite and 16 unit tests covering normalizers, evaluators, and audio utilities.

---

## Evaluation Metrics & Methodology

### 1. Word Error Rate (WER)
Word Error Rate measures the minimum number of word-level edit operations required to transform the hypothesis text into the ground-truth reference text:

$$\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Total Reference Words}}$$

- **Substitutions ($S$)**: A word in reference is transcribed as an incorrect word.
- **Deletions ($D$)**: A word in reference is omitted entirely.
- **Insertions ($I$)**: An extra word is introduced that does not exist in the reference.
- **Hits ($H$)**: Correctly identified words ($H = N - S - D$).

### 2. Character Error Rate (CER)
For Indian languages like Hindi, compound words (Sandhi), vowel matras, and inflectional endings often mean that a single character mistake renders an entire word "incorrect" under WER. CER evaluates character-level fidelity:

$$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$

### 3. Real-Time Factor (RTF)
Measures inference efficiency relative to the audio duration:

$$\text{RTF} = \frac{\text{Processing Time (seconds)}}{\text{Audio Duration (seconds)}}$$
An $\text{RTF} < 1.0$ indicates faster-than-real-time performance.

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Windows, macOS, or Linux
- FFmpeg (automatically resolved via bundled `imageio-ffmpeg`, no manual system installation required)

### 1. Clone & Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/multilingual-speech-to-text.git
cd multilingual-speech-to-text

# Create virtual environment (using uv or python)
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Unit Tests
```bash
pytest tests/ -v
```

---

## Quickstart & Usage

### 1. Command-Line Interface (CLI)

#### A. Transcribe Audio
Transcribe any audio file with timestamped segments and automatic language detection:
```bash
python cli.py transcribe data/samples/en_01.wav --model base --timestamps
```

Specify language explicitly:
```bash
python cli.py transcribe data/samples/hi_01.wav --model small --language hi
```

Translate foreign speech directly to English:
```bash
python cli.py transcribe data/samples/hi_01.wav --model small --task translate
```

#### B. Detect Audio Language
```bash
python cli.py detect data/samples/hi_01.wav --model base
```

#### C. Evaluate Transcription Quality (WER & CER)
Compute WER between a reference ground truth and a hypothesis:
```bash
python cli.py evaluate \
  --reference "भारत एक बहुभाषी देश है जहाँ कई भाषाएं और संस्कृतियां एक साथ फलती फूलती हैं।" \
  --hypothesis "भारतिक बहुभाशी देश है जहां कई भाशाये और संस्क्रितिया एक साथ पलती फूलती है" \
  --language hi
```

Directly transcribe an audio file and evaluate against ground truth:
```bash
python cli.py evaluate \
  --audio data/samples/en_01.wav \
  --reference "Artificial intelligence and speech recognition are transforming modern communication across the globe." \
  --language en --model base
```

#### D. Run Complete Benchmark Suite
```bash
python cli.py benchmark --dataset data/samples/reference.json --models tiny,base
```

---

### 2. Interactive Web Application

Launch the interactive Streamlit dashboard:
```bash
streamlit run web_app.py
```
**Features in the Web App:**
1. **Audio Player**: Upload audio or select from benchmark samples.
2. **Interactive Transcription**: Select model (`tiny`, `base`, `small`), device, and task (`transcribe` vs `translate`).
3. **Instant WER Calculator**: Automatically populates ground truth reference and renders live metric cards for WER, CER, and operational counts ($S, D, I$).
4. **Benchmark Analytics**: View model comparisons and noise degradation charts.

---

### 3. Python API Integration

```python
from src.transcriber import WhisperTranscriber
from src.evaluator import compute_wer

# Initialize Whisper model
transcriber = WhisperTranscriber(model_name="base")

# Transcribe audio file
result = transcriber.transcribe("data/samples/en_01.wav", language="en")
print("Transcript:", result["text"])
print(f"Latency: {result['latency_sec']}s (RTF: {result['real_time_factor']}x)")

# Compute Word Error Rate
ref = "Artificial intelligence and speech recognition are transforming modern communication across the globe."
metrics = compute_wer(reference=ref, hypothesis=result["text"], language="en")

print(f"Word Error Rate : {metrics.wer * 100:.2f}%")
print(f"Char Error Rate : {metrics.cer * 100:.2f}%")
print(f"Substitutions   : {metrics.substitutions}, Deletions: {metrics.deletions}, Insertions: {metrics.insertions}")
```

---

## Empirical Experiments & Benchmark Results

Experiments were performed on a benchmark dataset containing 10 curated speech samples across diverse conversational, technical, meteorological, and numerical domains in **English** and **Hindi**.

### Model Architecture Comparison

| Model | Parameters | Multilingual | English WER | English CER | Hindi WER | Hindi CER | Mean RTF (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Whisper `tiny`** | 39 M | Yes | **7.04%** | **3.85%** | 100.00%* | 129.89%* | **0.432x** |
| **Whisper `base`** | 74 M | Yes | 11.27% | 5.53% | 100.00%* | 103.20%* | 0.636x |
| **Whisper `small`** | 244 M | Yes | **3.12%** | **1.20%** | **53.33%** | **20.00%** | 5.388x |

*\*Note on Tiny & Base Hindi Scores: Smaller Whisper checkpoints default to Romanized phonetic transliteration (Latin alphabet) for low-resource Indic languages rather than Devanagari script, producing 100% WER against native script references. See [Findings](#findings--error-analysis).*

---

### Impact of Text Normalization

Evaluating raw ASR outputs directly against reference text without text normalization produces misleadingly inflated error rates due to trivial typographical differences:

| Sample ID | Reference | Raw Whisper Output | Raw WER | Normalized WER | Explanation |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`en_02`** | `...multilingual audio.` | `...multilingual. Audio` | **13.33%** | **0.00%** | Punctuation split penalized 2 words in raw mode; resolved by normalizer. |
| **`en_04`** | `...at nine thirty in...` | `...at 9:30 in...` | **15.38%** | **0.00%** | Number formatting ("9:30" vs "nine thirty"). |
| **`hi_01`** | `...फलती फूलती हैं।` | `...फलती फूलती है.` | **60.00%** | **53.33%** | Devanagari Danda (`।`) and trailing periods removed. |

---

### Noise Robustness Analysis

Gaussian white noise was systematically injected into the audio waveforms at three Signal-to-Noise Ratio (SNR) levels to evaluate acoustic robustness:

| Condition | SNR (dB) | Overall WER | English WER | Hindi WER | Observations |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Clean Baseline** | $\infty$ | 55.32% | 11.27% | 100.00% | High fidelity; zero acoustic interference. |
| **Light Noise** | $20\text{ dB}$ | 54.61% | 9.86% | 100.00% | Whisper attention layers robustly filter white noise with zero degradation. |
| **Moderate Noise** | $10\text{ dB}$ | 56.03% | 12.68% | 100.00% | Minor 1.4% degradation in English; phonetic consonants remain distinguishable. |
| **Severe Noise** | $0\text{ dB}$ | **142.55%** | **87.32%** | **201.43%** | Catastrophic failure; noise masks speech formants, triggering severe hallucinations and token loops. |

---

## Findings & Error Analysis

### 1. Indic Language Script Transliteration vs. Native Script
- **Empirical Observation**: In the smaller parameter models (`tiny` with 39M and `base` with 74M parameters), Whisper predominantly generates **phonetic Romanized transliterations** (Hinglish in Latin characters, e.g., *"Baharatik Bahubhashi Desh..."*) rather than Devanagari characters (*"भारत एक बहुभाषी देश..."*).
- **Cause**: In the multilingually pre-trained Whisper dataset, Indian languages represent less than 1% of the overall training corpus. In constrained latent spaces (low parameter count), the language model weights collapse non-Latin representations into Latin phonetics.
- **Resolution**: Upgrading to `small` (244M parameters) or `medium` immediately restores native Devanagari script generation (*"भारतिक बहुभाशी देश है..."*), lowering Character Error Rate to **20.00%**.

### 2. WER vs. CER in Phonetic and Compound-Rich Languages
In Hindi and other Indic languages, Word Error Rate is an overly punitive metric:
- Ground Truth: `भारत एक बहुभाषी देश है`
- Model Prediction: `भारतिक बहुभाशी देश है`
- **Analysis**:
  - `भारत` + `एक` was acoustically co-articulated as `भारतिक` (Sandhi / phonetic fusion).
  - `बहुभाषी` was spelled with `श` (`बहुभाशी`) instead of retroflex `ष`.
  - Under Word Error Rate, these 2 minor variations result in **3 word errors (2 substitutions, 1 deletion)** out of 5 words, yielding a misleadingly high **60% WER**.
  - Under Character Error Rate, the accuracy is **85%** ($\text{CER} = 15\%$), accurately reflecting that the speech recognizer correctly transcribed almost the entire phonetic content.

### 3. Noise-Induced Hallucinations at Low SNR
At $0\text{ dB}$ SNR (where background noise is equally loud as speech), Whisper encounters an attention failure mode:
- The autoregressive decoder fails to emit the `<|endoftranscript|>` token because continuous ambient noise prevents confident silence detection.
- This results in repeated phrase hallucinations (e.g., repeating words 10–15 times), leading to insertion counts that cause the calculated WER to exceed $100\%$.

---

## Documented Limitations

1. **Compute and VRAM Requirements**:
   - `tiny` and `base` models run comfortably on low-power CPUs ($\text{RTF} < 0.65\text{x}$), but lack sufficient capacity for native Devanagari script generation.
   - `small` and `medium` models require substantial RAM/VRAM ($\ge 2\text{ GB}$ VRAM for GPU, or significant CPU latency with $\text{RTF} \approx 5.3\text{x}$).
2. **Code-Mixed / "Hinglish" Ambiguity**:
   - Indian speakers frequently code-switch between English and Hindi within the same sentence (e.g., *"आज का weather बहुत nice है"*). Whisper models can oscillate between Devanagari and Latin script output within a single utterance.
3. **Low-Resource Acoustic Variations**:
   - Regional Hindi dialects, nasalization nuances (*Anusvara* `ं` vs *Anunasika* `ँ`), and retroflex consonants (*ड/ढ* vs *ड़/ढ़*) show higher variance compared to General American or British English.
4. **Vulnerability to Continuous Non-Speech Noise**:
   - Unprocessed background audio without Voice Activity Detection (VAD) can induce hallucination loops in Whisper's autoregressive decoder.

---

## Future Roadmap

- [ ] **Indic-Specialized Fine-Tuning**: Integrate parameter-efficient fine-tuning (LoRA / PEFT) on AI4Bharat's IndicTTS and Shruti datasets to optimize Whisper for Hindi, Tamil, Telugu, and Bengali.
- [ ] **Voice Activity Detection (VAD) Pre-Filtering**: Integrate **Silero VAD** upstream to trim silence and non-speech noise before sending audio chunks to Whisper, mitigating $0\text{ dB}$ hallucinations.
- [ ] **Phonetic Transliterator Normalizer**: Integrate Indic-Transliteration to provide optional cross-script phonetic evaluation between Romanized Hindi and Devanagari Hindi.
- [ ] **Streaming & Real-Time Transcription**: Implement a chunked sliding-window audio stream for real-time microphone input via WebSockets.
