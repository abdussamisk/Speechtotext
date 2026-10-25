#!/usr/bin/env python3
"""
Command Line Interface (CLI) for Multilingual Speech-to-Text System.
Provides commands for:
  - transcribe: Transcribe an audio file with language detection or specified language.
  - evaluate: Evaluate transcription accuracy (WER / CER) against reference text.
  - benchmark: Run systematic evaluation on a test dataset across models and noise levels.
  - detect: Identify spoken language in audio.
"""

import argparse
import sys
import json
import os
from tabulate import tabulate

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.transcriber import WhisperTranscriber
from src.evaluator import compute_wer, compute_aggregate_wer
from src.benchmark import BenchmarkRunner
from src.normalizer import normalize_text


def cmd_transcribe(args):
    """Handles audio transcription."""
    print(f"\n--- Transcribing: {args.audio} ---")
    transcriber = WhisperTranscriber(model_name=args.model, device=args.device)
    result = transcriber.transcribe(
        args.audio,
        language=args.language,
        task=args.task
    )

    print("\n[RESULT]")
    print(f"Detected/Selected Language : {result['language']}")
    print(f"Audio Duration             : {result['audio_duration_sec']}s")
    print(f"Latency                    : {result['latency_sec']}s (RTF: {result['real_time_factor']})")
    print("\n[TRANSCRIPT]")
    print(result['text'])

    if args.timestamps and result.get("segments"):
        print("\n[TIMESTAMPED SEGMENTS]")
        table = [[s["id"], f"{s['start']}s -> {s['end']}s", s["text"]] for s in result["segments"]]
        print(tabulate(table, headers=["#", "Timestamp", "Segment Text"], tablefmt="grid"))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved detailed output to: {args.output}")


def cmd_detect(args):
    """Detects spoken language from audio file."""
    print(f"\n--- Detecting Language: {args.audio} ---")
    transcriber = WhisperTranscriber(model_name=args.model, device=args.device)
    detection = transcriber.detect_language(args.audio)

    print(f"Primary Detected Language : {detection['detected_language']} (Confidence: {detection['confidence'] * 100:.1f}%)")
    print("\nTop 5 Candidates:")
    candidates_table = [[lang, f"{conf * 100:.2f}%"] for lang, conf in detection["top_candidates"]]
    print(tabulate(candidates_table, headers=["Language Code", "Probability"], tablefmt="fancy_grid"))


def cmd_evaluate(args):
    """Evaluates reference text vs hypothesis text or runs evaluation on an audio file."""
    if args.audio and args.reference:
        print(f"\n--- Transcribing & Evaluating: {args.audio} ---")
        transcriber = WhisperTranscriber(model_name=args.model, device=args.device)
        res = transcriber.transcribe(args.audio, language=args.language)
        hypothesis = res["text"]
        reference = args.reference
    elif args.reference and args.hypothesis:
        reference = args.reference
        hypothesis = args.hypothesis
    else:
        print("Error: Must provide either (--audio and --reference) or (--reference and --hypothesis).")
        sys.exit(1)

    eval_result = compute_wer(
        reference=reference,
        hypothesis=hypothesis,
        language=args.language or "en",
        normalize=not args.no_norm
    )

    print("\n================== EVALUATION REPORT ==================")
    print(f"Language              : {args.language}")
    print(f"Reference             : {eval_result.reference}")
    print(f"Hypothesis            : {eval_result.hypothesis}")
    print(f"Normalized Reference  : {eval_result.normalized_reference}")
    print(f"Normalized Hypothesis : {eval_result.normalized_hypothesis}")
    print("-------------------------------------------------------")
    print(f"Word Error Rate (WER) : {eval_result.wer * 100:.2f}%")
    print(f"Raw WER (No Norm)     : {eval_result.raw_wer * 100:.2f}%")
    print(f"Char Error Rate (CER) : {eval_result.cer * 100:.2f}%")
    print(f"Substitutions         : {eval_result.substitutions}")
    print(f"Deletions             : {eval_result.deletions}")
    print(f"Insertions            : {eval_result.insertions}")
    print(f"Hits / Correct Words  : {eval_result.hits}")
    print(f"Total Ref Words       : {eval_result.reference_words}")
    print("=======================================================")


def cmd_benchmark(args):
    """Runs systematic benchmark across test suite."""
    print(f"\n--- Running Benchmark on Dataset: {args.dataset} ---")
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)

    with open(args.dataset, "r", encoding="utf-8") as f:
        samples = json.load(f)

    runner = BenchmarkRunner(models=args.models.split(","), device=args.device)
    
    overall_summary = []
    full_reports = {}

    for model_name in args.models.split(","):
        print(f"\nEvaluating Model: {model_name}...")
        report = runner.evaluate_dataset(samples, model_name=model_name)
        full_reports[model_name] = report

        overall_summary.append([
            model_name,
            f"{report['overall_wer'] * 100:.2f}%",
            f"{report['english_metrics'].get('corpus_wer', 0) * 100:.2f}%",
            f"{report['hindi_metrics'].get('corpus_wer', 0) * 100:.2f}%",
            f"{report['overall_cer'] * 100:.2f}%",
            f"{report['overall_rtf']:.3f}",
            f"{report['total_inference_time_sec']:.2f}s"
        ])

    print("\n======================= BENCHMARK SUMMARY =======================")
    headers = ["Model", "Overall WER", "English WER", "Hindi WER", "Overall CER", "RTF", "Total Latency"]
    print(tabulate(overall_summary, headers=headers, tablefmt="github"))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(full_reports, f, ensure_ascii=False, indent=2)
        print(f"\nFull benchmark report exported to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual Speech-to-Text System using OpenAI Whisper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Transcribe subcommand
    p_transcribe = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    p_transcribe.add_argument("audio", type=str, help="Path to audio file")
    p_transcribe.add_argument("--model", type=str, default="base", help="Whisper model (tiny, base, small, medium, large)")
    p_transcribe.add_argument("--language", type=str, default=None, help="Language code (e.g. en, hi). Auto-detected if None")
    p_transcribe.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Task")
    p_transcribe.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    p_transcribe.add_argument("--timestamps", action="store_true", help="Display segment timestamps")
    p_transcribe.add_argument("--output", type=str, default=None, help="JSON output file path")
    p_transcribe.set_defaults(func=cmd_transcribe)

    # Detect subcommand
    p_detect = subparsers.add_parser("detect", help="Detect audio language")
    p_detect.add_argument("audio", type=str, help="Path to audio file")
    p_detect.add_argument("--model", type=str, default="base", help="Whisper model")
    p_detect.add_argument("--device", type=str, default=None, help="Device")
    p_detect.set_defaults(func=cmd_detect)

    # Evaluate subcommand
    p_eval = subparsers.add_parser("evaluate", help="Compute Word Error Rate (WER)")
    p_eval.add_argument("--audio", type=str, default=None, help="Audio file path")
    p_eval.add_argument("--reference", type=str, required=True, help="Ground truth reference transcript")
    p_eval.add_argument("--hypothesis", type=str, default=None, help="Hypothesis transcript")
    p_eval.add_argument("--model", type=str, default="base", help="Model to transcribe audio if --audio given")
    p_eval.add_argument("--language", type=str, default="en", help="Language code (en, hi)")
    p_eval.add_argument("--device", type=str, default=None, help="Device")
    p_eval.add_argument("--no-norm", action="store_true", help="Disable text normalization")
    p_eval.set_defaults(func=cmd_evaluate)

    # Benchmark subcommand
    p_bench = subparsers.add_parser("benchmark", help="Run multi-model benchmark on dataset")
    p_bench.add_argument("--dataset", type=str, default="data/samples/reference.json", help="Path to reference JSON")
    p_bench.add_argument("--models", type=str, default="tiny,base", help="Comma-separated models (e.g. tiny,base,small)")
    p_bench.add_argument("--device", type=str, default=None, help="Device")
    p_bench.add_argument("--output", type=str, default="data/benchmarks/benchmark_results.json", help="Output path")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
