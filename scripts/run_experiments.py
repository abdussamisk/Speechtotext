"""
End-to-End Experiment Execution Script.
Runs comprehensive tests on English and Hindi audio samples,
evaluates Word Error Rate (WER) and Character Error Rate (CER),
compares model architectures (e.g. tiny vs base), and assesses noise robustness.
"""

import os
import sys
import json
import time

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcriber import WhisperTranscriber
from src.evaluator import compute_wer, compute_aggregate_wer
from src.benchmark import BenchmarkRunner
from scripts.prepare_samples import generate_audio_dataset


def run_full_suite(
    dataset_path: str = "data/samples/reference.json",
    models: list = None,
    output_dir: str = "data/benchmarks"
):
    models = models or ["tiny", "base"]
    os.makedirs(output_dir, exist_ok=True)

    # 1. Ensure dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset '{dataset_path}' not found. Generating audio samples now...")
        generate_audio_dataset(os.path.dirname(dataset_path))

    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} benchmark samples from {dataset_path}")

    runner = BenchmarkRunner(models=models)
    experiment_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated": models,
        "sample_count": len(samples),
        "model_benchmarks": {},
        "noise_robustness": {}
    }

    # 2. Benchmark each model
    for model_name in models:
        print(f"\n==========================================")
        print(f"Running Benchmark: Whisper [{model_name}]")
        print(f"==========================================")
        report = runner.evaluate_dataset(samples, model_name=model_name)
        experiment_results["model_benchmarks"][model_name] = report

        print(f"--> Overall WER : {report['overall_wer'] * 100:.2f}%")
        print(f"--> English WER : {report['english_metrics']['corpus_wer'] * 100:.2f}% (CER: {report['english_metrics']['corpus_cer'] * 100:.2f}%)")
        print(f"--> Hindi WER   : {report['hindi_metrics']['corpus_wer'] * 100:.2f}% (CER: {report['hindi_metrics']['corpus_cer'] * 100:.2f}%)")
        print(f"--> Mean RTF    : {report['overall_rtf']:.3f}x")

    # 3. Noise robustness experiment with base model
    print(f"\n==========================================")
    print(f"Running Noise Robustness Experiment (base model)")
    print(f"==========================================")
    noise_rep = runner.run_noise_robustness_experiment(
        samples,
        model_name="base",
        snr_levels=[20.0, 10.0, 0.0],
        output_dir=os.path.join(output_dir, "noisy_samples")
    )
    experiment_results["noise_robustness"] = noise_rep

    for condition, stats in noise_rep.items():
        print(f"Condition: {condition:<12} | Overall WER: {stats['overall_wer'] * 100:.2f}% | EN: {stats['english_wer'] * 100:.2f}% | HI: {stats['hindi_wer'] * 100:.2f}%")

    # 4. Save results to JSON
    report_file = os.path.join(output_dir, "experiments_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Full experiment report saved to: {report_file}")
    return experiment_results


if __name__ == "__main__":
    run_full_suite()
