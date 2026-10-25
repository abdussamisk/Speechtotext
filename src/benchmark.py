"""
Benchmark Runner.
Automates evaluation across audio samples, languages (English & Hindi),
model architectures (tiny, base, etc.), and noise conditions.
Generates comprehensive benchmark metrics and reports.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from src.transcriber import WhisperTranscriber
from src.evaluator import compute_wer, compute_aggregate_wer
from src.audio_utils import load_audio_wav, save_audio_wav, add_gaussian_noise


class BenchmarkRunner:
    """
    Executes systematic ASR evaluation experiments.
    """

    def __init__(self, models: Optional[List[str]] = None, device: Optional[str] = None):
        self.models = models or ["tiny", "base"]
        self.device = device

    def evaluate_dataset(
        self,
        samples: List[Dict[str, str]],
        model_name: str,
        transcriber: Optional[WhisperTranscriber] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a set of audio samples against ground truth references.
        
        Args:
            samples: List of dicts, each with keys:
                     'audio_path', 'reference', 'language', 'id', 'category'
            model_name: Whisper model size
            transcriber: Pre-loaded WhisperTranscriber instance
            
        Returns:
            Dict containing detailed and aggregate benchmark statistics.
        """
        if transcriber is None:
            transcriber = WhisperTranscriber(model_name=model_name, device=self.device)

        results = []
        total_audio_duration = 0.0
        total_inference_time = 0.0

        for item in samples:
            audio_path = item["audio_path"]
            reference = item["reference"]
            lang = item.get("language", "en")
            sample_id = item.get("id", os.path.basename(audio_path))
            category = item.get("category", "clean")

            transcription_out = transcriber.transcribe(audio_path, language=lang)
            hyp = transcription_out["text"]

            eval_res = compute_wer(reference, hyp, language=lang, normalize=True)

            total_audio_duration += transcription_out["audio_duration_sec"]
            total_inference_time += transcription_out["latency_sec"]

            record = {
                "id": sample_id,
                "language": lang,
                "category": category,
                "audio_path": audio_path,
                "reference": reference,
                "hypothesis": hyp,
                "normalized_reference": eval_res.normalized_reference,
                "normalized_hypothesis": eval_res.normalized_hypothesis,
                "wer": eval_res.wer,
                "raw_wer": eval_res.raw_wer,
                "cer": eval_res.cer,
                "substitutions": eval_res.substitutions,
                "deletions": eval_res.deletions,
                "insertions": eval_res.insertions,
                "hits": eval_res.hits,
                "reference_words": eval_res.reference_words,
                "latency_sec": transcription_out["latency_sec"],
                "audio_duration_sec": transcription_out["audio_duration_sec"],
                "rtf": transcription_out["real_time_factor"]
            }
            results.append(record)

        # Aggregate metrics
        en_records = [r for r in results if r["language"] == "en"]
        hi_records = [r for r in results if r["language"] == "hi"]

        def calc_aggregate(subset, lang):
            if not subset:
                return {}
            refs = [r["reference"] for r in subset]
            hyps = [r["hypothesis"] for r in subset]
            return compute_aggregate_wer(refs, hyps, language=lang, normalize=True)

        en_agg = calc_aggregate(en_records, "en")
        hi_agg = calc_aggregate(hi_records, "hi")

        all_refs = [r["reference"] for r in results]
        all_hyps = [r["hypothesis"] for r in results]
        overall_agg = compute_aggregate_wer(all_refs, all_hyps, normalize=True)

        overall_rtf = (
            total_inference_time / total_audio_duration
            if total_audio_duration > 0 else 0.0
        )

        return {
            "model": model_name,
            "device": transcriber.device,
            "total_samples": len(results),
            "total_audio_duration_sec": round(total_audio_duration, 2),
            "total_inference_time_sec": round(total_inference_time, 2),
            "overall_rtf": round(overall_rtf, 3),
            "overall_wer": overall_agg.get("corpus_wer", 0.0),
            "overall_cer": overall_agg.get("corpus_cer", 0.0),
            "english_metrics": {
                "sample_count": len(en_records),
                "corpus_wer": en_agg.get("corpus_wer", 0.0),
                "corpus_cer": en_agg.get("corpus_cer", 0.0)
            },
            "hindi_metrics": {
                "sample_count": len(hi_records),
                "corpus_wer": hi_agg.get("corpus_wer", 0.0),
                "corpus_cer": hi_agg.get("corpus_cer", 0.0)
            },
            "sample_details": results
        }

    def run_noise_robustness_experiment(
        self,
        samples: List[Dict[str, str]],
        model_name: str = "base",
        snr_levels: Optional[List[float]] = None,
        output_dir: str = "data/noisy_samples"
    ) -> Dict[str, Any]:
        """
        Tests model performance across various SNR degradation levels.
        """
        snr_levels = snr_levels or [20.0, 10.0, 0.0]
        os.makedirs(output_dir, exist_ok=True)

        transcriber = WhisperTranscriber(model_name=model_name, device=self.device)
        noise_results = {}

        # First baseline (clean)
        clean_res = self.evaluate_dataset(samples, model_name=model_name, transcriber=transcriber)
        noise_results["clean"] = {
            "overall_wer": clean_res["overall_wer"],
            "english_wer": clean_res["english_metrics"].get("corpus_wer", 0.0),
            "hindi_wer": clean_res["hindi_metrics"].get("corpus_wer", 0.0),
        }

        for snr in snr_levels:
            noisy_samples = []
            for item in samples:
                audio, sr = load_audio_wav(item["audio_path"])
                noisy_audio = add_gaussian_noise(audio, snr_db=snr)
                
                base_name = os.path.splitext(os.path.basename(item["audio_path"]))[0]
                noisy_filename = f"{base_name}_snr_{int(snr)}db.wav"
                noisy_path = os.path.join(output_dir, noisy_filename)
                save_audio_wav(noisy_path, noisy_audio, sr=sr)

                noisy_item = item.copy()
                noisy_item["audio_path"] = noisy_path
                noisy_item["category"] = f"noisy_snr_{snr}db"
                noisy_samples.append(noisy_item)

            eval_res = self.evaluate_dataset(noisy_samples, model_name=model_name, transcriber=transcriber)
            noise_results[f"snr_{int(snr)}db"] = {
                "overall_wer": eval_res["overall_wer"],
                "english_wer": eval_res["english_metrics"].get("corpus_wer", 0.0),
                "hindi_wer": eval_res["hindi_metrics"].get("corpus_wer", 0.0)
            }

        return noise_results
