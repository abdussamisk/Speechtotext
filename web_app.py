"""
Streamlit Web Application for Multilingual Speech-to-Text.
Features:
 - Audio file upload and sample selection (English & Hindi)
 - Whisper transcription with language detection / forcing
 - Audio playback
 - WER & CER calculation against reference text
 - Alignment and error metrics breakdown (Substitutions, Deletions, Insertions)
 - Benchmark summary visualization
"""

import os
import json
import tempfile
import streamlit as st

from src.transcriber import WhisperTranscriber
from src.evaluator import compute_wer
from src.normalizer import normalize_text


st.set_page_config(
    page_title="Multilingual Speech-to-Text System",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Multilingual Speech-to-Text System")
st.markdown(
    "Automated Speech Recognition (ASR) system powered by **OpenAI Whisper**. "
    "Supports multilingual transcription, language identification, and **Word Error Rate (WER)** evaluation."
)

# Sidebar settings
st.sidebar.header("⚙️ Model Configuration")
model_choice = st.sidebar.selectbox(
    "Select Whisper Model:",
    ["tiny", "base", "small"],
    index=1,
    help="Tiny is fastest; Base balances accuracy and speed; Small yields higher accuracy."
)

device_choice = st.sidebar.selectbox(
    "Execution Device:",
    ["auto", "cuda", "cpu"],
    index=0
)
device_arg = None if device_choice == "auto" else device_choice

task_choice = st.sidebar.radio(
    "Task:",
    ["transcribe", "translate"],
    help="'transcribe' preserves spoken language; 'translate' translates speech into English."
)

# Load transcriber with caching
@st.cache_resource
def get_transcriber(model_name: str, device: str):
    return WhisperTranscriber(model_name=model_name, device=device)

try:
    with st.spinner(f"Loading Whisper '{model_choice}' model..."):
        transcriber = get_transcriber(model_choice, device_arg)
    st.sidebar.success(f"Loaded '{model_choice}' on {transcriber.device.upper()}")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    transcriber = None

# Tabs for workflow
tab1, tab2, tab3 = st.tabs(["🎯 Live Transcription & WER Eval", "📊 Benchmark Dashboard", "ℹ️ System Info & Methodology"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Audio Source")
        source_mode = st.radio("Choose Input Type:", ["Select Benchmark Sample", "Upload Custom Audio"])

        selected_audio_path = None
        default_ref = ""
        selected_lang = None

        if source_mode == "Select Benchmark Sample":
            ref_path = "data/samples/reference.json"
            if os.path.exists(ref_path):
                with open(ref_path, "r", encoding="utf-8") as f:
                    samples = json.load(f)
                options = {f"[{s['language'].upper()}] {s['id']} - {s['reference'][:45]}...": s for s in samples}
                chosen_label = st.selectbox("Choose Sample:", list(options.keys()))
                chosen_sample = options[chosen_label]
                selected_audio_path = chosen_sample["audio_path"]
                default_ref = chosen_sample["reference"]
                selected_lang = chosen_sample["language"]
            else:
                st.warning("Benchmark samples not generated yet. Run scripts/prepare_samples.py first.")
        else:
            uploaded_file = st.file_uploader("Upload Audio (WAV, MP3, M4A, FLAC):", type=["wav", "mp3", "m4a", "flac"])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                    tmp.write(uploaded_file.read())
                    selected_audio_path = tmp.name

        if selected_audio_path and os.path.exists(selected_audio_path):
            st.audio(selected_audio_path)

        lang_options = {"Auto-detect": None, "English (en)": "en", "Hindi (hi)": "hi", "Spanish (es)": "es", "French (fr)": "fr"}
        default_idx = 0
        if selected_lang == "en":
            default_idx = 1
        elif selected_lang == "hi":
            default_idx = 2
        lang_selection = st.selectbox("Spoken Language:", list(lang_options.keys()), index=default_idx)
        lang_code = lang_options[lang_selection]

        transcribe_button = st.button("🚀 Transcribe Audio", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Transcription Result")
        if transcribe_button and selected_audio_path and transcriber:
            with st.spinner("Processing speech recognition..."):
                result = transcriber.transcribe(
                    selected_audio_path,
                    language=lang_code,
                    task=task_choice
                )
                st.session_state["last_result"] = result
                st.session_state["last_audio"] = selected_audio_path

        if "last_result" in st.session_state:
            res = st.session_state["last_result"]
            st.success(f"Recognized Language: **{res['language'].upper()}** | Latency: **{res['latency_sec']}s** | Duration: **{res['audio_duration_sec']}s** (RTF: **{res['real_time_factor']}x**)")
            st.text_area("Hypothesis Transcript:", value=res["text"], height=130)

            # WER Evaluation Section
            st.markdown("---")
            st.subheader("3. Word Error Rate (WER) Evaluation")
            ref_input = st.text_area("Reference (Ground Truth) Transcript:", value=default_ref, height=90)

            if st.button("📈 Compute WER & Error Breakdown"):
                if ref_input.strip():
                    eval_out = compute_wer(
                        reference=ref_input,
                        hypothesis=res["text"],
                        language=res.get("language", "en")
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Word Error Rate (WER)", f"{eval_out.wer * 100:.2f}%")
                    m2.metric("Char Error Rate (CER)", f"{eval_out.cer * 100:.2f}%")
                    m3.metric("Raw WER (No Norm)", f"{eval_out.raw_wer * 100:.2f}%")
                    m4.metric("Reference Words", eval_out.reference_words)

                    st.markdown("#### Detailed Operations Breakdown")
                    st.write(f"- **Substitutions (S):** {eval_out.substitutions}")
                    st.write(f"- **Deletions (D):** {eval_out.deletions}")
                    st.write(f"- **Insertions (I):** {eval_out.insertions}")
                    st.write(f"- **Correct Hits (H):** {eval_out.hits}")

                    st.markdown("#### Normalized Comparison")
                    st.code(f"Ref : {eval_out.normalized_reference}\nHyp : {eval_out.normalized_hypothesis}")
                else:
                    st.warning("Please enter a reference transcript to evaluate WER.")

with tab2:
    st.subheader("📊 Systematic Benchmark Results")
    exp_report_path = "data/benchmarks/experiments_report.json"
    if os.path.exists(exp_report_path):
        with open(exp_report_path, "r", encoding="utf-8") as f:
            exp_data = json.load(f)

        st.markdown(f"**Report Timestamp:** {exp_data.get('timestamp')}")
        st.markdown("### Model Comparison (Tiny vs Base)")
        models_data = exp_data.get("model_benchmarks", {})
        table_rows = []
        for m, rep in models_data.items():
            table_rows.append({
                "Model": m,
                "Device": rep.get("device"),
                "Overall WER": f"{rep.get('overall_wer', 0)*100:.2f}%",
                "English WER": f"{rep.get('english_metrics', {}).get('corpus_wer', 0)*100:.2f}%",
                "Hindi WER": f"{rep.get('hindi_metrics', {}).get('corpus_wer', 0)*100:.2f}%",
                "Overall CER": f"{rep.get('overall_cer', 0)*100:.2f}%",
                "RTF": rep.get("overall_rtf"),
                "Total Latency": f"{rep.get('total_inference_time_sec')}s"
            })
        st.dataframe(table_rows, use_container_width=True)

        st.markdown("### Noise Robustness (Gaussian Noise at varying SNR)")
        noise_data = exp_data.get("noise_robustness", {})
        noise_rows = []
        for cond, stats in noise_data.items():
            noise_rows.append({
                "Condition": cond,
                "Overall WER": f"{stats.get('overall_wer', 0)*100:.2f}%",
                "English WER": f"{stats.get('english_wer', 0)*100:.2f}%",
                "Hindi WER": f"{stats.get('hindi_wer', 0)*100:.2f}%"
            })
        st.dataframe(noise_rows, use_container_width=True)
    else:
        st.info("Run `python scripts/run_experiments.py` to generate the complete benchmark report.")

with tab3:
    st.subheader("Methodology and Architectural Details")
    st.markdown("""
    ### Word Error Rate (WER) Formula
    $$WER = \\frac{S + D + I}{N} = \\frac{\\text{Substitutions} + \\text{Deletions} + \\text{Insertions}}{\\text{Total Reference Words}}$$
    
    ### Character Error Rate (CER) Formula
    $$CER = \\frac{S_c + D_c + I_c}{N_c}$$
    
    ### Multilingual Considerations:
    1. **Text Normalization:**
       - Devanagari script features punctuation (Danda `।`, double danda `॥`) and special glyphs.
       - Case insensitivity does not apply to Indic languages.
       - Devanagari numbers (०-९) need mapping to standard Arabic digits (0-9).
    2. **Character Error Rate (CER) Importance:**
       - In phonetic and compound-rich languages like Hindi, a single minor matra variance will penalize an entire word in WER. CER provides an essential fine-grained metric.
    """)
