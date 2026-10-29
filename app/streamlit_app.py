from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import BOT_THRESHOLD, HUMAN_THRESHOLD, MODEL_PATH, RESULTS_DIR
from src.document_utils import extract_document_text
from src.predict import predict, predict_sentences, split_paragraphs


st.set_page_config(page_title="Signal - AI text analysis", page_icon="S", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
            --bg: #09110f;
            --panel: #111d1a;
            --panel-strong: #182723;
            --panel-soft: #20332e;
            --line: rgba(175, 202, 190, 0.22);
            --ink: #edf5f0;
            --muted: #b8cbc2;
            --green: #78e0b2;
            --green-deep: #47c18d;
            --blue: #91d9d0;
            --amber: #f3c66f;
            --red: #f29489;
            --shadow: rgba(1, 8, 6, 0.42);
    }
        .stApp { background: linear-gradient(180deg, #09110f 0%, #101c19 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
        header[data-testid="stHeader"] { background:rgba(9,17,15,.96); }
    .block-container { max-width:1160px; padding:2rem 2rem 3rem; }
        h1,h2,h3 { color:var(--ink); font-family:'Manrope',sans-serif; letter-spacing:0; }
    .brand-row { display:flex; align-items:center; gap:.65rem; color:var(--green); font-size:.78rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; }
    .brand-mark { display:inline-grid; place-items:center; width:1.8rem; height:1.8rem; border-radius:.6rem; background:linear-gradient(135deg, var(--green), var(--blue)); color:#07111a; font-family:'Manrope',sans-serif; font-size:1rem; letter-spacing:0; }
    .hero { padding:1rem 0 1.4rem; }
    .hero h1 { font-size:3.35rem; line-height:1.05; margin:.8rem 0 .7rem; }
    .hero p { color:var(--muted); max-width:690px; font-size:1.05rem; line-height:1.65; margin:0; }
    .side-card { background:rgba(17,29,26,0.96); border:1px solid var(--line); border-radius:8px; padding:1.15rem; box-shadow:0 7px 24px var(--shadow); }
    .side-card h3 { font-size:1.03rem; margin:0 0 .65rem; }
    .side-card p { color:var(--muted); font-size:.9rem; line-height:1.55; margin:.25rem 0 .7rem; }
    .eyebrow { color:var(--blue); font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.4rem; }
    [data-testid="stForm"] { background:rgba(17,29,26,0.96); border:1px solid var(--line); border-radius:8px; padding:1.25rem; box-shadow:0 8px 26px var(--shadow); }
    .input-title { font-family:'Manrope',sans-serif; font-size:1.12rem; font-weight:700; margin-bottom:.2rem; }
    .input-hint { color:var(--muted); font-size:.88rem; margin-bottom:.9rem; }
    .stTextArea textarea { background:#101a17; color:var(--ink); border:1px solid var(--line); border-radius:6px; font-size:.98rem; line-height:1.6; }
    .stTextArea textarea::placeholder { color:var(--muted); }
    .stTextArea textarea:focus { border-color:var(--blue); box-shadow:0 0 0 2px rgba(125,211,252,.18); }
    .stButton > button, [data-testid="stFormSubmitButton"] > button { background:linear-gradient(135deg, var(--green), var(--blue)); border:0; border-radius:6px; color:#07111a; font-weight:800; min-height:2.8rem; padding:0 1.3rem; }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { filter:brightness(1.06); border:0; color:#07111a; }
    .result-panel { background:rgba(17,29,26,0.96); border:1px solid var(--line); border-radius:8px; padding:1.2rem; margin:.65rem 0 .7rem; box-shadow:0 8px 26px var(--shadow); }
    .result-top { display:flex; align-items:center; justify-content:space-between; gap:1rem; }
    .result-label { color:var(--ink); font-family:'Manrope',sans-serif; font-size:1.55rem; font-weight:800; }
    .result-caption { color:var(--muted); font-size:.82rem; margin-top:.15rem; }
    .score-pill { border-radius:99px; padding:.45rem .8rem; font-size:.82rem; font-weight:700; white-space:nowrap; }
    .score-ai { background:rgba(252,165,165,.18); color:#fecaca; }
    .score-human { background:rgba(110,231,183,.16); color:#bbf7d0; }
    .score-unsure { background:rgba(251,191,36,.18); color:#fde68a; }
    .score-mixed { background:rgba(243,198,111,.18); color:#f8dfa5; }
    .sentence { background:rgba(32,51,46,0.88); border:1px solid var(--line); border-left:4px solid var(--amber); border-radius:6px; padding:.85rem 1rem; margin:.55rem 0; line-height:1.55; overflow-wrap:anywhere; color:var(--ink); }
    .sentence.ai { border-left-color:var(--red); background:rgba(102,43,37,0.2); }
    .sentence.human { border-left-color:var(--green); background:rgba(25,87,60,0.2); }
    .sentence small { display:block; color:var(--muted); font-size:.77rem; margin-top:.35rem; }
    .notice { color:#dfe9f7; background:rgba(17,24,39,0.92); border:1px solid var(--line); border-radius:13px; padding:.9rem 1rem; font-size:.86rem; line-height:1.55; }
    [data-testid="stSidebar"] { background:rgba(8,16,14,0.97); border-right:1px solid var(--line); }
    [data-testid="stMetric"] { background:rgba(17,29,26,0.96); border:1px solid var(--line); border-radius:6px; padding:.7rem .8rem; }
    [data-testid="stMetricValue"] { color:var(--ink) !important; }
    [data-testid="stMetricLabel"] { color:var(--muted) !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:transparent; border-color:var(--line); border-radius:8px; }
    .stAlert { border-radius:12px; border:1px solid var(--line); background:rgba(17,24,39,0.9); }
    footer { visibility:hidden; }
        @media (max-width:700px) {
            .block-container { padding:1.1rem 1rem 2rem; }
            .hero { padding:.5rem 0 1rem; }
            .hero h1 { font-size:2.5rem; }
            .result-top { align-items:flex-start; flex-direction:column; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_metrics() -> dict:
    path = RESULTS_DIR / "robust_evaluation.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def classify(probability: float) -> tuple[str, str]:
    if probability >= BOT_THRESHOLD:
        return "Likely AI", "ai"
    if probability <= HUMAN_THRESHOLD:
        return "Likely human", "human"
    return "Uncertain", "unsure"


def display_label(label: str) -> tuple[str, str]:
    return {
        "likely_ai": ("Likely AI", "ai"),
        "likely_human": ("Likely human", "human"),
        "uncertain": ("Uncertain", "unsure"),
    }.get(label, ("Uncertain", "unsure"))


with st.sidebar:
    st.markdown("<div class='brand-row'><span class='brand-mark'>S</span> Signal</div>", unsafe_allow_html=True)
    st.markdown("### Text analysis")
    st.caption("An AI writing detector for exploratory review.")
    st.markdown("**TF-IDF + Logistic Regression**")
    st.caption("Trained on HC3 + GPT-wiki-intro")
    if MODEL_PATH.exists():
        st.success("Model is ready")
    else:
        st.error("Model files not found")

    result_file = load_metrics()
    selected = result_file.get("selected_model", "")
    selected_results = result_file.get("results", {}).get(selected, {})
    combined = selected_results.get("combined_test", {})
    heldout = selected_results.get("heldout_hc3_wiki_csai", {})
    if combined:
        st.markdown("---")
        st.markdown("**Combined test results**")
        st.metric("Accuracy", f"{combined.get('accuracy', 0):.1%}")
        st.metric("F1 score", f"{combined.get('f1', 0):.3f}")
        st.caption(f"Current model: {selected.replace('_', ' ')}")
    if heldout:
        st.caption(f"HC3 wiki_csai held-out accuracy: {heldout.get('accuracy', 0):.1%}")


st.markdown(
    """
    <div class="hero">
      <div class="brand-row"><span class="brand-mark">S</span> Writing signals</div>
      <h1>AI or human?<br>Get a second look.</h1>
      <p>Paste a passage for an overall estimate and contextual writing signals. The model offers a clue, not proof of who wrote something.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

main_col, guide_col = st.columns([1.75, 1], gap="large")
with main_col:
    with st.form("analysis_form"):
        st.markdown('<div class="input-title">Analyze a passage</div>', unsafe_allow_html=True)
        st.markdown('<div class="input-hint">Paste text or upload a PDF, image, or text file.</div>', unsafe_allow_html=True)

        upload = st.file_uploader(
            "Upload a document",
            type=["txt", "pdf", "png", "jpg", "jpeg", "bmp", "tif", "tiff", "gif"],
            help="PDFs are extracted with OCR-like text parsing; images are read with OCR.",
        )

        if upload is not None:
            try:
                extracted_text = extract_document_text(upload)
                st.caption(f"{len(extracted_text):,} characters extracted from {upload.name}.")
            except (ValueError, RuntimeError) as exc:
                st.warning(str(exc))
                extracted_text = ""
        else:
            extracted_text = ""

        text = st.text_area(
            "Passage text",
            height=230,
            value=extracted_text,
            placeholder="Paste or type text here...",
            help="Paragraphs separated by a blank line receive individual results.",
        )
        words = len(text.split()) if text.strip() else 0
        left, right = st.columns([1.5, 1])
        with left:
            st.caption(f"{words:,} words  |  {len(text):,} characters")
        with right:
            submitted = st.form_submit_button("Analyze text", use_container_width=True)

with guide_col:
    st.markdown(
        f"""
        <div class="side-card">
          <div class="eyebrow">How to read the result</div>
          <h3>Three confidence zones</h3>
          <p><strong>Likely human</strong><br>At or below {HUMAN_THRESHOLD:.0%} AI likelihood.</p>
          <p><strong>Uncertain</strong><br>Between {HUMAN_THRESHOLD:.0%} and {BOT_THRESHOLD:.0%}. The model sees mixed signals.</p>
          <p><strong>Likely AI</strong><br>At or above {BOT_THRESHOLD:.0%} AI likelihood.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        <div class="side-card">
          <div class="eyebrow">Keep in mind</div>
          <h3>Context matters</h3>
          <p>A low AI score does not prove human authorship. Reworded AI text can evade word-pattern detectors, and formal or non-native writing can be misclassified.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if submitted:
    if not text.strip():
        st.warning("Add some text before starting the analysis.")
    else:
        try:
            with st.spinner("Analyzing your text..."):
                overall = predict(text)
                paragraphs = split_paragraphs(text)
                paragraph_results = [predict(paragraph) for paragraph in paragraphs]
                sentence_results = predict_sentences(text)
            probability = float(overall["ai_probability"])
            label, tone = display_label(str(overall["label"]))
            paragraph_labels = {str(result["label"]) for result in paragraph_results}
            mixed_paragraphs = {"likely_ai", "likely_human"}.issubset(paragraph_labels)
            if mixed_paragraphs:
                label, tone = "Mixed passage", "mixed"
                result_caption = f"Full-passage score across {overall['word_count']:,} words; paragraph estimates differ."
            else:
                result_caption = f"Based on {overall['word_count']:,} words of text"

            st.markdown("## Your result")
            st.markdown(
                f"""
                <div class="result-panel">
                  <div class="result-top">
                    <div><div class="eyebrow">Full-passage estimate</div><div class="result-label">{label}</div><div class="result-caption">{result_caption}</div></div>
                    <span class="score-pill score-{tone}">{probability:.1%} AI likelihood</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(probability)
            metric_cols = st.columns(2)
            metric_cols[0].metric("Words", f"{overall['word_count']:,}")
            metric_cols[1].metric("Passage windows", f"{len(sentence_results):,}")

            if int(overall["word_count"]) < 30:
                st.warning("This is a short sample. Detection is less reliable when there is little text to analyze.")

            st.markdown("### Paragraph-by-paragraph results")
            for index, (paragraph, result) in enumerate(zip(paragraphs, paragraph_results), start=1):
                paragraph_label, _ = display_label(str(result["label"]))
                paragraph_probability = float(result["ai_probability"])
                paragraph_words = int(result["word_count"])
                with st.container(border=True):
                    paragraph_cols = st.columns([1.15, 1, 1.6])
                    paragraph_cols[0].markdown(f"**Paragraph {index}**")
                    paragraph_cols[0].caption(paragraph_label)
                    paragraph_cols[1].metric("AI likelihood", f"{paragraph_probability:.1%}")
                    paragraph_cols[2].caption(f"{paragraph_words:,} words")
                    st.progress(paragraph_probability)
                    if int(result["word_count"]) < 30:
                        st.warning("Short sample; this paragraph estimate is less reliable.")
                    with st.expander(f"Read paragraph {index}"):
                        st.write(paragraph)

            st.markdown("### Passage highlights")
            st.caption("Overlapping windows of at least 40 words where the passage is long enough. These are rough hints and noisier than the overall score.")
            for sentence in sentence_results:
                score = float(sentence["ai_probability"])
                sentence_label, sentence_tone = display_label(str(sentence["label"]))
                safe_sentence = html.escape(str(sentence["sentence"]))
                window = f"Sentences {sentence['start_sentence']}–{sentence['end_sentence']} · {sentence['word_count']} words · rough hint"
                st.markdown(
                    f"<div class='sentence {sentence_tone}'>{safe_sentence}<small>{window} · {sentence_label}  |  {score:.1%} AI likelihood</small></div>",
                    unsafe_allow_html=True,
                )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            st.error(f"The model could not analyze this text: {exc}")

st.markdown(
    "<div class='notice'>Detection is probabilistic and can be wrong, especially for short, edited, or non-native English text. Do not use as sole evidence for academic or disciplinary decisions. A low AI score does not prove human authorship.</div>",
    unsafe_allow_html=True,
)
