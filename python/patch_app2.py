"""
app.py — AI Data Scientist · Visual EDA Dashboard
Upload CSV → Full EDA runs → Visual patterns displayed in tabbed sections
No text summaries — pure data-driven visual insights
"""

import os
import glob
import tempfile
import json

import streamlit as st

st.set_page_config(
    page_title="AI Data Scientist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0a18 0%, #0f0f23 50%, #0d1020 100%);
}

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.025) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}

.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #a78bfa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.25rem;
}
.hero p { color: rgba(255,255,255,0.5); font-size: 0.88rem; margin: 0; }

/* Section headers */
.sec-hdr {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: rgba(255,255,255,0.32);
    margin-bottom: 0.5rem;
}

/* Stat card */
.scard {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 0.9rem 1rem; text-align: center;
}
.scard .v { font-size: 1.6rem; font-weight: 700; display: block; }
.scard .l { font-size: 0.68rem; text-transform: uppercase;
            letter-spacing: 0.07em; color: rgba(255,255,255,0.35); }

/* Insight box */
.insight {
    background: rgba(99,102,241,0.07);
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    color: rgba(255,255,255,0.8);
    font-size: 0.9rem; line-height: 1.7;
    margin-bottom: 0.8rem;
}

/* Outlier table */
.otable { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.otable th {
    background: rgba(99,102,241,0.15);
    color: #818cf8; font-weight: 600;
    padding: 0.5rem 0.8rem; text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.otable td {
    padding: 0.45rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: rgba(255,255,255,0.75);
}
.otable tr:hover td { background: rgba(255,255,255,0.03); }
.badge-red   { color: #f87171; font-weight: 600; }
.badge-green { color: #34d399; font-weight: 600; }
.badge-amber { color: #fbbf24; font-weight: 600; }

/* Stats table */
.stattbl { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.stattbl th {
    background: rgba(52,211,153,0.1); color: #34d399; font-weight: 600;
    padding: 0.5rem 0.8rem; text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.stattbl td {
    padding: 0.45rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: rgba(255,255,255,0.75);
}
.stattbl tr:hover td { background: rgba(52,211,153,0.03); }

/* Dataset Preview Panel */
.dp-stat {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 0.75rem 1rem; text-align: center;
}
.dp-stat .val { font-size: 1.5rem; font-weight: 700; display: block; }
.dp-stat .lbl { font-size: 0.65rem; text-transform: uppercase;
                letter-spacing: 0.08em; color: rgba(255,255,255,0.35); }
.dp-card-hdr {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.75rem 1.1rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    font-size: 0.85rem; font-weight: 600; color: rgba(255,255,255,0.85);
    margin-bottom: 0.6rem;
}
.col-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    border-radius: 6px; padding: 0.22rem 0.6rem;
    font-size: 0.74rem; font-weight: 500; margin: 0.12rem;
}
.col-badge.safe { background: rgba(52,211,153,0.1);  color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.col-badge.warn { background: rgba(251,191,36,0.1);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
.col-badge.drop { background: rgba(248,113,113,0.1); color: #f87171; border: 1px solid rgba(248,113,113,0.2); }
.col-badge .dr  { font-size: 0.68rem; color: rgba(255,255,255,0.3); margin-left: 0.2rem; }

/* Upload area */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02);
    border: 2px dashed rgba(99,102,241,0.35);
    border-radius: 14px; padding: 0.3rem;
}

/* MCQ Clarification Card */
.mcq-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    overflow: hidden;
    margin-top: 1rem;
}
.mcq-card-header {
    display: flex; align-items: center; gap: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding: 1.1rem 1.4rem;
    background: rgba(255,255,255,0.02);
}
.mcq-card-header .mcq-icon {
    width: 34px; height: 34px; background: rgba(255,255,255,0.07);
    border-radius: 8px; display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem;
}
.mcq-card-header h3 { margin: 0; font-size: 0.92rem; font-weight: 600; color: white; }
.mcq-card-header p  { margin: 0; font-size: 0.73rem; color: rgba(255,255,255,0.38); }
.mcq-q-block {
    padding: 1.1rem 1.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.mcq-q-num {
    display: inline-flex; width: 22px; height: 22px;
    background: rgba(255,255,255,0.08); border-radius: 50%;
    align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700; color: rgba(255,255,255,0.6);
    margin-right: 0.55rem; flex-shrink: 0;
}
.mcq-q-text  { font-size: 0.875rem; font-weight: 600; color: white; }
.mcq-q-hint  { font-size: 0.75rem; color: rgba(255,255,255,0.38); margin: 0.3rem 0 0.75rem 1.85rem; line-height: 1.5; }
.mcq-opts    { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-left: 1.85rem; }

/* Pill option buttons */
.pill-btn {
    display: inline-flex; align-items: center;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem; font-weight: 500;
    color: rgba(255,255,255,0.75);
    background: transparent;
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
    user-select: none;
}
.pill-btn:hover {
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.4);
    color: white;
}
.pill-btn.selected {
    background: rgba(99,102,241,0.2);
    border-color: #818cf8;
    color: #c7d2fe;
    font-weight: 600;
}

/* Progress dots */
.mcq-progress { font-size: 0.8rem; color: rgba(255,255,255,0.38); display: flex; align-items: center; gap: 0.45rem; }
.mcq-dot { width: 8px; height: 8px; border-radius: 50%; background: #f59e0b; display: inline-block; flex-shrink:0; }
.mcq-dot.done { background: #34d399; }

/* MCQ pill buttons — override ALL Streamlit button defaults in horizontal blocks */
div.mcq-q-block ~ div[data-testid="stHorizontalBlock"] button,
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    border-radius: 999px !important;
    padding: 0.25rem 1rem !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: rgba(255,255,255,0.72) !important;
    height: 32px !important;
    min-height: 32px !important;
    line-height: 1 !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
    box-shadow: none !important;
    margin: 0 !important;
}
div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.45) !important;
    color: white !important;
}
/* Hide the full-width expander on the column wrapper itself */
div[data-testid="stHorizontalBlock"] [data-testid="column"] {
    padding: 0 4px !important;
}


/* Tabs */
button[data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

hr { border-color: rgba(255,255,255,0.06) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ──────────────────────────────────────────────────────────
for k, v in [
    ("csv_path",        None),
    ("csv_name",        ""),
    ("csv_size_kb",     0),
    ("eda_result",      None),
    ("eda_result_initial", None),
    ("done",            False),
    ("awaiting_user",   False),
    ("phase2_ready",    False),
    ("messages",        []),
    ("preview_done",    False),
    ("dropped_cols",    []),
    # Phase tracking
    ("eda_done",        False),
    ("fe_ready",        False),
    ("fe_done",         False),
    ("fe_result",       None),
    ("fe_report",       ""),
    ("final_ready",     False),
    ("awaiting_fe_mcq", False),
    # Legacy stat keys
    ("stats_json",      None),
    ("outlier_json",    None),
    ("missing_json",    None),
]:
    if k not in st.session_state:
        st.session_state[k] = v


def clear_charts():
    d = os.path.join(os.path.dirname(__file__), "charts")
    if os.path.isdir(d):
        for f in glob.glob(os.path.join(d, "*.png")):
            os.remove(f)


def charts_by_prefix(prefix):
    d = os.path.join(os.path.dirname(__file__), "charts")
    return sorted(glob.glob(os.path.join(d, f"{prefix}*.png")))


def chart_exists(name):
    p = os.path.join(os.path.dirname(__file__), "charts", name)
    return p if os.path.exists(p) else None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(99,102,241,0.25),rgba(139,92,246,0.2));
                border:1px solid rgba(99,102,241,0.2);border-radius:14px;
                padding:1.1rem;text-align:center;margin-bottom:1rem;">
        <div style="font-size:1.7rem;">🧠</div>
        <div style="font-weight:700;font-size:1rem;color:#818cf8;">AI Data Scientist</div>
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.4);margin-top:0.2rem;">
            LangChain · OpenAI · PyTorch
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">📁 Upload Dataset · Max 10 MB</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
        if uploaded.size > 10 * 1024 * 1024:
            st.error("⚠️ File exceeds 10 MB limit.")
            uploaded = None
        elif uploaded.name != st.session_state.csv_name:
            tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False,
                                              dir=os.path.dirname(__file__))
            tmp.write(uploaded.getvalue())
            tmp.close()
            st.session_state.csv_path       = tmp.name
            st.session_state.csv_name       = uploaded.name
            st.session_state.csv_size_kb    = round(uploaded.size / 1024, 1)
            # ── Reset all pipeline phase state completely ──────────────────────
            st.session_state.eda_result         = None
            st.session_state.eda_result_initial  = None
            st.session_state.done           = False
            st.session_state.awaiting_user  = False
            st.session_state.phase2_ready   = False
            st.session_state.preview_done   = False
            st.session_state.dropped_cols   = []
            st.session_state.messages       = []
            st.session_state.eda_done       = False
            st.session_state.fe_ready       = False
            st.session_state.fe_done        = False
            st.session_state.fe_result      = None
            st.session_state.fe_report      = ""
            st.session_state.final_ready    = False
            st.session_state.awaiting_fe_mcq = False
            st.session_state.stats_json     = None
            st.session_state.outlier_json   = None
            st.session_state.missing_json   = None
            # Clear any MCQ question keys from prior run
            for _k in ["eda_mcq_questions", "eda_mcq_questions_answers",
                       "fe_mcq_questions",  "fe_mcq_questions_answers"]:
                st.session_state.pop(_k, None)

    if st.session_state.csv_name:
        st.success(f"✅ **{st.session_state.csv_name}** · {st.session_state.csv_size_kb} KB")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">⚙️ Status</div>', unsafe_allow_html=True)

    if not st.session_state.csv_name:
        st.info("Waiting for CSV upload…")
    elif not st.session_state.done:
        st.warning("📊 Running full EDA…")
    else:
        r = st.session_state.eda_result
        if r and r["success"]:
            n_charts = len(glob.glob(os.path.join(os.path.dirname(__file__), "charts", "*.png")))
            st.success("✅ EDA Phase")
            st.markdown(f"""
            <div style="font-size:0.79rem;color:rgba(255,255,255,0.5);line-height:2;">
            📊 Charts: <b style="color:#34d399;">{n_charts}</b>
            </div>""", unsafe_allow_html=True)
        else:
            st.error("Error during EDA")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""<div class="sec-hdr">🔧 EDA Pipeline</div>
    <div style="font-size:0.76rem;color:rgba(255,255,255,0.4);line-height:1.95;">
    📐 Dataset info<br>📊 Descriptive stats<br>❓ Missing values<br>
    📈 Distributions (histograms)<br>📦 Box plots & outliers<br>
    🌡️ Correlation heatmap<br>🔵 Scatter relationships<br>📉 Column means
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        clear_charts()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main — Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 AI Data Scientist</h1>
    <p>Upload any CSV → Agent runs full EDA automatically → All visual patterns displayed below</p>
</div>""", unsafe_allow_html=True)

# ─── Idle ────────────────────────────────────────────────────────────────────
if not st.session_state.csv_name:
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, lbl, desc in [
        (c1, "📁", "Upload CSV",       "Max 10 MB · Sidebar"),
        (c2, "📈", "Distributions",    "Per-column histograms + skew"),
        (c3, "📦", "Outlier Detection","Box plots + IQR analysis"),
        (c4, "🌡️", "Correlations",     "Heatmap + scatter plots"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        border-radius:12px;padding:1.2rem;text-align:center;">
                <div style="font-size:1.9rem;">{icon}</div>
                <div style="font-weight:600;color:white;margin:0.4rem 0 0.25rem;font-size:0.9rem;">{lbl}</div>
                <div style="color:rgba(255,255,255,0.38);font-size:0.78rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()

# ─── Dataset Preview & Column Cleanup ────────────────────────────────────────
if not st.session_state.preview_done and st.session_state.csv_path:
    import pandas as pd
    try:
        _pf = pd.read_csv(st.session_state.csv_path)
    except Exception as _e:
        st.error(f"Could not read CSV: {_e}")
        st.stop()

    _nr, _nc = _pf.shape
    _nnum = int(_pf.select_dtypes(include='number').shape[1])
    _ncat = int(_pf.select_dtypes(exclude='number').shape[1])

    # Stat cards
    st.markdown("#### 📊 Dataset Overview")
    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
    for _cw, _v, _l, _cl in [
        (_sc1, f"{_nr:,}",  "Total Rows",       "#818cf8"),
        (_sc2, f"{_nc}",    "Total Columns",    "#34d399"),
        (_sc3, f"{_nnum}",  "Numeric Cols",     "#60a5fa"),
        (_sc4, f"{_ncat}",  "Categorical Cols", "#f472b6"),
    ]:
        _cw.markdown(
            '<div class="dp-stat">'
            f'<span class="val" style="color:{_cl};">{_v}</span>'
            f'<span class="lbl">{_l}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Column intelligence
    st.markdown(
        '<div class="dp-card-hdr">🔬 Column Intelligence — '
        'Red = recommended to drop, Yellow = high missing, Green = healthy</div>',
        unsafe_allow_html=True
    )

    _suggested = []
    _badge_html = ""
    for _col in _pf.columns:
        _miss  = _pf[_col].isnull().mean()
        _nuniq = _pf[_col].nunique()
        if _nuniq == _nr:
            _b, _r = "drop", "ID/index"
            _suggested.append(_col)
        elif _nuniq <= 1:
            _b, _r = "drop", "zero variance"
            _suggested.append(_col)
        elif _miss >= 0.5:
            _b, _r = "warn", f"{_miss*100:.0f}% missing"
        else:
            _b, _r = "safe", ""
        _note = f'<span class="dr">({_r})</span>' if _r else ""
        _badge_html += f'<span class="col-badge {_b}">{_col}{_note}</span>'

    st.markdown(f'<div style="padding:0.4rem 0 0.8rem;">{_badge_html}</div>', unsafe_allow_html=True)

    # Multiselect to choose which columns to drop
    st.markdown("**Select columns to remove before analysis:**")
    cols_to_drop = st.multiselect(
        label="", options=list(_pf.columns), default=_suggested,
        placeholder="Choose columns to drop...", label_visibility="collapsed",
    )

    # Dataset preview
    with st.expander("📋 Preview Dataset (first 50 rows)", expanded=False):
        st.dataframe(_pf.head(50), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Apply & Start Analysis", type="primary", use_container_width=True):
        if cols_to_drop:
            _pf = _pf.drop(columns=cols_to_drop, errors="ignore")
            _pf.to_csv(st.session_state.csv_path, index=False)
        st.session_state.dropped_cols = cols_to_drop
        st.session_state.preview_done = True
        st.session_state.messages     = []
        clear_charts()
        st.rerun()
    st.stop()

import json, re
from langchain_core.messages import HumanMessage

def extract_mcq_json(text):
    """Robustly extract MCQ JSON from agent message that may have surrounding prose."""
    brace_match = re.search(r'(\{[\s\S]+\})', text, re.DOTALL)
    candidates = [brace_match.group(1)] if brace_match else []
    candidates += re.findall(r'\{[\s\S]*?\}', text, re.DOTALL)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if "clarification_needed" in obj or "questions" in obj:
                return obj
        except Exception:
            continue
    return None

def render_mcq_card(title: str, subtitle: str, mcq_key: str, submit_label: str):
    """
    Renders a full MCQ card for a given set of questions stored in session_state[mcq_key].
    Returns (answered, total, submitted_summary or None).
    """
    questions = st.session_state.get(mcq_key)
    if not questions:
        return 0, 0, None

    ans_key   = f"{mcq_key}_answers"
    if ans_key not in st.session_state or len(st.session_state[ans_key]) != len(questions):
        st.session_state[ans_key] = {q["id"]: None for q in questions}

    answered = sum(1 for v in st.session_state[ans_key].values() if v is not None)
    total    = len(questions)

    st.markdown(f"""
    <div class="mcq-card">
      <div class="mcq-card-header">
        <div class="mcq-icon">🤖</div>
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
      </div>
    """, unsafe_allow_html=True)

    for idx, q in enumerate(questions):
        qid  = q["id"]
        cur  = st.session_state[ans_key].get(qid)
        hint_html = f'<div class="mcq-q-hint">{q.get("hint","")}</div>' if q.get("hint") else ""
        st.markdown(f"""
        <div class="mcq-q-block">
          <div style="display:flex;align-items:flex-start;">
            <span class="mcq-q-num">{idx+1}</span>
            <div>
              <div class="mcq-q-text">{q["question"]}</div>
              {hint_html}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        btn_cols = st.columns(len(q["options"]))
        for bcol, opt in zip(btn_cols, q["options"]):
            label = ("✅ " if cur == opt else "") + str(opt)
            if bcol.button(label, key=f"{mcq_key}_{qid}_{opt}"):
                st.session_state[ans_key][qid] = opt
                st.rerun()

    answered = sum(1 for v in st.session_state[ans_key].values() if v is not None)
    dot_html = " ".join(
        f'<span class="mcq-dot{" done" if st.session_state[ans_key].get(q["id"]) else ""}"></span>'
        for q in questions
    )
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                margin-top:0.5rem;padding:0.4rem 0.2rem;">
      <div class="mcq-progress">{dot_html} <span>{answered} / {total} answered</span></div>
    </div>""", unsafe_allow_html=True)

    summary = None
    if answered == total:
        if st.button(submit_label, use_container_width=True, type="primary"):
            summary = "; ".join(
                f'Q{q["id"]}: {st.session_state[ans_key][q["id"]]}' for q in questions
            )
    else:
        st.button(submit_label, use_container_width=True, disabled=True,
                  help=f"Answer all {total} questions to proceed.")
    return answered, total, summary



# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION & TAB DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("preview_done"):
    _tab_labels = ["📊 Initial EDA", "🔧 Feature Engineering", "📈 Final Analysis"]
    _tabs = st.tabs(_tab_labels)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — Initial EDA & Clarification
    # ─────────────────────────────────────────────────────────────────────────
    with _tabs[0]:
        _phase_idle = (
            not st.session_state.done
            and not st.session_state.awaiting_user
            and not st.session_state.phase2_ready
            and not st.session_state.get("eda_done")
            and not st.session_state.get("fe_ready")
            and not st.session_state.get("fe_done")
            and not st.session_state.get("final_ready")
        )

        if _phase_idle:
            with st.spinner("🤖 Reading dataset and checking for clarification needs…"):
                try:
                    from agent import run_analysis_graph
                    if not st.session_state.messages:
                        st.session_state.messages = [HumanMessage(
                            content=f"Dataset '{st.session_state.csv_name}' has been uploaded. "
                                    "Read the data info and decide if you have any clarification questions before generating charts."
                        )]
                        clear_charts()
                    res = run_analysis_graph(st.session_state.csv_path, st.session_state.messages, phase="read")
                    st.session_state.messages   = res["messages"]
                    st.session_state.eda_result = res
                    last = res["messages"][-1].content.strip()
                    if "READY_TO_ANALYZE" in last:
                        st.session_state.phase2_ready = True
                    else:
                        st.session_state.awaiting_user = True
                except Exception as e:
                    st.session_state.eda_result = {"success": False, "answer": str(e), "error": str(e)}
                    st.session_state.done = True
            st.rerun()

        elif st.session_state.awaiting_user and not st.session_state.get("eda_done"):
            parsed = extract_mcq_json(st.session_state.messages[-1].content)
            if parsed and parsed.get("clarification_needed"):
                st.session_state["eda_mcq_questions"] = parsed.get("questions", [])

            if st.session_state.get("eda_mcq_questions"):
                _, _, summary = render_mcq_card(
                    title="AI Data Scientist — Clarification",
                    subtitle="Help the agent understand your dataset better to produce the best EDA.",
                    mcq_key="eda_mcq_questions",
                    submit_label="🔍 Start Analysis",
                )
                if summary:
                    st.session_state.messages.append(HumanMessage(
                        content=f"User analysis preferences: {summary}"
                    ))
                    st.session_state.awaiting_user = False
                    st.session_state.phase2_ready  = True
                    st.session_state.pop("eda_mcq_questions", None)
                    st.session_state.pop("eda_mcq_questions_answers", None)
                    st.rerun()
            else:
                st.info(f"🤖 **Agent:** {st.session_state.messages[-1].content}")
                reply = st.chat_input("Provide clarification…")
                if reply:
                    st.session_state.messages.append(HumanMessage(content=reply))
                    st.session_state.awaiting_user = False
                    st.session_state.phase2_ready  = True
                    st.rerun()

        elif st.session_state.phase2_ready and not st.session_state.get("eda_done"):
            with st.spinner("📊 Generating initial EDA charts based on dataset patterns…"):
                try:
                    from agent import run_analysis_graph
                    res = run_analysis_graph(st.session_state.csv_path, st.session_state.messages, phase="eda")
                    st.session_state.messages              = res["messages"]
                    st.session_state.eda_result            = res
                    st.session_state.eda_result_initial    = res   # preserved for Tab 1 display
                    st.session_state.eda_done              = True
                    st.session_state.phase2_ready          = False
                    st.session_state.fe_ready              = True   # auto-trigger FE next
                except Exception as e:
                    err = {"success": False, "answer": str(e), "error": str(e)}
                    st.session_state.eda_result         = err
                    st.session_state.eda_result_initial = err
                    st.session_state.eda_done           = True
                    st.session_state.done               = True
                    st.session_state.phase2_ready       = False
            st.rerun()

        elif st.session_state.get("eda_done"):
            import pandas as pd
            eda_res = st.session_state.get("eda_result_initial") or st.session_state.eda_result
            if not eda_res or not eda_res.get("success"):
                st.error(f"EDA Error: {eda_res.get('error','') if eda_res else 'No result'}")
            else:
                with st.expander("📂 Original Dataset", expanded=False):
                    try:
                        st.dataframe(pd.read_csv(st.session_state.csv_path), use_container_width=True)
                    except Exception:
                        pass
                import glob, os
                eda_charts = sorted([
                    p for p in glob.glob(os.path.join(os.path.dirname(__file__), "charts", "*.png"))
                    if "cleaned" not in os.path.basename(p)
                ])
                if eda_charts:
                    for i in range(0, len(eda_charts), 2):
                        row = eda_charts[i:i+2]
                        cols = st.columns(len(row))
                        for col, path in zip(cols, row):
                            with col:
                                st.image(path, use_container_width=True)
                                st.markdown(
                                    f'<div style="text-align:center;font-size:0.75rem;color:rgba(255,255,255,0.38);">'
                                    f'{os.path.splitext(os.path.basename(path))[0]}</div>',
                                    unsafe_allow_html=True
                                )
                else:
                    st.info("No EDA charts generated yet.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — Feature Engineering & Extraction
    # ─────────────────────────────────────────────────────────────────────────
    with _tabs[1]:
        if not st.session_state.get("eda_done"):
            st.info("Awaiting completion of Initial EDA phase...")

        elif st.session_state.get("fe_ready") and not st.session_state.get("fe_done") and not st.session_state.get("awaiting_fe_mcq"):
            with st.spinner("🔧 AI is reviewing EDA to plan feature engineering (column drops, encoding, binarization)…"):
                try:
                    from agent import run_analysis_graph
                    res = run_analysis_graph(st.session_state.csv_path, st.session_state.messages, phase="fe")
                    st.session_state.messages  = res["messages"]
                    st.session_state.fe_result = res
                    last_fe = res["messages"][-1].content.strip()

                    try:
                        parsed_fe = extract_mcq_json(last_fe)
                    except Exception:
                        parsed_fe = None

                    if parsed_fe and parsed_fe.get("clarification_needed"):
                        st.session_state["fe_mcq_questions"] = parsed_fe.get("questions", [])
                        st.session_state.awaiting_fe_mcq     = True
                    else:
                        st.session_state.fe_done     = True
                        st.session_state.fe_ready    = False
                        st.session_state.final_ready = True
                        st.session_state.fe_report   = last_fe
                except Exception as _fe_err:
                    st.session_state.fe_result = {"success": False, "answer": str(_fe_err), "error": str(_fe_err)}
                    st.session_state.fe_done   = True
                    st.session_state.fe_ready  = False
                    st.session_state.fe_report = f"Error: {_fe_err}"
                    st.session_state.final_ready = True
            st.rerun()

        elif st.session_state.get("awaiting_fe_mcq") and not st.session_state.get("fe_done"):
            _, _, fe_summary = render_mcq_card(
                title="Feature Engineering — Clarification",
                subtitle="The AI needs your input before cleaning the data",
                mcq_key="fe_mcq_questions",
                submit_label="🔧 Apply Feature Engineering",
            )
            if fe_summary:
                st.session_state.messages.append(HumanMessage(
                    content=f"Feature engineering preferences: {fe_summary}"
                ))
                st.session_state.awaiting_fe_mcq = False
                st.session_state.fe_ready        = True
                st.session_state.pop("fe_mcq_questions", None)
                st.session_state.pop("fe_mcq_questions_answers", None)
                st.rerun()

        elif st.session_state.get("fe_done"):
            import pandas as pd
            fe_report = st.session_state.get("fe_report", "")
            fe_ops_lines = []
            if fe_report:
                in_table = False
                for raw in fe_report.split("\n"):
                    line = raw.strip()
                    if not line or line.startswith("FE_COMPLETE:") or line.startswith("{"):
                        continue
                    if "Feature Extraction (Binarization) Complete" in line:
                        in_table = True
                        continue
                    if in_table:
                        if line.startswith("-") or line.startswith("Original") or line.startswith("New shape"):
                            continue
                    else:
                        if not line.startswith("-"):
                            fe_ops_lines.append(line)

            st.markdown('<div class="dp-card-hdr">⚙️ Feature Engineering Operations</div>', unsafe_allow_html=True)
            if fe_ops_lines:
                def _badge(text):
                    text_l = text.lower()
                    if "dropped" in text_l:
                        color, bg, icon = "#f87171", "rgba(248,113,113,0.1)", "🗑️"
                    elif "encoded" in text_l or "binary" in text_l:
                        color, bg, icon = "#60a5fa", "rgba(96,165,250,0.1)", "🔄"
                    elif "missing" in text_l or "filled" in text_l or "handled" in text_l:
                        color, bg, icon = "#fbbf24", "rgba(251,191,36,0.1)", "🩹"
                    elif "binarized" in text_l:
                        color, bg, icon = "#34d399", "rgba(52,211,153,0.1)", "⚡"
                    else:
                        color, bg, icon = "rgba(255,255,255,0.6)", "rgba(255,255,255,0.04)", "•"
                    return (
                        f'<div style="display:flex;align-items:flex-start;gap:0.5rem;' +
                        f'padding:0.5rem 0.8rem;margin:0.25rem 0;' +
                        f'background:{bg};border-radius:8px;border-left:3px solid {color};">' +
                        f'<span style="font-size:0.9rem;">{icon}</span>' +
                        f'<span style="font-size:0.82rem;color:rgba(255,255,255,0.8);">{text}</span></div>'
                    )
                ops_html = "".join(_badge(l) for l in fe_ops_lines if l)
                st.markdown(f'<div style="margin-bottom:1rem;">{ops_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight">No additional cleaning operations were needed.</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="dp-card-hdr">⚡ Feature Extraction — Numeric → Binary Indicators</div>', unsafe_allow_html=True)
            try:
                cleaned_df = pd.read_csv(st.session_state.csv_path)
                bin_cols = [(c.replace("_bin", ""), c) for c in cleaned_df.columns if c.endswith("_bin")]
            except Exception:
                cleaned_df = None
                bin_cols = []

            if bin_cols and cleaned_df is not None:
                rows_html = ""
                for orig, binc in bin_cols:
                    if orig in cleaned_df.columns:
                        ones  = int(cleaned_df[binc].sum())
                        zeros = int(len(cleaned_df) - ones)
                        pct   = round(ones / len(cleaned_df) * 100, 1) if len(cleaned_df) > 0 else 0
                        o_min  = round(cleaned_df[orig].min(), 2)
                        o_max  = round(cleaned_df[orig].max(), 2)
                        o_mean = round(cleaned_df[orig].mean(), 2)
                    else:
                        zeros = ones = pct = o_min = o_max = o_mean = "—"
                    pct_bar = ""
                    if isinstance(pct, (int, float)):
                        pct_bar = (
                            f'<div style="display:flex;align-items:center;gap:0.4rem;">' +
                            f'<div style="flex:1;height:6px;background:rgba(255,255,255,0.08);border-radius:3px;">' +
                            f'<div style="width:{pct}%;height:6px;background:#34d399;border-radius:3px;"></div></div>' +
                            f'<span style="font-size:0.75rem;color:#34d399;">{pct}%</span></div>'
                        )
                    rows_html += (
                        f"<tr>"
                        f'<td><code style="color:#818cf8;">{orig}</code></td>' +
                        f'<td><code style="color:#34d399;">{binc}</code></td>' +
                        f'<td style="color:rgba(255,255,255,0.5);">{o_min}</td>' +
                        f'<td style="color:rgba(255,255,255,0.5);">{o_max}</td>' +
                        f'<td style="color:rgba(255,255,255,0.5);">{o_mean}</td>' +
                        f'<td style="color:#f87171;">{zeros}</td>' +
                        f'<td style="color:#34d399;">{ones}</td>' +
                        f"<td>{pct_bar}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f'''<table class="otable" style="width:100%;">
                      <thead><tr>
                        <th>Original Column</th><th>Binary Column</th>
                        <th>Min</th><th>Max</th><th>Mean</th>
                        <th style="color:#f87171;">Zeros (0)</th>
                        <th style="color:#34d399;">Ones (1)</th>
                        <th>% Positive</th>
                      </tr></thead>
                      <tbody>{rows_html}</tbody>
                    </table>''',
                    unsafe_allow_html=True
                )
            else:
                st.info("No binary (*_bin) columns found. The agent may not have executed binarization.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="dp-card-hdr">📂 Cleaned & Extracted Dataset</div>', unsafe_allow_html=True)
            if cleaned_df is not None:
                _nr2, _nc2 = cleaned_df.shape
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows",            f"{_nr2:,}")
                m2.metric("Total Columns",   f"{_nc2}")
                m3.metric("Binary Features", f"{len(bin_cols)}")
                with st.expander("View full cleaned dataset", expanded=False):
                    st.dataframe(cleaned_df, use_container_width=True)

            import glob, os
            fe_heatmap = os.path.join(os.path.dirname(__file__), "charts", "heatmap_correlation_cleaned.png")
            if os.path.exists(fe_heatmap):
                st.markdown("#### 🌡️ Post-Feature Engineering Correlation Heatmap")
                st.image(fe_heatmap, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — Final Analysis
    # ─────────────────────────────────────────────────────────────────────────
    with _tabs[2]:
        if not st.session_state.get("fe_done"):
            st.info("Awaiting completion of Feature Engineering phase...")

        elif st.session_state.get("final_ready") and not st.session_state.get("done"):
            with st.spinner("📈 Generating final optimized charts on cleaned data…"):
                try:
                    from agent import run_analysis_graph
                    res = run_analysis_graph(st.session_state.csv_path, st.session_state.messages, phase="final")
                    st.session_state.messages    = res["messages"]
                    st.session_state.eda_result  = res    # Optional, mostly sets the state
                    st.session_state.done        = True
                    st.session_state.final_ready = False
                except Exception as e:
                    st.session_state.eda_result = {"success": False, "answer": str(e), "error": str(e)}
                    st.session_state.done       = True
                    st.session_state.final_ready = False
            st.rerun()

        elif st.session_state.get("done"):
            final_res = st.session_state.eda_result
            if not final_res or not final_res.get("success"):
                st.error(f"Error: {final_res.get('error', 'Unknown') if final_res else 'No result'}")
            else:
                import glob, os
                all_charts = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "charts", "*.png")))
                eda_chart_names = {
                    os.path.basename(p) for p in all_charts
                    if (os.path.basename(p).startswith("heatmap_") or
                        os.path.basename(p).startswith("hist_") or
                        os.path.basename(p).startswith("scatter_") or
                        os.path.basename(p).startswith("box_") or
                        os.path.basename(p).startswith("count_"))
                    and "cleaned" not in os.path.basename(p)
                }
                final_charts = [p for p in all_charts
                                 if os.path.basename(p) not in eda_chart_names
                                 and "cleaned" not in os.path.basename(p)]
                if final_charts:
                    for i in range(0, len(final_charts), 2):
                        row = final_charts[i:i+2]
                        cols = st.columns(len(row))
                        for col, path in zip(cols, row):
                            with col:
                                st.image(path, use_container_width=True)
                                st.markdown(
                                    f'<div style="text-align:center;font-size:0.75rem;color:rgba(255,255,255,0.38);">'
                                    f'{os.path.splitext(os.path.basename(path))[0]}</div>',
                                    unsafe_allow_html=True
                                )
                else:
                    st.info("No new Final charts were generated. Check the previous tabs.")

                last_content = st.session_state.messages[-1].content if st.session_state.messages else ""
                if last_content and "FINAL_COMPLETE:" in last_content:
                    summary_text = last_content.split("FINAL_COMPLETE:")[-1].strip()
                    lines_s = [l.strip("•- ").strip() for l in summary_text.split("\n") if l.strip()]
                    bullets = "".join(f"<li>{l}</li>" for l in lines_s[:5] if l)
                    st.markdown(
                        f'<div class="insight"><strong>🤖 Key Insights:</strong>' +
                        f'<ul style="margin:0.4rem 0 0 1.2rem;">{bullets}</ul></div>',
                        unsafe_allow_html=True
                    )
