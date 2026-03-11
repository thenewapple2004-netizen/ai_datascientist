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
for k, v in [("csv_path", None), ("csv_name", ""), ("csv_size_kb", 0),
             ("eda_result", None), ("done", False), ("stats_json", None),
             ("outlier_json", None), ("missing_json", None)]:
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
            st.session_state.csv_path    = tmp.name
            st.session_state.csv_name    = uploaded.name
            st.session_state.csv_size_kb = round(uploaded.size / 1024, 1)
            st.session_state.eda_result  = None
            st.session_state.done        = False
            st.session_state.stats_json  = None
            st.session_state.outlier_json = None
            st.session_state.missing_json = None
            st.session_state.messages = []
            st.session_state.awaiting_user = False

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

# ─── Phase 1: Read data + ask clarification MCQs ──────────────────────────────
if not st.session_state.done and not st.session_state.awaiting_user and not st.session_state.get("phase2_ready"):
    with st.spinner("🤖 Agent is reading the dataset and preparing questions..."):
        try:
            from agent import run_analysis_graph
            from langchain_core.messages import HumanMessage

            if not st.session_state.messages:
                st.session_state.messages = [HumanMessage(
                    content=f"Dataset '{st.session_state.csv_name}' has been uploaded. Read the data info and decide if you have any clarification questions before generating charts."
                )]
                clear_charts()

            res = run_analysis_graph(st.session_state.csv_path, st.session_state.messages, phase=1)
            st.session_state.messages = res["messages"]
            st.session_state.eda_result = res

            last = res["messages"][-1].content.strip()
            if "READY_TO_ANALYZE" in last:
                # No questions — go straight to charting
                st.session_state.phase2_ready = True
            else:
                # Agent has questions — show MCQ panel
                st.session_state.awaiting_user = True
        except Exception as e:
            import traceback
            st.session_state.eda_result = {"success": False, "answer": str(e), "error": str(e)}
            st.session_state.done = True
    st.rerun()

# ─── Phase 2: Generate charts after user answered ──────────────────────────────
if st.session_state.get("phase2_ready") and not st.session_state.done:
    with st.spinner("📊 Generating best visual patterns based on your preferences..."):
        try:
            from agent import run_analysis_graph

            res = run_analysis_graph(st.session_state.csv_path, st.session_state.messages, phase=2)
            st.session_state.messages = res["messages"]
            st.session_state.eda_result = res
            st.session_state.done = True
            st.session_state.phase2_ready = False
        except Exception as e:
            import traceback
            st.session_state.eda_result = {"success": False, "answer": str(e), "error": str(e)}
            st.session_state.done = True
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# REPORT: Simplified Dashboard — only show after analysis is done
# ─────────────────────────────────────────────────────────────────────────────

# If agent is waiting for MCQ answers, skip the dashboard and go straight to MCQ UI
if st.session_state.awaiting_user:
    result = st.session_state.eda_result
    pass  # fall through to HITL section below
elif not st.session_state.eda_result:
    st.stop()
else:
    result = st.session_state.eda_result
    if not result["success"]:
        st.error(f"**EDA Error:** {result.get('error', result['answer'])}")
        st.stop()

if not st.session_state.awaiting_user:
    # ── Top metric bar ───────────────────────────────────────────────────────────
    n_charts = len(glob.glob(os.path.join(os.path.dirname(__file__), "charts", "*.png")))

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="scard">
            <span class="v" style="color:#818cf8;">{st.session_state.csv_name}</span>
            <span class="l">Dataset</span>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="scard">
            <span class="v" style="color:#34d399;">{n_charts}</span>
            <span class="l">Charts Generated</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ── Loaded Dataset + Charts (only when done and not awaiting MCQ)
if not st.session_state.awaiting_user:
    with st.expander("📂 View Loaded Dataset (Raw Data)", expanded=False):
        import pandas as pd
        try:
            temp_df = pd.read_csv(st.session_state.csv_path)
            st.dataframe(temp_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load dataframe preview: {e}")

    if st.session_state.done:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### 📊 Visual Analysis")
        chart_paths = glob.glob(os.path.join(os.path.dirname(__file__), "charts", "*.png"))
        if chart_paths:
            for i in range(0, len(chart_paths), 2):
                row = chart_paths[i:i+2]
                cols = st.columns(len(row))
                for col, path in zip(cols, row):
                    col_name = os.path.splitext(os.path.basename(path))[0]
                    with col:
                        st.image(path, use_container_width=True)
                        st.markdown(f'<div style="text-align:center;font-size:0.77rem;color:rgba(255,255,255,0.4);">{col_name}</div>',
                                    unsafe_allow_html=True)
        else:
            st.info("No charts were generated by the agent yet.")

# ── Hitl / Clarification UI ─────────────────────────────────
if st.session_state.awaiting_user:

    msg_content = st.session_state.messages[-1].content
    parsed_mcq = None
    import json, re

    # Robustly extract JSON block from agent message (handles text before/after JSON)
    def extract_json(text):
        # Try finding a JSON block between { } that contains clarification_needed
        matches = re.findall(r'\{[\s\S]*?\}', text, re.DOTALL)
        # Try multi-level: find the outermost { ... } block
        brace_match = re.search(r'(\{[\s\S]+\})', text, re.DOTALL)
        candidates = [brace_match.group(1)] if brace_match else []
        candidates += matches
        for candidate in candidates:
            try:
                obj = json.loads(candidate)
                if "clarification_needed" in obj or "question" in obj:
                    return obj
            except:
                continue
        return None

    parsed_mcq = extract_json(msg_content)

    questions = None
    if parsed_mcq and parsed_mcq.get("clarification_needed") and isinstance(parsed_mcq.get("questions"), list):
        questions = parsed_mcq["questions"]
    elif parsed_mcq and "question" in parsed_mcq and "options" in parsed_mcq:
        # backward compat single-question format
        questions = [{"id": 1, "question": parsed_mcq["question"], "hint": "", "options": parsed_mcq["options"]}]

    if questions:
        # Track answers per question in session
        if "mcq_answers" not in st.session_state or len(st.session_state.mcq_answers) != len(questions):
            st.session_state.mcq_answers = {q["id"]: None for q in questions}

        answered = sum(1 for v in st.session_state.mcq_answers.values() if v is not None)
        total    = len(questions)

        # ── MCQ card header
        st.markdown("""
        <div class="mcq-card">
          <div class="mcq-card-header">
            <div class="mcq-icon">🤖</div>
            <div>
              <h3>AI Data Scientist — Clarification</h3>
              <p>Refine the analysis by answering these questions</p>
            </div>
          </div>
        """, unsafe_allow_html=True)

        # ── Render each question
        for idx, q in enumerate(questions):
            qid   = q["id"]
            qtext = q["question"]
            hint  = q.get("hint", "")
            opts  = q["options"]
            cur   = st.session_state.mcq_answers.get(qid)

            # Question header (number + text + hint)
            st.markdown(f"""
            <div class="mcq-q-block">
              <div style="display:flex;align-items:flex-start;">
                <span class="mcq-q-num">{idx+1}</span>
                <div>
                  <div class="mcq-q-text">{qtext}</div>
                  <div class="mcq-q-hint">{hint}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ONE row of compact pill buttons — no use_container_width so they stay small
            btn_cols = st.columns(len(opts))
            for bcol, opt in zip(btn_cols, opts):
                label = ("✅ " if cur == opt else "") + str(opt)
                if bcol.button(label, key=f"mcq_{qid}_{opt}"):
                    st.session_state.mcq_answers[qid] = opt
                    st.rerun()


        # ── Footer: progress + submit
        answered = sum(1 for v in st.session_state.mcq_answers.values() if v is not None)
        dot_html = " ".join(
            f'<span class="mcq-dot{" done" if st.session_state.mcq_answers.get(q["id"]) else ""}"></span>'
            for q in questions
        )
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    margin-top:0.5rem;padding:0.4rem 0.2rem;">
          <div class="mcq-progress">{dot_html} <span>{answered} / {total} answered</span></div>
        </div>
        """, unsafe_allow_html=True)

        if answered == total:
            if st.button("🔍 Start Analysis", use_container_width=True, type="primary"):
                summary = "User clarification answers: "
                summary += "; ".join(f'Q{q["id"]}: {st.session_state.mcq_answers[q["id"]]}' for q in questions)
                from langchain_core.messages import HumanMessage
                st.session_state.messages.append(HumanMessage(content=summary))
                st.session_state.awaiting_user = False
                st.session_state.phase2_ready = True
                del st.session_state.mcq_answers
                st.rerun()
        else:
            st.button("🔍 Start Analysis", use_container_width=True, disabled=True,
                      help=f"Answer all {total} questions to proceed.")
    else:
        # Fallback: plain text clarification
        st.info(f"🤖 **Agent:** {msg_content}")
        reply = st.chat_input("Provide clarification to the agent...")
        if reply:
            from langchain_core.messages import HumanMessage
            st.session_state.messages.append(HumanMessage(content=reply))
            st.session_state.awaiting_user = False
            st.rerun()

elif st.session_state.done:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.success("🤖 **Agent Summary:**")
    st.markdown(st.session_state.messages[-1].content)
