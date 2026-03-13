"""
app.py — AI Data Scientist  ·  Main Dashboard
Upload CSV → 4-phase AI pipeline → EDA + Feature Engineering + Final Analysis
Run: streamlit run python/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import json
import tempfile

import pandas as pd
import streamlit as st
import base64
from langchain_core.messages import HumanMessage

from backend.agent import run_analysis_graph
from frontend.styles import inject_styles
from frontend.components import (
    extract_mcq_json, render_mcq_card,
    stat_cards, section_banner, card_header,
    generate_pdf_report, calc_cost, format_cost,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Scientist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

# ─── Constants ────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR  = os.path.join(_HERE, "charts")


def clear_charts():
    if os.path.isdir(CHARTS_DIR):
        for f in glob.glob(os.path.join(CHARTS_DIR, "*.png")):
            os.remove(f)

def _get_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─── Session State ────────────────────────────────────────────────────────────
_DEFAULTS = {
    "csv_path":         None,
    "csv_name":         "",
    "csv_size_kb":      0,
    "messages":         [],
    "original_df":      None,
    "cleaned_df":       None,
    "token_usage":      [],
    # phase flags
    "preview_done":     False,
    "awaiting_user":    False,
    "phase2_ready":     False,
    "eda_done":         False,
    "eda_result":       None,
    "eda_result_initial": None,
    "fe_ready":         False,
    "fe_done":          False,
    "fe_result":        None,
    "fe_report":        "",
    "awaiting_fe_mcq":  False,
    "final_ready":      False,
    "done":             False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _reset_pipeline():
    """Reset all pipeline state for a new upload."""
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    for _k in ["eda_mcq_questions", "eda_mcq_questions_answers",
               "fe_mcq_questions",  "fe_mcq_questions_answers"]:
        st.session_state.pop(_k, None)


def _accumulate_tokens(res: dict):
    """Append phase token usage to session cumulative list."""
    usage = res.get("token_usage")
    if usage:
        st.session_state.token_usage.append(usage)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(99,102,241,0.1),rgba(139,92,246,0.05));
                border:1px solid #334155;border-radius:16px;
                padding:1.4rem;text-align:center;margin-bottom:1.2rem;">
      <div style="font-size:2.2rem;margin-bottom:0.5rem;">🧠</div>
      <div style="font-weight:800;font-size:1.1rem;background:linear-gradient(135deg,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">AI Data Scientist</div>
      <div style="font-size:0.75rem;color:#64748b;margin-top:0.3rem;font-weight:500;">
        LangGraph · GPT-4o-mini
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">📁 Upload Dataset · Max 10 MB</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
        if uploaded.size > 10 * 1024 * 1024:
            st.error("⚠️ File exceeds 10 MB limit.")
            uploaded = None
        elif uploaded.name != st.session_state.csv_name:
            tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False, dir=_HERE)
            tmp.write(uploaded.getvalue())
            tmp.close()
            _reset_pipeline()
            st.session_state.csv_path    = tmp.name
            st.session_state.csv_name    = uploaded.name
            st.session_state.csv_size_kb = round(uploaded.size / 1024, 1)

    if st.session_state.csv_name:
        st.success(f"✅ **{st.session_state.csv_name}** · {st.session_state.csv_size_kb} KB")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">⚙️ Pipeline Status</div>', unsafe_allow_html=True)

    if not st.session_state.csv_name:
        st.info("Waiting for CSV upload…")
    else:
        phases = [
            ("📖 Read",               st.session_state.preview_done),
            ("📊 Initial EDA",         st.session_state.eda_done),
            ("🔧 Feature Engineering", st.session_state.fe_done),
            ("📈 Final Analysis",      st.session_state.done),
        ]
        for label, done in phases:
            icon  = "✅" if done else ("🔄" if not st.session_state.done else "⏳")
            color = "#4ade80" if done else "#64748b"
            st.markdown(
                f'<div style="font-size:0.85rem;font-weight:700;color:{color};padding:0.2rem 0;">{icon} {label}</div>',
                unsafe_allow_html=True,
            )

        # Token summary in sidebar
        token_usage = st.session_state.get("token_usage", [])
        if token_usage:
            total_in  = sum(u.get("input_tokens",  0) for u in token_usage)
            total_out = sum(u.get("output_tokens", 0) for u in token_usage)
            cost      = calc_cost(total_in, total_out)
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="sec-hdr">💰 API Usage</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:0.85rem;color:#94a3b8;line-height:2.2;font-weight:600;">'
                f'📥 Input Tokens: <b style="color:#ffffff;">{total_in:,}</b><br>'
                f'📤 Output Tokens: <b style="color:#ffffff;">{total_out:,}</b><br>'
                f'💵 Total Cost: <b style="color:#f472b6;">{format_cost(cost)}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🔄 Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        clear_charts()
        st.rerun()


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 AI Data Scientist</h1>
  <p>Upload any CSV — the agent runs full EDA, engineers features, and delivers clean data automatically</p>
  <span class="pill-badge">📊 Auto EDA</span>
  <span class="pill-badge">🔧 Feature Engineering</span>
  <span class="pill-badge">📈 Final Analysis</span>
  <span class="pill-badge">📄 PDF Report</span>
</div>""", unsafe_allow_html=True)

# ─── Idle (no upload) ─────────────────────────────────────────────────────────
if not st.session_state.csv_name:
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, lbl, desc in [
        (c1, "📁", "Upload CSV",         "Max 10 MB · Any tabular data"),
        (c2, "📊", "Automatic EDA",      "Heatmaps, distributions, scatter plots"),
        (c3, "🔧", "Feature Engineering","Impute, encode, scale, binarize"),
        (c4, "📄", "PDF Report",         "Full downloadable documentation"),
    ]:
        col.markdown(f"""
        <div class="feature-card">
          <div style="font-size:2rem;margin-bottom:0.8rem;">{icon}</div>
          <div style="font-weight:800;color:#f8fafc;margin-bottom:0.4rem;font-size:1rem;">{lbl}</div>
          <div style="color:#94a3b8;font-size:0.84rem;line-height:1.5;">{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ─── Dataset Preview (before analysis starts) ─────────────────────────────────
if not st.session_state.preview_done and st.session_state.csv_path:
    try:
        _pf = pd.read_csv(st.session_state.csv_path)
    except Exception as _e:
        st.error(f"Could not read CSV: {_e}")
        st.stop()

    _nr, _nc = _pf.shape
    _nnum    = int(_pf.select_dtypes(include="number").shape[1])
    _ncat    = int(_pf.select_dtypes(exclude="number").shape[1])

    st.markdown("#### 📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    stat_cards([
        (c1, f"{_nr:,}", "Total Rows",       "#818cf8"),
        (c2, f"{_nc}",   "Total Columns",    "#34d399"),
        (c3, f"{_nnum}", "Numeric Cols",     "#60a5fa"),
        (c4, f"{_ncat}", "Categorical Cols", "#f472b6"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    # Column health badges
    card_header("🔬 Column Overview — AI will auto-clean during Feature Engineering")
    _badge_html = ""
    for _col in _pf.columns:
        _miss  = _pf[_col].isnull().mean()
        _nuniq = _pf[_col].nunique()
        if _nuniq == _nr:
            _cls, _note = "drop", "ID/index"
        elif _nuniq <= 1:
            _cls, _note = "drop", "zero variance"
        elif _miss >= 0.5:
            _cls, _note = "warn", f"{_miss*100:.0f}% missing"
        else:
            _cls, _note = "safe", ""
        _note_html = f'<span class="dr">({_note})</span>' if _note else ""
        _badge_html += f'<span class="col-badge {_cls}">{_col}{_note_html}</span>'
    st.markdown(f'<div style="padding:0.4rem 0 0.5rem;">{_badge_html}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:#64748b;margin-bottom:1.5rem;">'
        '🔴 Dropped  ·  🟡 High Missing  ·  🟢 Healthy</div>',
        unsafe_allow_html=True,
    )

    with st.expander("📋 Preview Dataset", expanded=False):
        st.dataframe(_pf, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Start Full Analysis", type="primary", use_container_width=True):
        st.session_state.original_df  = _pf
        st.session_state.preview_done = True
        st.session_state.messages     = []
        clear_charts()
        st.rerun()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE TABS
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.preview_done:
    tab1, tab2, tab3 = st.tabs([
        "📊 Initial EDA",
        "🔧 Feature Engineering",
        "📈 Final Analysis",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — Initial EDA
    # ═════════════════════════════════════════════════════════════════════════
    with tab1:
        # Always show dataset summary at the top of EDA tab
        try:
            _pf = st.session_state.original_df or pd.read_csv(st.session_state.csv_path)
            _nr, _nc = _pf.shape
            _nnum    = int(_pf.select_dtypes(include="number").shape[1])
            _ncat    = int(_pf.select_dtypes(exclude="number").shape[1])
            
            st.markdown("#### 📊 Dataset Overview (Loaded)")
            c1, c2, c3, c4 = st.columns(4)
            stat_cards([
                (c1, f"{_nr:,}", "Total Rows",       "#818cf8"),
                (c2, f"{_nc}",   "Total Columns",    "#34d399"),
                (c3, f"{_nnum}", "Numeric Cols",     "#60a5fa"),
                (c4, f"{_ncat}", "Categorical Cols", "#f472b6"),
            ])
            with st.expander("📂 Preview Original Dataset", expanded=False):
                st.dataframe(_pf, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
        except Exception:
            pass

        _idle = (
            not st.session_state.done
            and not st.session_state.awaiting_user
            and not st.session_state.phase2_ready
            and not st.session_state.eda_done
            and not st.session_state.fe_ready
            and not st.session_state.fe_done
            and not st.session_state.final_ready
        )

        # Phase 1 — Read & decide clarification
        if _idle:
            with st.spinner("🤖 Reading dataset and checking for clarification needs…"):
                try:
                    if not st.session_state.messages:
                        st.session_state.messages = [HumanMessage(
                            content=f"Dataset '{st.session_state.csv_name}' has been uploaded. "
                                    "Read the data info and decide if you have clarification questions."
                        )]
                        clear_charts()
                    res = run_analysis_graph(st.session_state.csv_path, st.session_state.messages, phase="read")
                    st.session_state.messages   = res["messages"]
                    st.session_state.eda_result = res
                    _accumulate_tokens(res)
                    last = res["messages"][-1].content.strip()
                    if "READY_TO_ANALYZE" in last:
                        st.session_state.phase2_ready = True
                    else:
                        st.session_state.awaiting_user = True
                except Exception as e:
                    st.session_state.eda_result = {"success": False, "answer": str(e), "error": str(e)}
                    st.session_state.eda_done   = True
            st.rerun()

        # Phase 1 MCQ — awaiting clarification
        elif st.session_state.awaiting_user and not st.session_state.eda_done:
            parsed = extract_mcq_json(st.session_state.messages[-1].content)
            if parsed and parsed.get("clarification_needed"):
                st.session_state["eda_mcq_questions"] = parsed.get("questions", [])

            if st.session_state.get("eda_mcq_questions"):
                _, _, summary = render_mcq_card(
                    title="AI Needs Your Input",
                    subtitle="Help the agent understand your dataset to produce the best analysis.",
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

        # Phase 2 — Generate EDA charts
        elif st.session_state.phase2_ready and not st.session_state.eda_done:
            with st.spinner("📊 Generating initial EDA charts…"):
                try:
                    res = run_analysis_graph(st.session_state.csv_path, st.session_state.messages, phase="eda")
                    st.session_state.messages           = res["messages"]
                    st.session_state.eda_result         = res
                    st.session_state.eda_result_initial = res
                    st.session_state.eda_done           = True
                    st.session_state.phase2_ready       = False
                    st.session_state.fe_ready           = True
                    _accumulate_tokens(res)
                except Exception as e:
                    err = {"success": False, "answer": str(e), "error": str(e)}
                    st.session_state.eda_result         = err
                    st.session_state.eda_result_initial = err
                    st.session_state.eda_done           = True
                    st.session_state.done               = True
                    st.session_state.phase2_ready       = False
            st.rerun()

        # EDA complete — show results
        elif st.session_state.eda_done:
            eda_res = st.session_state.get("eda_result_initial") or st.session_state.eda_result
            if not eda_res or not eda_res.get("success"):
                st.error(f"EDA Error: {eda_res.get('error', '') if eda_res else 'No result'}")
            else:

                eda_charts = sorted([
                    p for p in glob.glob(os.path.join(CHARTS_DIR, "*.png"))
                    if "cleaned" not in os.path.basename(p) and "_final" not in os.path.basename(p)
                ])
                if eda_charts:
                    card_header(f"📊 Initial EDA — {len(eda_charts)} Visual Insights")
                    
                    # Chart selection state
                    if "eda_chart_idx" not in st.session_state:
                        st.session_state.eda_chart_idx = 0
                    if "eda_view_mode" not in st.session_state:
                        st.session_state.eda_view_mode = "Step-by-Step"
                    
                    # Navigation Callbacks
                    def _eda_next(): st.session_state.eda_chart_idx = (st.session_state.eda_chart_idx + 1) % len(eda_charts)
                    def _eda_prev(): st.session_state.eda_chart_idx = (st.session_state.eda_chart_idx - 1) % len(eda_charts)
                    
                    # Display Mode Selector
                    vm_c1, vm_c2 = st.columns([2, 1])
                    with vm_c1:
                        st.session_state.eda_view_mode = st.radio(
                            "Display Mode", ["Step-by-Step", "View All Charts"],
                            horizontal=True, key="eda_vtoggle", label_visibility="collapsed"
                        )
                    
                    if st.session_state.eda_view_mode == "Step-by-Step":
                        c_nav1, c_nav2, c_nav3 = st.columns([1, 4, 1])
                        with c_nav1:
                            st.button("◀ Prev", key="btn_eda_prev", use_container_width=True, on_click=_eda_prev)
                        with c_nav2:
                            idx = st.select_slider(
                                "Navigate Steps", options=range(len(eda_charts)),
                                value=min(st.session_state.eda_chart_idx, len(eda_charts)-1),
                                format_func=lambda x: f"Step {x+1}", label_visibility="collapsed"
                            )
                            st.session_state.eda_chart_idx = idx
                        with c_nav3:
                            st.button("Next ▶", key="btn_eda_next", use_container_width=True, on_click=_eda_next)

                        # Current chart
                        current_path = eda_charts[st.session_state.eda_chart_idx]
                        chart_title = os.path.splitext(os.path.basename(current_path))[0].replace("_", " ").title()
                        
                        st.markdown(f"""
                        <div style="background:#1e293b; border:1px solid #334155; border-radius:32px; padding:2.5rem; margin-top:1.5rem; text-align:center; box-shadow: 0 20px 50px rgba(0,0,0,0.5);">
                            <div style="color:#818cf8; font-weight:800; font-size:1.25rem; margin-bottom:1.5rem; letter-spacing:-0.02em;">{chart_title}</div>
                            <img src="data:image/png;base64,{_get_image_base64(current_path)}" style="width:100%; border-radius:16px; margin-bottom:1.5rem; border:1px solid #334155;">
                            <div style="color:#64748b; font-size:0.9rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Visual {st.session_state.eda_chart_idx + 1} of {len(eda_charts)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with open(current_path, "rb") as file:
                            st.download_button(label=f"📥 Download {chart_title}", data=file, file_name=f"{os.path.basename(current_path)}",
                                              mime="image/png", key=f"dl_eda_{st.session_state.eda_chart_idx}", use_container_width=True)
                    else:
                        for idx, path in enumerate(eda_charts):
                            title = os.path.splitext(os.path.basename(path))[0].replace("_", " ").title()
                            st.markdown(f'<div style="margin-top:2rem; border-top:1px solid #1e293b; padding-top:1.5rem;">'
                                        f'<div style="color:#818cf8; font-weight:800; font-size:1rem; margin-bottom:1.2rem;">{idx+1}. {title}</div></div>', unsafe_allow_html=True)
                            st.image(path, use_container_width=True)
                else:
                    st.info("📊 The AI is finalizing the visual insights. If they don't appear shortly, click Refresh.")
                    if st.button("🔄 Refresh EDA Charts", use_container_width=True): st.rerun()

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — Feature Engineering
    # ═════════════════════════════════════════════════════════════════════════
    with tab2:
        if not st.session_state.eda_done:
            st.info("⏳ Awaiting completion of Initial EDA phase…")

        # FE execution
        elif st.session_state.fe_ready and not st.session_state.fe_done and not st.session_state.awaiting_fe_mcq:
            with st.spinner("🔧 AI is planning and applying feature engineering…"):
                try:
                    res = run_analysis_graph(st.session_state.csv_path, st.session_state.messages, phase="fe")
                    st.session_state.messages  = res["messages"]
                    st.session_state.fe_result = res
                    _accumulate_tokens(res)
                    last_fe = res["messages"][-1].content.strip()
                    parsed_fe = extract_mcq_json(last_fe)

                    if parsed_fe and parsed_fe.get("clarification_needed"):
                        st.session_state["fe_mcq_questions"] = parsed_fe.get("questions", [])
                        st.session_state.awaiting_fe_mcq     = True
                    else:
                        st.session_state.fe_done     = True
                        st.session_state.fe_ready    = False
                        st.session_state.final_ready = True
                        st.session_state.fe_report   = last_fe
                        st.session_state.cleaned_df  = pd.read_csv(st.session_state.csv_path)
                except Exception as _fe_err:
                    st.session_state.fe_result   = {"success": False, "answer": str(_fe_err), "error": str(_fe_err)}
                    st.session_state.fe_done     = True
                    st.session_state.fe_ready    = False
                    st.session_state.fe_report   = f"Error: {_fe_err}"
                    st.session_state.final_ready = True
            st.rerun()

        # FE MCQ
        elif st.session_state.awaiting_fe_mcq and not st.session_state.fe_done:
            _, _, fe_summary = render_mcq_card(
                title="Feature Engineering — Clarification",
                subtitle="The AI needs your input before cleaning the data.",
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

        # FE complete — show results
        elif st.session_state.fe_done:
            fe_report   = st.session_state.get("fe_report", "")
            original_df = st.session_state.get("original_df")
            cleaned_df  = st.session_state.get("cleaned_df")

            # Parse operation bullets from FE report
            fe_ops_lines = []
            if fe_report:
                in_binarize = False
                for raw in fe_report.split("\n"):
                    line = raw.strip()
                    if not line or line.startswith("{"):
                        continue
                    if "Feature Extraction (Binarization) Complete" in line:
                        in_binarize = True
                        continue
                    if in_binarize:
                        if line.startswith("-") or line.startswith("Original") or line.startswith("New shape"):
                            continue
                        else:
                            in_binarize = False
                    cl = line.lstrip("-•*·▸▹ ").strip()
                    if any(kw in cl.lower() for kw in ["drop","encode","missing","fill","binariz","scale","impute","scaled","encoded"]):
                        if cl and cl not in fe_ops_lines:
                            fe_ops_lines.append(cl)

            # ── SECTION 1: Pre-FE Dataset Analysis ──────────────────────────
            section_banner("1", "📋 Pre-FE Dataset Analysis", "#818cf8")

            if original_df is not None:
                _or, _oc   = original_df.shape
                _onum      = int(original_df.select_dtypes(include="number").shape[1])
                _ocat      = int(original_df.select_dtypes(exclude="number").shape[1])
                _omiss     = int((original_df.isnull().sum() > 0).sum())
                _omiss_tot = int(original_df.isnull().sum().sum())

                c1, c2, c3, c4, c5 = st.columns(5)
                stat_cards([
                    (c1, f"{_or:,}",    "Rows",           "#818cf8"),
                    (c2, f"{_oc}",      "Columns",        "#34d399"),
                    (c3, f"{_onum}",    "Numeric",        "#60a5fa"),
                    (c4, f"{_ocat}",    "Categorical",    "#f472b6"),
                    (c5, f"{_omiss}",   "With Missing",   "#fbbf24"),
                ])
                st.markdown("<br>", unsafe_allow_html=True)

                # Missing values table
                miss_any = [(c, int(original_df[c].isnull().sum()),
                             round(original_df[c].isnull().mean()*100,1),
                             str(original_df[c].dtype),
                             int(original_df[c].nunique()))
                            for c in original_df.columns if original_df[c].isnull().sum() > 0]

                card_header("❓ Missing Values — Original Dataset")
                if miss_any:
                    rows_html = ""
                    for col, cnt, pct, dtype, uq in miss_any:
                        sev = "#f87171" if pct > 60 else "#fbbf24" if pct > 20 else "#4ade80"
                        bar = (f'<div style="display:flex;align-items:center;gap:0.4rem;">'
                               f'<div style="width:100px;height:6px;background:#0f172a;border-radius:3px;border:1px solid #334155;">'
                               f'<div style="width:{min(pct,100)}%;height:6px;background:{sev};border-radius:3px;"></div></div>'
                               f'<span style="color:{sev};font-size:0.85rem;font-weight:800;">{pct}%</span></div>')
                        rows_html += (f'<tr><td><code style="color:#818cf8;font-weight:700;">{col}</code></td>'
                                      f'<td style="color:#94a3b8;font-weight:600;">{dtype}</td>'
                                      f'<td style="color:#94a3b8;font-weight:600;">{uq:,}</td>'
                                      f'<td style="color:{sev};font-weight:800;">{cnt:,}</td>'
                                      f'<td>{bar}</td></tr>')
                    st.markdown(
                        f'<table class="otable" style="width:100%;margin-bottom:1.2rem;">'
                        f'<thead><tr><th>Column</th><th>Type</th><th>Unique</th>'
                        f'<th style="color:#f87171;">Missing Count</th><th>Missing %</th></tr></thead>'
                        f'<tbody>{rows_html}</tbody></table>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="insight" style="border-left-color:#34d399;">✅ No missing values found in the original dataset.</div>',
                        unsafe_allow_html=True,
                    )

                # Full column summary
                card_header("📊 Column Summary — Original Data")
                rows2 = ""
                for col in original_df.columns:
                    dtype = str(original_df[col].dtype)
                    uq    = int(original_df[col].nunique())
                    mc    = int(original_df[col].isnull().sum())
                    mp    = round(original_df[col].isnull().mean()*100, 1)
                    t_badge = (f'<span style="color:#818cf8;font-weight:800;font-size:0.8rem;">numeric</span>' if "int" in dtype or "float" in dtype
                               else f'<span style="color:#f472b6;font-weight:800;font-size:0.8rem;">categorical</span>' if dtype == "object"
                               else f'<span style="color:#94a3b8;font-weight:800;font-size:0.8rem;">{dtype}</span>')
                    mc_cell = f'<b style="color:#f87171;">{mc}</b>' if mc > 0 else f'<b style="color:#4ade80;">0</b>'
                    rows2 += (f'<tr><td><code style="color:#c7d2fe;font-weight:700;">{col}</code></td>'
                              f'<td>{t_badge}</td>'
                              f'<td style="color:#e2e8f0;font-weight:600;">{uq:,}</td>'
                              f'<td>{mc_cell}</td>'
                              f'<td style="color:#94a3b8;font-weight:600;">{mp}%</td></tr>')
                st.markdown(
                    f'<table class="otable" style="width:100%;margin-bottom:1.5rem;">'
                    f'<thead><tr><th>Column</th><th>Type</th><th>Unique Values</th>'
                    f'<th>Missing</th><th>Missing %</th></tr></thead>'
                    f'<tbody>{rows2}</tbody></table>',
                    unsafe_allow_html=True,
                )

            # ── SECTION 2: FE Operations Log ────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            section_banner("2", "⚙️ Feature Engineering Operations Log", "#34d399")

            _op_drop, _op_impute, _op_bin, _op_encode, _op_scale = [], [], [], [], []
            for op in fe_ops_lines:
                ol = op.lower()
                if   "drop" in ol:                                  _op_drop.append(op)
                elif "impute" in ol or "fill" in ol or "missing" in ol: _op_impute.append(op)
                elif "binariz" in ol:                               _op_bin.append(op)
                elif "encod" in ol:                                 _op_encode.append(op)
                else:                                               _op_scale.append(op)

            def _op_row(icon, title, explanation, ops, color):
                details = ("".join(f"<div style='margin-bottom:0.4rem;font-weight:700;color:#f1f5f9;'>• {o}</div>" for o in ops)
                           if ops else '<span style="color:#64748b;"><i>No action needed.</i></span>')
                return f'''<tr>
                  <td style="vertical-align:top;padding:1.4rem 1.5rem 1.4rem 0;">
                    <div style="color:{color};font-weight:800;font-size:1.05rem;margin-bottom:0.4rem;">{icon} {title}</div>
                    <div style="color:#94a3b8;font-size:0.88rem;line-height:1.5;font-weight:600;">{explanation}</div>
                  </td>
                  <td style="vertical-align:top;padding:1.4rem 0;color:#f8fafc;font-size:0.93rem;">{details}</td>
                </tr>'''

            st.markdown(
                '<table class="otable" style="width:100%;margin-bottom:1.5rem;">'
                '<thead><tr><th style="width:38%;">Data Cleaning Step</th><th>Changes Applied by AI</th></tr></thead>'
                '<tbody>'
                + _op_row("🗑️", "Removed Useless Columns",    "ID columns, zero-variance, high sparsity (>60% missing)",           _op_drop,  "#f87171")
                + _op_row("🩹", "Filled Missing Data",         "Mean/median for numeric, mode for categorical",                     _op_impute,"#fbbf24")
                + _op_row("⚡", "Binarized Features",          "Converted numeric columns to binary (1/0) indicators",              _op_bin,   "#34d399")
                + _op_row("🔄", "Encoded Categorical Columns", "Label-encoded text columns so the model can process them",          _op_encode,"#60a5fa")
                + _op_row("📏", "Scaled Numeric Features",     "StandardScaler applied — all numeric columns normalized",           _op_scale, "#6366f1")
                + '</tbody></table>',
                unsafe_allow_html=True,
            )

            if not fe_ops_lines:
                st.info("ℹ️ No additional cleaning operations were needed — the dataset was already clean.")

            # ── SECTION 3: Post-FE Comparison & Preview ──────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            section_banner("3", "✅ Post-FE Dataset Summary", "#a78bfa")

            if cleaned_df is not None:
                _cr, _cc     = cleaned_df.shape
                _cnum        = int(cleaned_df.select_dtypes(include="number").shape[1])
                _ccat        = int(cleaned_df.select_dtypes(exclude="number").shape[1])
                _cmiss       = int(cleaned_df.isnull().sum().sum())
                _bin_count   = len([c for c in cleaned_df.columns if c.endswith("_bin")])

                # Before vs After table
                if original_df is not None:
                    _or2, _oc2 = original_df.shape
                    def _chg(before, after, good="less"):
                        diff = after - before
                        if diff == 0:
                            return f'<span style="color:#94a3b8;">→ {after}</span>'
                        better = diff < 0 if good == "less" else diff > 0
                        c = "#34d399" if better else "#f87171"
                        arrow = "↓" if diff < 0 else "↑"
                        return f'<span style="color:{c};">{before} {arrow} {after}</span>'

                    card_header("📊 Before vs After Feature Engineering")
                    st.markdown(
                        f'<table class="otable" style="width:100%;margin-bottom:1rem;">'
                        f'<thead><tr><th>Metric</th><th>Before FE</th><th>After FE</th><th>Change</th></tr></thead>'
                        f'<tbody>'
                        f'<tr><td>Rows</td><td>{_or2:,}</td><td>{_cr:,}</td><td>{_chg(_or2,_cr,"same")}</td></tr>'
                        f'<tr><td>Columns</td><td>{_oc2}</td><td>{_cc}</td><td>{_chg(_oc2,_cc,"same")}</td></tr>'
                        f'<tr><td>Missing Cells (total)</td><td style="color:#f87171;">{int(original_df.isnull().sum().sum()):,}</td>'
                        f'<td style="color:#34d399;">{_cmiss:,}</td><td>{_chg(int(original_df.isnull().sum().sum()),_cmiss)}</td></tr>'
                        f'<tr><td>Categorical Columns</td><td>{int(original_df.select_dtypes(exclude="number").shape[1])}</td>'
                        f'<td>{_ccat}</td><td>{_chg(int(original_df.select_dtypes(exclude="number").shape[1]),_ccat)}</td></tr>'
                        f'<tr><td>Numeric Columns</td><td>{int(original_df.select_dtypes(include="number").shape[1])}</td>'
                        f'<td>{_cnum}</td><td>{_chg(int(original_df.select_dtypes(include="number").shape[1]),_cnum,"more")}</td></tr>'
                        f'<tr><td>New Binary (_bin) Features</td><td>0</td><td style="color:#34d399;">{_bin_count}</td>'
                        f'<td><span style="color:#34d399;">+{_bin_count}</span></td></tr>'
                        f'</tbody></table>',
                        unsafe_allow_html=True,
                    )

                c1, c2, c3, c4 = st.columns(4)
                stat_cards([
                    (c1, f"{_cr:,}", "Rows",          "#818cf8"),
                    (c2, f"{_cc}",   "Columns",       "#34d399"),
                    (c3, f"{_cnum}", "Numeric Cols",  "#60a5fa"),
                    (c4, f"{_cmiss}", "Missing Cells", "#34d399" if _cmiss == 0 else "#fbbf24"),
                ])
                st.markdown("<br>", unsafe_allow_html=True)

                card_header("🔍 Cleaned Dataset Preview")
                st.dataframe(cleaned_df, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # Cleaned heatmap — shown ONLY here in Tab 2
            fe_heatmap = os.path.join(CHARTS_DIR, "heatmap_correlation_cleaned.png")
            if os.path.exists(fe_heatmap):
                card_header("🌡️ Cleaned Data — Correlation Heatmap")
                st.image(fe_heatmap, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # ── Downloads ────────────────────────────────────────────────────
            section_banner("4", "📥 Downloads", "#60a5fa")
            dl_c1_t2, dl_c2_t2 = st.columns(2)

            with dl_c1_t2:
                if cleaned_df is not None:
                    st.download_button(
                        label="📥 Download Cleaned Dataset (CSV)",
                        data=cleaned_df.to_csv(index=False).encode("utf-8"),
                        file_name="cleaned_dataset.csv",
                        mime="text/csv",
                        key="dl_fe_csv",
                        use_container_width=True,
                    )

            with dl_c2_t2:
                # PDF report
                pdf_bytes_fe = generate_pdf_report(
                    original_df  = st.session_state.get("original_df"),
                    cleaned_df   = cleaned_df,
                    fe_ops_lines = fe_ops_lines,
                    insights     = [],
                    charts_dir   = CHARTS_DIR,
                    csv_name     = st.session_state.csv_name,
                )
                if pdf_bytes_fe:
                    st.download_button(
                        label="📄 Download Analysis Report (PDF)",
                        data=pdf_bytes_fe,
                        file_name="ai_datascientist_report.pdf",
                        mime="application/pdf",
                        key="dl_fe_pdf",
                        use_container_width=True,
                    )
    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — Final Analysis
    # ═════════════════════════════════════════════════════════════════════════
    with tab3:
        if not st.session_state.fe_done:
            st.info("⏳ Awaiting completion of Feature Engineering phase…")

        elif st.session_state.final_ready and not st.session_state.done:
            with st.spinner("📈 Generating final optimized charts on cleaned data…"):
                try:
                    res = run_analysis_graph(st.session_state.csv_path, st.session_state.messages, phase="final")
                    st.session_state.messages    = res["messages"]
                    st.session_state.done        = True
                    st.session_state.final_ready = False
                    _accumulate_tokens(res)
                except Exception as e:
                    st.session_state.done        = True
                    st.session_state.final_ready = False
            st.rerun()

        elif st.session_state.done:
            try:
                cleaned_df = (
                    st.session_state.cleaned_df
                    if st.session_state.cleaned_df is not None
                    else pd.read_csv(st.session_state.csv_path)
                )
                bin_cols = [(c.replace("_bin", ""), c) for c in cleaned_df.columns if c.endswith("_bin")]
            except Exception:
                cleaned_df = None
                bin_cols   = []

            # ── Key Insights ─────────────────────────────────────────────────
            last_content = st.session_state.messages[-1].content if st.session_state.messages else ""
            insight_lines = []
            if "FINAL_COMPLETE:" in last_content:
                summary_text  = last_content.split("FINAL_COMPLETE:")[-1].strip()
                insight_lines = [l.strip("•-* ").strip() for l in summary_text.split("\n") if l.strip()][:5]
                bullets       = "".join(f"<li style='margin-bottom:0.35rem;'>{l}</li>" for l in insight_lines if l)
                st.markdown(
                    f'<div class="insight"><strong>🤖 Key Insights from Cleaned Data:</strong>'
                    f'<ul style="margin:0.5rem 0 0 1.2rem;">{bullets}</ul></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)

            # ── Final Charts (suffix=_final) ──────────────────────────────────
            final_charts = sorted([
                p for p in glob.glob(os.path.join(CHARTS_DIR, "*.png"))
                if "_final" in os.path.basename(p)
            ])
            if final_charts:
                card_header(f"📈 Final Analysis — {len(final_charts)} Optimized Graphs")
                
                # Chart selection state
                if "final_chart_idx" not in st.session_state:
                    st.session_state.final_chart_idx = 0
                if "final_view_mode" not in st.session_state:
                    st.session_state.final_view_mode = "Step-by-Step"
                
                # Callbacks
                def _f_next(): st.session_state.final_chart_idx = (st.session_state.final_chart_idx + 1) % len(final_charts)
                def _f_prev(): st.session_state.final_chart_idx = (st.session_state.final_chart_idx - 1) % len(final_charts)
                
                # Display Mode Selector
                fvm_c1, fvm_c2 = st.columns([2, 1])
                with fvm_c1:
                    st.session_state.final_view_mode = st.radio(
                        "Final Preview Mode", ["Step-by-Step", "View All Charts"],
                        horizontal=True, key="f_vtoggle", label_visibility="collapsed"
                    )

                if st.session_state.final_view_mode == "Step-by-Step":
                    f_nav1, f_nav2, f_nav3 = st.columns([1, 4, 1])
                    with f_nav1:
                        st.button("◀ Prev", key="btn_f_prev", use_container_width=True, on_click=_f_prev)
                    with f_nav2:
                        f_idx = st.select_slider(
                            "Final Navigator", options=range(len(final_charts)),
                            value=min(st.session_state.final_chart_idx, len(final_charts)-1),
                            format_func=lambda x: f"Result {x+1}", label_visibility="collapsed"
                        )
                        st.session_state.final_chart_idx = f_idx
                    with f_nav3:
                        st.button("Next ▶", key="btn_f_next", use_container_width=True, on_click=_f_next)

                    # Current final chart
                    f_current_path = final_charts[st.session_state.final_chart_idx]
                    f_chart_title = os.path.splitext(os.path.basename(f_current_path))[0].replace("_", " ").title()
                    
                    st.markdown(f"""
                    <div style="background:#1e293b; border:1px solid #334155; border-radius:32px; padding:2.5rem; margin-top:1.5rem; text-align:center; box-shadow: 0 20px 50px rgba(0,0,0,0.5);">
                        <div style="color:#f472b6; font-weight:800; font-size:1.25rem; margin-bottom:1.5rem; letter-spacing:-0.02em;">{f_chart_title}</div>
                        <img src="data:image/png;base64,{_get_image_base64(f_current_path)}" style="width:100%; border-radius:16px; margin-bottom:1.5rem; border:1px solid #334155;">
                        <div style="color:#64748b; font-size:0.9rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Optimized {st.session_state.final_chart_idx + 1} of {len(final_charts)}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with open(f_current_path, "rb") as file:
                        st.download_button(label=f"📥 Download {f_chart_title}", data=file, file_name=f"{os.path.basename(f_current_path)}",
                                          mime="image/png", key=f"dl_f_{st.session_state.final_chart_idx}", use_container_width=True)
                else:
                    for idx, path in enumerate(final_charts):
                        title = os.path.splitext(os.path.basename(path))[0].replace("_", " ").title()
                        st.markdown(f'<div style="margin-top:2rem; border-top:1px solid #1e293b; padding-top:1.5rem;">'
                                    f'<div style="color:#f472b6; font-weight:800; font-size:1rem; margin-bottom:1.2rem;">{idx+1}. {title}</div></div>', unsafe_allow_html=True)
                        st.image(path, use_container_width=True)
            else:
                st.info("📈 Final analysis charts are being generated. Click Refresh if they don't appear.")
                if st.button("🔄 Refresh Final Charts", use_container_width=True): st.rerun()

            # ── Binary Feature Summary ────────────────────────────────────────
            if bin_cols and cleaned_df is not None:
                card_header("⚡ Binarized Features Summary")
                rows_html = ""
                for orig, binc in bin_cols:
                    if orig in cleaned_df.columns:
                        ones  = int(cleaned_df[binc].sum())
                        zeros = int(len(cleaned_df) - ones)
                        pct   = round(ones / len(cleaned_df) * 100, 1) if len(cleaned_df) else 0
                        o_min = round(float(cleaned_df[orig].min()), 3)
                        o_max = round(float(cleaned_df[orig].max()), 3)
                        o_mean= round(float(cleaned_df[orig].mean()), 3)
                    else:
                        zeros = ones = pct = o_min = o_max = o_mean = "—"
                    bar = ""
                    if isinstance(pct, (int, float)):
                        bar = (f'<div style="display:flex;align-items:center;gap:0.4rem;">'
                               f'<div style="flex:1;height:6px;background:#0f172a;border-radius:3px;border:1px solid #334155;">'
                               f'<div style="width:{pct}%;height:6px;background:#818cf8;border-radius:3px;"></div></div>'
                               f'<span style="font-size:0.75rem;color:#818cf8;font-weight:800;">{pct}%</span></div>')
                    rows_html += (
                        f"<tr>"
                        f'<td><code style="color:#818cf8;font-weight:700;">{orig}</code></td>'
                        f'<td><code style="color:#4ade80;font-weight:700;">{binc}</code></td>'
                        f'<td style="color:#e2e8f0;font-weight:600;">{o_min}</td>'
                        f'<td style="color:#e2e8f0;font-weight:600;">{o_max}</td>'
                        f'<td style="color:#e2e8f0;font-weight:600;">{o_mean}</td>'
                        f'<td style="color:#f87171;font-weight:800;">{zeros}</td>'
                        f'<td style="color:#4ade80;font-weight:800;">{ones}</td>'
                        f'<td>{bar}</td></tr>'
                    )
                st.markdown(
                    f'<table class="otable" style="width:100%;">'
                    f'<thead><tr><th>Original Column</th><th>Binary Column</th>'
                    f'<th>Min</th><th>Max</th><th>Mean</th>'
                    f'<th style="color:#f87171;">Zeros</th><th style="color:#4ade80;">Ones</th>'
                    f'<th>% Positive Indicator</th></tr></thead>'
                    f'<tbody>{rows_html}</tbody></table>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)

            # ── Final Dataset + Downloads ─────────────────────────────────────
            if cleaned_df is not None:
                _nr2, _nc2 = cleaned_df.shape
                card_header("📂 Final Cleaned Dataset")
                c1, c2, c3 = st.columns(3)
                stat_cards([
                    (c1, f"{_nr2:,}",       "Total Rows",      "#818cf8"),
                    (c2, f"{_nc2}",          "Total Columns",   "#34d399"),
                    (c3, f"{len(bin_cols)}", "Binary Features", "#f472b6"),
                ])
                st.markdown("<br>", unsafe_allow_html=True)

                with st.expander("📋 View full cleaned dataset", expanded=False):
                    st.dataframe(cleaned_df, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                section_banner("📥", "Downloads — Final Analysis", "#f472b6")
                dl_c1_t3, dl_c2_t3 = st.columns(2)

                with dl_c1_t3:
                    st.download_button(
                        label="📥 Download Final Cleaned Dataset (CSV)",
                        data=cleaned_df.to_csv(index=False).encode("utf-8"),
                        file_name="final_cleaned_dataset.csv",
                        mime="text/csv",
                        key="dl_final_csv_t3",
                        use_container_width=True,
                    )

                with dl_c2_t3:
                    # Full report PDF — include insights and final charts
                    pdf_bytes_final = generate_pdf_report(
                        original_df  = st.session_state.get("original_df"),
                        cleaned_df   = cleaned_df,
                        fe_ops_lines = fe_ops_lines,
                        insights     = insight_lines,
                        charts_dir   = CHARTS_DIR,
                        csv_name     = st.session_state.csv_name,
                    )
                    if pdf_bytes_final:
                        st.download_button(
                            label="📄 Download Full Analysis Report (PDF)",
                            data=pdf_bytes_final,
                            file_name="ai_datascientist_full_report.pdf",
                            mime="application/pdf",
                            key="dl_final_pdf_t3",
                            use_container_width=True,
                        )
                    else:
                        st.markdown(
                            '<div style="font-size:0.8rem;color:rgba(71,85,105,0.5);padding:0.6rem;">'
                            '📄 PDF report generation failed or fpdf2 is missing.</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No final data available.")


