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
            st.success("✅ EDA Complete")
            st.markdown(f"""
            <div style="font-size:0.79rem;color:rgba(255,255,255,0.5);line-height:2;">
            🔢 Steps: <b style="color:#818cf8;">{len(r['steps'])}</b><br>
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

# ─── Auto-run EDA ────────────────────────────────────────────────────────────
if not st.session_state.done:
    with st.spinner("🤖 Agent is running full EDA — generating all charts…"):
        try:
            from agent import run_analysis, AUTO_EDA_QUERY
            from tools import load_csv_into_tensor, generate_eda_summary, \
                              analyze_missing_values, detect_outliers_report

            clear_charts()
            result = run_analysis(
                query=AUTO_EDA_QUERY,
                csv_path=st.session_state.csv_path,
            )
            st.session_state.eda_result = result
            st.session_state.done = True

            # Pre-fetch JSON data for tables (bypass agent for speed)
            try:
                load_csv_into_tensor(st.session_state.csv_path)
                st.session_state.stats_json   = json.loads(generate_eda_summary.invoke(""))
                st.session_state.missing_json = json.loads(analyze_missing_values.invoke(""))
                st.session_state.outlier_json = json.loads(detect_outliers_report.invoke(""))
            except Exception:
                pass

        except Exception as e:
            st.session_state.eda_result = {
                "success": False, "answer": str(e),
                "steps": [], "chart_paths": [], "error": str(e)
            }
            st.session_state.done = True
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# REPORT: Tabbed EDA Dashboard
# ─────────────────────────────────────────────────────────────────────────────
result = st.session_state.eda_result
if not result:
    st.stop()

if not result["success"]:
    st.error(f"**EDA Error:** {result.get('error', result['answer'])}")
    st.stop()

# ── Top metric bar ───────────────────────────────────────────────────────────
n_charts = len(glob.glob(os.path.join(os.path.dirname(__file__), "charts", "*.png")))
stats = st.session_state.stats_json or {}
missing = st.session_state.missing_json or {}
outliers = st.session_state.outlier_json or {}

n_cols_detected = len(stats)
total_missing   = sum(v.get("missing_count", 0) for v in missing.values())
total_outliers  = sum(v.get("outlier_count", 0) for v in outliers.values()
                      if isinstance(v, dict) and "outlier_count" in v)

m1, m2, m3, m4, m5 = st.columns(5)
for col, val, lbl, clr in [
    (m1, st.session_state.csv_name[:14] + "…" if len(st.session_state.csv_name) > 14
         else st.session_state.csv_name, "Dataset", "#818cf8"),
    (m2, f"{n_cols_detected}", "Numeric Cols", "#60a5fa"),
    (m3, f"{total_missing}", "Missing Cells", "#fbbf24" if total_missing > 0 else "#34d399"),
    (m4, f"{total_outliers}", "Outliers (IQR)", "#f87171" if total_outliers > 0 else "#34d399"),
    (m5, f"{n_charts}", "Charts Generated", "#34d399"),
]:
    with col:
        st.markdown(f"""
        <div class="scard">
            <span class="v" style="color:{clr};">{val}</span>
            <span class="l">{lbl}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Key insight (2 sentences from agent) ────────────────────────────────────
if result["answer"].strip():
    st.markdown(f'<div class="insight">💡 {result["answer"].strip()}</div>',
                unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Stats",
    "📈 Distributions",
    "📦 Outliers",
    "🌡️ Correlations",
    "🔍 Data Quality",
])

# ── TAB 1: Descriptive Statistics ────────────────────────────────────────────
with tab1:
    st.markdown("#### 📊 Descriptive Statistics — All Columns")
    if stats:
        # Build an HTML table
        rows_html = ""
        for col_name, s in stats.items():
            missing_pct = s.get("missing_pct", 0)
            badge_cls = ("badge-red" if missing_pct > 20
                         else "badge-amber" if missing_pct > 0
                         else "badge-green")
            rows_html += f"""<tr>
                <td><b>{col_name}</b></td>
                <td>{s.get('count', '—')}</td>
                <td>{s.get('min', '—')}</td>
                <td>{s.get('max', '—')}</td>
                <td>{s.get('mean', '—')}</td>
                <td>{s.get('median', '—')}</td>
                <td>{s.get('std', '—')}</td>
                <td class="{badge_cls}">{missing_pct}%</td>
            </tr>"""

        st.markdown(f"""
        <table class="stattbl">
            <thead><tr>
                <th>Column</th><th>Count</th><th>Min</th><th>Max</th>
                <th>Mean</th><th>Median</th><th>Std Dev</th><th>Missing %</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)
    else:
        st.info("Statistics not available.")

    st.markdown("<br>", unsafe_allow_html=True)
    # Bar chart of means
    bar_path = chart_exists("bar_chart.png")
    if bar_path:
        st.markdown("##### Column Means — Bar Chart")
        st.image(bar_path, use_container_width=True)


# ── TAB 2: Distributions ────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 📈 Value Distributions — Histograms per Column")
    st.caption("🟡 Dashed line = Mean &nbsp;|&nbsp; 🟢 Dotted line = Median")
    hist_paths = charts_by_prefix("hist_")
    if hist_paths:
        # Show 2 per row
        for i in range(0, len(hist_paths), 2):
            row = hist_paths[i:i+2]
            cols = st.columns(len(row))
            for col, path in zip(cols, row):
                col_name = os.path.splitext(os.path.basename(path))[0].replace("hist_", "")
                with col:
                    st.image(path, use_container_width=True)
                    st.markdown(f'<div style="text-align:center;font-size:0.77rem;'
                                f'color:rgba(255,255,255,0.4);">{col_name}</div>',
                                unsafe_allow_html=True)
    else:
        st.info("Histograms not generated yet.")


# ── TAB 3: Outliers ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 📦 Outlier Detection — Box Plots + IQR Analysis")

    # Box plot
    box_path = chart_exists("boxplot_all.png")
    if box_path:
        st.markdown("##### Box Plot — All Columns (🔴 dots = outliers)")
        st.image(box_path, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # IQR table
    if outliers:
        st.markdown("##### IQR Outlier Report")
        rows_html = ""
        for col_name, o in outliers.items():
            if not isinstance(o, dict) or "outlier_count" not in o:
                continue
            cnt = o["outlier_count"]
            badge_cls = "badge-red" if cnt > 0 else "badge-green"
            vals = ", ".join(str(v) for v in o.get("outlier_values", [])[:5])
            if len(o.get("outlier_values", [])) > 5:
                vals += "…"
            rows_html += f"""<tr>
                <td><b>{col_name}</b></td>
                <td>{o.get('q1','—')}</td>
                <td>{o.get('q3','—')}</td>
                <td>{o.get('iqr','—')}</td>
                <td>{o.get('lower_fence','—')}</td>
                <td>{o.get('upper_fence','—')}</td>
                <td class="{badge_cls}">{cnt}</td>
                <td style="color:#f87171;font-size:0.78rem;">{vals if vals else '—'}</td>
            </tr>"""

        if rows_html:
            st.markdown(f"""
            <table class="otable">
                <thead><tr>
                    <th>Column</th><th>Q1</th><th>Q3</th><th>IQR</th>
                    <th>Lower Fence</th><th>Upper Fence</th><th>Outliers</th><th>Values</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>""", unsafe_allow_html=True)
    else:
        st.info("Outlier data not available.")


# ── TAB 4: Correlations ──────────────────────────────────────────────────────
with tab4:
    st.markdown("#### 🌡️ Feature Correlations")

    heatmap_path = chart_exists("heatmap_chart.png")
    scatter_path = chart_exists("scatter_chart.png")

    if heatmap_path:
        st.markdown("##### Pearson Correlation Heatmap")
        st.caption("🔴 Strong positive  ·  🔵 Strong negative  ·  ⚪ No correlation")
        st.image(heatmap_path, use_container_width=True)

    if scatter_path:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Scatter Plot — First Two Columns Relationship")
        st.image(scatter_path, use_container_width=True)

    if not heatmap_path and not scatter_path:
        st.info("Correlation charts not generated.")


# ── TAB 5: Data Quality ──────────────────────────────────────────────────────
with tab5:
    st.markdown("#### 🔍 Data Quality Report")

    mv_path = chart_exists("missing_values.png")
    if mv_path:
        st.markdown("##### Missing Values per Column")
        st.caption("🟢 < 5%  ·  🟡 5–20%  ·  🔴 > 20%")
        st.image(mv_path, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if missing:
        st.markdown("##### Missing Value Details")
        rows_html = ""
        for col_name, m in missing.items():
            cnt = m.get("missing_count", 0)
            pct = m.get("missing_pct", 0)
            badge_cls = ("badge-red" if pct > 20
                         else "badge-amber" if pct > 0
                         else "badge-green")
            bar_w = min(int(pct), 100)
            bar = f'<div style="background:#f87171;height:6px;width:{bar_w}%;border-radius:3px;"></div>' \
                  if pct > 20 else \
                  f'<div style="background:#fbbf24;height:6px;width:{bar_w}%;border-radius:3px;"></div>' \
                  if pct > 0 else \
                  f'<div style="background:#34d399;height:6px;width:4px;border-radius:3px;"></div>'
            rows_html += f"""<tr>
                <td><b>{col_name}</b></td>
                <td class="{badge_cls}">{cnt} cells</td>
                <td>{bar} <span style="font-size:0.75rem;">{pct}%</span></td>
            </tr>"""

        st.markdown(f"""
        <table class="otable">
            <thead><tr><th>Column</th><th>Missing Count</th><th>Visual</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)
