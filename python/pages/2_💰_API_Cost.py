"""
pages/2_💰_API_Cost.py — API token usage & cost tracking dashboard.
Reads from st.session_state.token_usage set by the main app.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from frontend.styles import inject_styles
from frontend.components import calc_cost, format_cost

st.set_page_config(
    page_title="API Cost — AI Data Scientist",
    page_icon="💰",
    layout="wide",
)
inject_styles()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>💰 API Cost Dashboard</h1>
  <p>Real-time token usage and cost breakdown for every phase of your analysis</p>
  <span class="pill-badge">GPT-4o-mini</span>
  <span class="pill-badge">Input · $0.15 / 1M tokens</span>
  <span class="pill-badge">Output · $0.60 / 1M tokens</span>
</div>
""", unsafe_allow_html=True)

# ── No data state ─────────────────────────────────────────────────────────────
token_usage = st.session_state.get("token_usage", [])

if not token_usage:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;">
      <div style="font-size:3rem;">📊</div>
      <div style="font-size:1.1rem;font-weight:600;color:rgba(255,255,255,0.6);margin-top:1rem;">
        No analysis run yet
      </div>
      <div style="font-size:0.85rem;color:rgba(255,255,255,0.3);margin-top:0.5rem;">
        Upload a dataset and run the analysis pipeline to see token usage here.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Aggregate totals ──────────────────────────────────────────────────────────
total_input  = sum(u.get("input_tokens",  0) for u in token_usage)
total_output = sum(u.get("output_tokens", 0) for u in token_usage)
total_all    = total_input + total_output
total_cost   = calc_cost(total_input, total_output)

# ── Summary Metrics ───────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📈 Session Totals</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

for col, val, label, color in [
    (c1, f"{total_input:,}",  "Input Tokens",   "#818cf8"),
    (c2, f"{total_output:,}", "Output Tokens",  "#34d399"),
    (c3, f"{total_all:,}",    "Total Tokens",   "#60a5fa"),
    (c4, format_cost(total_cost), "Est. Cost",  "#f472b6"),
]:
    col.markdown(
        f'<div class="cost-card">'
        f'<span class="big" style="color:{color};">{val}</span>'
        f'<span class="sub">{label}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Progress bars ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🔢 Token Distribution</div>', unsafe_allow_html=True)
if total_all > 0:
    in_pct  = round(total_input  / total_all * 100, 1)
    out_pct = round(total_output / total_all * 100, 1)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-bottom:0.3rem;">'
            f'Input Tokens — {in_pct}%</div>', unsafe_allow_html=True)
        st.progress(in_pct / 100)
    with c2:
        st.markdown(
            f'<div style="font-size:0.8rem;color:rgba(255,255,255,0.5);margin-bottom:0.3rem;">'
            f'Output Tokens — {out_pct}%</div>', unsafe_allow_html=True)
        st.progress(out_pct / 100)

st.markdown("<br>", unsafe_allow_html=True)

# ── Per-Phase Breakdown ───────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">⚙️ Per-Phase Breakdown</div>', unsafe_allow_html=True)

PHASE_META = {
    "read":  ("📖 Read",              "#818cf8", "130,140,248"),
    "eda":   ("📊 EDA",               "#34d399", "52,211,153"),
    "fe":    ("🔧 Feature Engineering","#fbbf24", "251,191,36"),
    "final": ("📈 Final Analysis",     "#f472b6", "244,114,182"),
}

rows_html = ""
for u in token_usage:
    phase = u.get("phase", "?")
    inp   = u.get("input_tokens",  0)
    out   = u.get("output_tokens", 0)
    tot   = inp + out
    cost  = calc_cost(inp, out)
    label, color, rgb = PHASE_META.get(phase, (phase, "#94a3b8", "148,163,184"))
    rows_html += (
        f"<tr>"
        f'<td><span class="phase-badge" style="background:rgba({rgb},0.15);color:{color};">{label}</span></td>'
        f'<td style="color:#818cf8;font-weight:600;">{inp:,}</td>'
        f'<td style="color:#34d399;font-weight:600;">{out:,}</td>'
        f'<td style="color:#60a5fa;font-weight:600;">{tot:,}</td>'
        f'<td style="color:#f472b6;font-weight:600;">{format_cost(cost)}</td>'
        f"</tr>"
    )

st.markdown(f"""
<table class="otable" style="width:100%;margin-bottom:1.5rem;">
  <thead><tr>
    <th>Phase</th>
    <th style="color:#818cf8;">Input Tokens</th>
    <th style="color:#34d399;">Output Tokens</th>
    <th style="color:#60a5fa;">Total Tokens</th>
    <th style="color:#f472b6;">Est. Cost (USD)</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)

# ── Totals row ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
            border-radius:12px;padding:0.9rem 1.2rem;display:flex;gap:2rem;flex-wrap:wrap;">
  <div>
    <div style="font-size:0.68rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:.1em;">
      Total Input</div>
    <div style="font-size:1.2rem;font-weight:700;color:#818cf8;">{total_input:,} tokens</div>
  </div>
  <div>
    <div style="font-size:0.68rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:.1em;">
      Total Output</div>
    <div style="font-size:1.2rem;font-weight:700;color:#34d399;">{total_output:,} tokens</div>
  </div>
  <div>
    <div style="font-size:0.68rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:.1em;">
      Grand Total</div>
    <div style="font-size:1.2rem;font-weight:700;color:#60a5fa;">{total_all:,} tokens</div>
  </div>
  <div>
    <div style="font-size:0.68rem;color:rgba(255,255,255,0.35);text-transform:uppercase;letter-spacing:.1em;">
      Total Cost</div>
    <div style="font-size:1.2rem;font-weight:700;color:#f472b6;">{format_cost(total_cost)}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Pricing Reference ─────────────────────────────────────────────────────────
with st.expander("📋 GPT-4o-mini Pricing Reference", expanded=False):
    st.markdown("""
    <table class="otable" style="width:100%;">
      <thead><tr><th>Token Type</th><th>Price per 1M Tokens</th><th>Price per 1K Tokens</th></tr></thead>
      <tbody>
        <tr>
          <td style="color:#818cf8;">Input (Prompt)</td>
          <td style="color:#34d399;">$0.150</td>
          <td style="color:#34d399;">$0.00015</td>
        </tr>
        <tr>
          <td style="color:#f472b6;">Output (Completion)</td>
          <td style="color:#f87171;">$0.600</td>
          <td style="color:#f87171;">$0.00060</td>
        </tr>
      </tbody>
    </table>
    <div style="font-size:0.78rem;color:rgba(255,255,255,0.3);margin-top:0.8rem;">
    Prices as of 2024. Check <a href="https://openai.com/pricing" style="color:#818cf8;">openai.com/pricing</a> for current rates.
    </div>
    """, unsafe_allow_html=True)

# ── Reset note ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-size:0.78rem;color:rgba(255,255,255,0.28);text-align:center;margin-top:1rem;">
  Token counts reset when you upload a new dataset or click Reset on the main page.
</div>
""", unsafe_allow_html=True)
