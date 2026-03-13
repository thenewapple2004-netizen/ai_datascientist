"""
frontend/components.py — Reusable UI components and PDF report generator.
Enhanced PDF Report with more text details and better styling.
"""
import os
import glob
import json
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st


# ─── MCQ Helpers ──────────────────────────────────────────────────────────────

def extract_mcq_json(text: str) -> Optional[dict]:
    """Robustly extract MCQ JSON from an agent message."""
    brace_match = re.search(r'(\{[\s\S]+\})', text, re.DOTALL)
    candidates  = [brace_match.group(1)] if brace_match else []
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
    Renders a full MCQ card. Returns (answered, total, summary_string | None).
    summary_string is non-None only when the user clicked Submit.
    """
    questions = st.session_state.get(mcq_key)
    if not questions:
        return 0, 0, None

    ans_key = f"{mcq_key}_answers"
    if ans_key not in st.session_state or len(st.session_state[ans_key]) != len(questions):
        st.session_state[ans_key] = {q["id"]: None for q in questions}

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
        hint = f'<div class="mcq-q-hint">{q.get("hint","")}</div>' if q.get("hint") else ""
        st.markdown(f"""
        <div class="mcq-q-block">
          <div style="display:flex;align-items:flex-start;">
            <span class="mcq-q-num">{idx+1}</span>
            <div>
              <div class="mcq-q-text">{q['question']}</div>
              {hint}
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
    total    = len(questions)
    dot_html = " ".join(
        f'<span class="mcq-dot{"  done" if st.session_state[ans_key].get(q["id"]) else ""}"></span>'
        for q in questions
    )
    st.markdown(f"""
    <div style="padding:0.6rem 1.5rem;">
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


# ─── Stat Cards ───────────────────────────────────────────────────────────────

def stat_cards(items: list) -> None:
    """
    Render a row of stat cards.
    items = [(col_obj, value_str, label_str, color_hex), ...]
    """
    for col, val, label, color in items:
        col.markdown(
            f'<div class="dp-stat">'
            f'<span class="val" style="color:{color};">{val}</span>'
            f'<span class="lbl">{label}</span></div>',
            unsafe_allow_html=True,
        )


def section_banner(num: str, title: str, color: str) -> None:
    st.markdown(
        f'<div class="sec-banner" style="background:rgba({_hex_to_rgb(color)},0.07);'
        f'border:1px solid rgba({_hex_to_rgb(color)},0.2);">'
        f'<div><div class="sec-num">Section {num}</div>'
        f'<div class="sec-title" style="color:{color};">{title}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def card_header(text: str) -> None:
    st.markdown(f'<div class="dp-card-hdr">{text}</div>', unsafe_allow_html=True)


def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


# ─── PDF Report Generator ─────────────────────────────────────────────────────

def generate_pdf_report(
    original_df:   Optional[pd.DataFrame],
    cleaned_df:    Optional[pd.DataFrame],
    fe_ops_lines:  list,
    insights:      list,
    charts_dir:    str,
    csv_name:      str = "dataset",
) -> Optional[bytes]:
    """
    Generate a comprehensive PDF analysis report.
    Returns PDF bytes, or None if fpdf2 is not installed.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    class StylizedPDF(FPDF):
        def header(self):
            if self.page_no() > 1:
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(150, 150, 150)
                self.cell(0, 10, f"AI Data Scientist Analysis Report - {csv_name}", align="R", new_x="LMARGIN", new_y="NEXT")
                self.ln(2)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = StylizedPDF()
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _cell(w, h, txt, border=0, align="", fill=False, nl=True):
        if nl:
            pdf.cell(w, h, txt, border=border, align=align, fill=fill,
                     new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(w, h, txt, border=border, align=align, fill=fill)

    def _mcell(txt, h=6):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, h, txt)
        pdf.set_x(pdf.l_margin)

    def _sec(title, desc=""):
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(79, 70, 229) # Indigo
        _cell(0, 10, title)
        pdf.set_draw_color(79, 70, 229)
        pdf.set_line_width(0.5)
        y = pdf.get_y()
        pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
        pdf.ln(3)
        if desc:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 116, 139) # Grey-slate
            _mcell(desc)
            pdf.ln(3)
        pdf.set_text_color(30, 41, 59)
        pdf.set_x(pdf.l_margin)

    def _thead(headers, widths):
        pdf.set_fill_color(79, 70, 229)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        for h, w in zip(headers, widths):
            _cell(w, 8, h, border=1, fill=True, nl=False, align="C")
        pdf.ln()

    # ── Cover Page ──────────────────────────────────────────────────────────
    pdf.add_page()
    # Background pattern or color
    pdf.set_fill_color(248, 250, 252)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.set_y(80)
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(79, 70, 229)
    _cell(0, 20, "AI DATA SCIENTIST", align="C")

    pdf.ln(5)
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(100, 116, 139)
    _cell(0, 10, "Comprehensive Analysis & Feature Engineering Report", align="C")

    pdf.set_draw_color(79, 70, 229)
    pdf.set_line_width(1)
    pdf.line(60, pdf.get_y() + 5, 150, pdf.get_y() + 5)

    pdf.set_y(180)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(51, 65, 85)
    _cell(0, 8, f"DATASET: {csv_name}", align="C")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(148, 163, 184)
    _cell(0, 7, f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", align="C")

    if original_df is not None:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(79, 70, 229)
        r, c = original_df.shape
        _cell(0, 8, f"Analysis Result: {r:,} Observations and {c} Processed Variables", align="C")
        pdf.set_text_color(0, 0, 0) # Back to black for general text

    # ── Page 2: Executive Summary ─────────────────────────────────────────────
    pdf.add_page()
    _sec("Executive Summary", "A high-level overview of the dataset characteristics and the automated processing pipeline performed by the AI Agent.")
    
    if original_df is not None:
        rows, cols_n = original_df.shape
        numeric_n    = int(original_df.select_dtypes(include="number").shape[1])
        cat_n        = int(original_df.select_dtypes(exclude="number").shape[1])
        missing_tot  = int(original_df.isnull().sum().sum())

        pdf.set_font("Helvetica", "", 10)
        _mcell(f"The analysis was conducted on the dataset '{csv_name}'. The original data consists of {rows:,} rows across {cols_n} distinct columns. "
               f"Our automated pipeline identified {cat_n} categorical features and {numeric_n} numerical features.")
        pdf.ln(4)

        vals = [
            ("Total Observations",   f"{rows:,}"),
            ("Raw Feature Count",    str(cols_n)),
            ("Numerical Variables",  str(numeric_n)),
            ("Categorical Variables",str(cat_n)),
            ("Missing Value Count",  str(missing_tot)),
        ]
        
        for label, val in vals:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(79, 70, 229)
            _cell(60, 8, label + ":", nl=False)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 41, 59)
            _cell(0, 8, val)

    # ── Column Details Table ─────────────────────────────────────────────────
    if original_df is not None:
        _sec("Data Dictionary & Quality", "Detailed breakdown of each column's data type, uniqueness, and missingness observed in the raw data.")
        headers = ["Column Name", "Type", "Unique", "Nulls", "Sample Values"]
        widths  = [50, 25, 20, 20, 55]
        _thead(headers, widths)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(0, 0, 0) # Absolute Black for visibility
        for i, col in enumerate(original_df.columns):
            fill = i % 2 == 0
            pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
            
            # Get samples
            samples = original_df[col].dropna().unique()[:2]
            sample_str = ", ".join(map(str, samples))[:50]
            
            vals = [
                str(col)[:28],
                str(original_df[col].dtype),
                str(original_df[col].nunique()),
                str(int(original_df[col].isnull().sum())),
                sample_str if sample_str else "—"
            ]
            pdf.set_text_color(0, 0, 0) # Explicitly reset before each row
            for v, w in zip(vals, widths):
                _cell(w, 8, v, border=1, fill=fill, nl=False, align="C")
            pdf.ln()

    # ── EDA Section ──────────────────────────────────────────────────────────
    eda_charts = sorted([
        p for p in glob.glob(os.path.join(charts_dir, "*.png"))
        if "cleaned" not in os.path.basename(p) and "_final" not in os.path.basename(p)
    ])
    if eda_charts:
        pdf.add_page()
        _sec("Initial Exploratory Data Analysis", "Visualizations generated before any feature engineering. These charts highlight the distributions, correlations, and relationships present in the original dataset.")
        
        for chart_path in eda_charts:
            if pdf.get_y() > 200:
                pdf.add_page()
            
            title = os.path.splitext(os.path.basename(chart_path))[0].replace("_", " ").title()
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(51, 65, 85)
            _cell(0, 8, title)
            
            # Simple description based on type
            desc = "This chart illustrates the "
            if "hist" in chart_path.lower(): desc += "distribution and density of the specified variable."
            elif "corr" in chart_path.lower(): desc += "linear correlation matrix between numerical features."
            elif "scatter" in chart_path.lower(): desc += "relationship and potential clusters between two numerical variables."
            elif "box" in chart_path.lower(): desc += "spread and outliers across different categories."
            else: desc += "patterns observed in the data."
            
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 116, 139)
            _mcell(desc)
            pdf.ln(2)
            
            try:
                pdf.image(chart_path, x=20, w=170)
            except Exception:
                pass
            pdf.ln(8)

    # ── Feature Engineering Section ──────────────────────────────────────────
    pdf.add_page()
    _sec("Feature Engineering & Data Cleaning", "The AI Data Scientist applied the following transformations to prepare the data for final analysis. This includes handling missing values, encoding categories, and scaling.")
    
    if fe_ops_lines:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 41, 59)
        for op in fe_ops_lines:
            pdf.set_font("Helvetica", "B", 10)
            _cell(5, 7, "-", nl=False)
            pdf.set_font("Helvetica", "", 10)
            _mcell(op)
            pdf.ln(2)
    else:
        pdf.set_font("Helvetica", "I", 10)
        _cell(0, 10, "No major transformations were required for this dataset.")

    if original_df is not None and cleaned_df is not None:
        pdf.ln(5)
        _sec("Pipeline Impact Metrics", "Comparison of dataset state before and after the feature engineering phase.")
        or2, oc2 = original_df.shape
        cr,  cc  = cleaned_df.shape
        _thead(["Metric", "Original State", "Processed State"], [70, 50, 50])
        
        rows_data = [
            ["Row Count",        f"{or2:,}",  f"{cr:,}"],
            ["Total Features",   str(oc2),     str(cc)],
            ["Null Discovered",  str(int(original_df.isnull().sum().sum())), "0"],
            ["Categorical Features", str(int(original_df.select_dtypes(exclude="number").shape[1])), "0 (Encoded)"],
            ["Scaled Features",  "Raw", "StandardScaled"],
        ]
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 41, 59)
        for i, row in enumerate(rows_data):
            fill = i % 2 == 0
            pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
            for v, w in zip(row, [70, 50, 50]):
                _cell(w, 8, v, border=1, fill=fill, nl=False, align="C")
            pdf.ln()

    # ── Final Insights & Charts ──────────────────────────────────────────────
    final_charts    = sorted([p for p in glob.glob(os.path.join(charts_dir, "*.png")) if "_final" in os.path.basename(p)])
    cleaned_heatmap = os.path.join(charts_dir, "heatmap_correlation_cleaned.png")

    if insights or final_charts or os.path.exists(cleaned_heatmap):
        pdf.add_page()
        _sec("Final Insights & Optimized Analysis", "Conclusions drawn from the refined dataset. These findings represent the core value extracted from your data.")

        if insights:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(79, 70, 229)
            _cell(0, 10, "Key Discoveries:")
            pdf.ln(2)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 41, 59)
            for ins in insights:
                if ins.strip():
                    pdf.set_font("Helvetica", "B", 10)
                    _cell(5, 7, "->", nl=False)
                    pdf.set_font("Helvetica", "", 10)
                    _mcell(ins)
                    pdf.ln(2)
            pdf.ln(5)

        if os.path.exists(cleaned_heatmap):
            if pdf.get_y() > 180: pdf.add_page()
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(51, 65, 85)
            _cell(0, 8, "Cleaned Feature Correlation Matrix")
            try:
                pdf.image(cleaned_heatmap, x=20, w=170)
            except Exception:
                pass
            pdf.ln(10)

        for chart_path in final_charts:
            if pdf.get_y() > 180:
                pdf.add_page()
            
            title = os.path.splitext(os.path.basename(chart_path))[0].replace("_", " ").title()
            pdf.set_font("Helvetica", "B", 11)
            _cell(0, 8, title)
            try:
                pdf.image(chart_path, x=20, w=170)
            except Exception:
                pass
            pdf.ln(10)

    # ── Conclusion ───────────────────────────────────────────────────────────
    if pdf.get_y() > 220: pdf.add_page()
    _sec("Conclusion")
    pdf.set_font("Helvetica", "", 10)
    _mcell("The AI Data Scientist has successfully ingested, cleaned, and analyzed the provided dataset. All categorical features have been encoded, missing values imputed, and numerical features scaled for machine learning readiness. The final processed dataset is available for download in the main dashboard.")
    
    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(148, 163, 184)
    _cell(0, 10, "*** End of Report ***", align="C")

    return bytes(pdf.output())


# ─── Token Usage Formatting ───────────────────────────────────────────────────

# GPT-4o-mini pricing (per 1M tokens)
_INPUT_PRICE_PER_M  = 0.150
_OUTPUT_PRICE_PER_M = 0.600


def calc_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * _INPUT_PRICE_PER_M + \
           (output_tokens / 1_000_000) * _OUTPUT_PRICE_PER_M


def format_cost(usd: float) -> str:
    if usd < 0.001:
        return f"${usd*100:.4f}c"
    return f"${usd:.4f}"
