"""
Restructure app.py display sections into st.tabs().
Keeps all execution/spinner blocks as-is.
Replaces the display sections (from Phase 2 DISPLAY onward) with one tabs block.
"""

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line that starts "# PHASE 2 DISPLAY"
start_display = None
for i, l in enumerate(lines):
    if '# PHASE 2 DISPLAY' in l and 'Show initial EDA' in l:
        start_display = i
        break

if start_display is None:
    print("ERROR: could not find PHASE 2 DISPLAY marker")
    exit(1)

print(f"Cutting from line {start_display+1} (0-indexed {start_display})")

# Keep everything up to (but not including) the Phase 2 DISPLAY block
kept = lines[:start_display]

# The new tab-based display block
TABS_BLOCK = '''\
# ═══════════════════════════════════════════════════════════════════════════════
# TABBED DISPLAY — All pipeline results shown in separate sections
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("eda_done"):
    import pandas as pd

    _tab_labels = ["📊 Initial EDA"]
    if st.session_state.get("fe_done"):
        _tab_labels.append("🔧 Feature Engineering & Extraction")
    if st.session_state.done:
        _tab_labels.append("📈 Final Analysis")

    _tabs = st.tabs(_tab_labels)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — Initial EDA Charts
    # ─────────────────────────────────────────────────────────────────────────
    with _tabs[0]:
        eda_res = st.session_state.eda_result
        if not eda_res or not eda_res.get("success"):
            st.error(f"EDA Error: {eda_res.get('error','') if eda_res else 'No result'}")
        else:
            with st.expander("📂 Original Dataset", expanded=False):
                try:
                    st.dataframe(pd.read_csv(st.session_state.csv_path), use_container_width=True)
                except Exception:
                    pass

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
                                f\'<div style="text-align:center;font-size:0.75rem;color:rgba(255,255,255,0.38);">\'
                                f\'{os.path.splitext(os.path.basename(path))[0]}</div>\',
                                unsafe_allow_html=True
                            )
            else:
                st.info("No EDA charts generated yet.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — Feature Engineering & Extraction
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.get("fe_done"):
        with _tabs[1]:
            fe_report = st.session_state.get("fe_report", "")

            # Parse FE report
            fe_ops_lines = []
            if fe_report:
                in_table = False
                for raw in fe_report.split("\\n"):
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

            # ── A: Operations summary ──────────────────────────────────────────
            st.markdown(\'<div class="dp-card-hdr">⚙️ Feature Engineering Operations</div>\', unsafe_allow_html=True)

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
                        f\'<div style="display:flex;align-items:flex-start;gap:0.5rem;\' +
                        f\'padding:0.5rem 0.8rem;margin:0.25rem 0;\' +
                        f\'background:{bg};border-radius:8px;border-left:3px solid {color};">\' +
                        f\'<span style="font-size:0.9rem;">{icon}</span>\' +
                        f\'<span style="font-size:0.82rem;color:rgba(255,255,255,0.8);">{text}</span></div>\'
                    )

                ops_html = "".join(_badge(l) for l in fe_ops_lines if l)
                st.markdown(f\'<div style="margin-bottom:1rem;">{ops_html}</div>\', unsafe_allow_html=True)
            else:
                st.markdown(\'<div class="insight">No additional cleaning operations were needed.</div>\', unsafe_allow_html=True)

            # ── B: Feature Extraction table ───────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(\'<div class="dp-card-hdr">⚡ Feature Extraction — Numeric → Binary Indicators</div>\', unsafe_allow_html=True)

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
                            f\'<div style="display:flex;align-items:center;gap:0.4rem;">\' +
                            f\'<div style="flex:1;height:6px;background:rgba(255,255,255,0.08);border-radius:3px;">\' +
                            f\'<div style="width:{pct}%;height:6px;background:#34d399;border-radius:3px;"></div></div>\' +
                            f\'<span style="font-size:0.75rem;color:#34d399;">{pct}%</span></div>\'
                        )
                    rows_html += (
                        f"<tr>"
                        f\'<td><code style="color:#818cf8;">{orig}</code></td>\' +
                        f\'<td><code style="color:#34d399;">{binc}</code></td>\' +
                        f\'<td style="color:rgba(255,255,255,0.5);">{o_min}</td>\' +
                        f\'<td style="color:rgba(255,255,255,0.5);">{o_max}</td>\' +
                        f\'<td style="color:rgba(255,255,255,0.5);">{o_mean}</td>\' +
                        f\'<td style="color:#f87171;">{zeros}</td>\' +
                        f\'<td style="color:#34d399;">{ones}</td>\' +
                        f"<td>{pct_bar}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f\'\'\'<table class="otable" style="width:100%;">
                      <thead><tr>
                        <th>Original Column</th><th>Binary Column (_bin)</th>
                        <th>Min</th><th>Max</th><th>Mean</th>
                        <th style="color:#f87171;">Zeros (0)</th>
                        <th style="color:#34d399;">Ones (1)</th>
                        <th>% Positive</th>
                      </tr></thead>
                      <tbody>{rows_html}</tbody>
                    </table>\'\'\',
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)
            else:
                st.info("No binary (\\*_bin) columns found. The agent may not have executed binarization.")

            # ── C: Cleaned dataset + post-FE heatmap ──────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(\'<div class="dp-card-hdr">📂 Cleaned & Extracted Dataset</div>\', unsafe_allow_html=True)

            if cleaned_df is not None:
                _nr2, _nc2 = cleaned_df.shape
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows",            f"{_nr2:,}")
                m2.metric("Total Columns",   f"{_nc2}")
                m3.metric("Binary Features", f"{len(bin_cols)}")
                with st.expander("View full cleaned dataset", expanded=False):
                    st.dataframe(cleaned_df, use_container_width=True)

            fe_heatmap = os.path.join(os.path.dirname(__file__), "charts", "heatmap_correlation_cleaned.png")
            if os.path.exists(fe_heatmap):
                st.markdown("#### 🌡️ Post-Feature Engineering Correlation Heatmap")
                st.image(fe_heatmap, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — Final Analysis Charts
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.done:
        _final_tab_idx = (2 if st.session_state.get("fe_done") else 1)
        with _tabs[_final_tab_idx]:
            final_res = st.session_state.eda_result
            if not final_res or not final_res.get("success"):
                st.error(f"Error: {final_res.get('error', 'Unknown') if final_res else 'No result'}")
            else:
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
                                    f\'<div style="text-align:center;font-size:0.75rem;color:rgba(255,255,255,0.38);">\'
                                    f\'{os.path.splitext(os.path.basename(path))[0]}</div>\',
                                    unsafe_allow_html=True
                                )
                else:
                    # Fallback: show all charts
                    for i in range(0, len(all_charts), 2):
                        row = all_charts[i:i+2]
                        cols = st.columns(len(row))
                        for col, path in zip(cols, row):
                            with col:
                                st.image(path, use_container_width=True)

                last_content = st.session_state.messages[-1].content if st.session_state.messages else ""
                if last_content and "FINAL_COMPLETE:" in last_content:
                    summary_text = last_content.split("FINAL_COMPLETE:")[-1].strip()
                    lines_s = [l.strip("•- ").strip() for l in summary_text.split("\\n") if l.strip()]
                    bullets = "".join(f"<li>{l}</li>" for l in lines_s[:5] if l)
                    st.markdown(
                        f\'<div class="insight"><strong>🤖 Key Insights:</strong>\' +
                        f\'<ul style="margin:0.4rem 0 0 1.2rem;">{bullets}</ul></div>\',
                        unsafe_allow_html=True
                    )
'''

new_content = "".join(kept) + TABS_BLOCK + "\n"

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"Done. New file has {new_content.count(chr(10))} lines.")
