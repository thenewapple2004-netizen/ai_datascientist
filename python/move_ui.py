import os

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# I will find the lines corresponding to the Feature Extraction table, the Dataset Preview, and the Heatmap,
# and move them into Tab 3 (_tabs[2]).

block_to_move = """
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
                
                csv_data = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned Dataset (CSV)",
                    data=csv_data,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv"
                )

            import glob, os
            fe_heatmap = os.path.join(os.path.dirname(__file__), "charts", "heatmap_correlation_cleaned.png")
            if os.path.exists(fe_heatmap):
                st.markdown("#### 🌡️ Post-Feature Engineering Correlation Heatmap")
                st.image(fe_heatmap, use_container_width=True)
"""

if block_to_move in content:
    content = content.replace(block_to_move, "")
else:
    print("Block not found!")

target_insertion = """        elif st.session_state.get("done"):
            final_res = st.session_state.eda_result
            if not final_res or not final_res.get("success"):
                st.error(f"Error: {final_res.get('error', 'Unknown') if final_res else 'No result'}")
            else:"""

if target_insertion in content:
    content = content.replace(target_insertion, target_insertion + "\n" + block_to_move + "\n")
else:
    print("Target insertion not found!")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("done")
