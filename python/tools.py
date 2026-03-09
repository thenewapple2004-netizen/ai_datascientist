"""
tools.py — Custom EDA Toolset
Pattern: df = pd.read_csv(path) → PyTorch tensors for math → Seaborn/Matplotlib for charts
Each function is a LangChain Tool available to the Agent.
"""

import json
import os
import csv as csv_module

import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive — safe for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from langchain.tools import tool

# ─────────────────────────────────────────────────────────────────────────────
# Module-level state  (one dataset at a time)
# ─────────────────────────────────────────────────────────────────────────────
_df: pd.DataFrame | None = None          # full dataframe (all dtypes)
_tensor: torch.Tensor | None = None      # numeric columns → float32 tensor
_columns: list[str] = []                 # numeric column names
_charts_dir: str = os.path.join(os.path.dirname(__file__), "charts")

# Seaborn dark theme — matches Streamlit dark background
sns.set_theme(style="darkgrid", palette="muted")
DARK_BG   = "#12121f"
PLOT_BG   = "#1e1e2e"
TEXT_CLR  = "white"
PALETTE   = ["#6366f1","#34d399","#60a5fa","#f59e0b",
             "#a78bfa","#f472b6","#fb923c","#4ade80"]

def _apply_dark(fig, axes):
    """Apply dark theme to any matplotlib figure."""
    fig.patch.set_facecolor(DARK_BG)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=TEXT_CLR)
        ax.title.set_color(TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d2d3f")
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(PLOT_BG)
            for text in legend.get_texts():
                text.set_color(TEXT_CLR)


# ─────────────────────────────────────────────────────────────────────────────
# Public loader — called from app.py before running the agent
# ─────────────────────────────────────────────────────────────────────────────
def load_csv_into_tensor(csv_path: str) -> str:
    """
    STEP 1 — df = pd.read_csv(csv_path)
    Loads the CSV into a pandas DataFrame (handles mixed dtypes, text cols, etc.)
    Then extracts numeric columns into a PyTorch float32 tensor for math.
    Returns a status message.
    """
    global _df, _tensor, _columns

    # ── Load with pandas (same as: df = pd.read_csv(path)) ──────────────────
    _df = pd.read_csv(csv_path)

    print(f"\n[tools] df.shape = {_df.shape}")
    print(f"[tools] df.info():\n{_df.dtypes}\n")
    print(f"[tools] df.head():\n{_df.head()}\n")

    # ── Extract numeric columns → PyTorch tensor ─────────────────────────────
    numeric_df = _df.select_dtypes(include="number")
    _columns   = list(numeric_df.columns)

    if not _columns:
        _tensor = None
        return f"Loaded {len(_df)} rows, {len(_df.columns)} columns. No numeric columns found."

    # Convert to float tensor — NaN values preserved as float('nan')
    _tensor = torch.tensor(numeric_df.values.astype("float32"), dtype=torch.float32)

    return (
        f"Loaded {_df.shape[0]} rows × {_df.shape[1]} columns. "
        f"Numeric cols ({len(_columns)}): {', '.join(_columns)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LangChain Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_dataframe_info(dummy: str = "") -> str:
    """
    Returns basic dataset info: shape, all column names & dtypes, 
    and the first 5 rows (like df.info() + df.head()).
    Always call this first to understand the dataset.
    """
    if _df is None:
        return "No dataset loaded yet."

    info = {
        "shape": {"rows": int(_df.shape[0]), "columns": int(_df.shape[1])},
        "all_columns": list(_df.columns),
        "dtypes": {col: str(dtype) for col, dtype in _df.dtypes.items()},
        "numeric_columns": _columns,
        "head": _df.head().to_dict(orient="records"),
    }
    return json.dumps(info, indent=2, default=str)


@tool
def generate_eda_summary(dummy: str = "") -> str:
    """
    Computes descriptive statistics for ALL numeric columns using PyTorch:
    count, min, max, mean, median, std, and missing value count & %.
    Returns a JSON report.
    """
    if _tensor is None or _df is None:
        return "No dataset loaded yet."

    n_rows = _tensor.shape[0]
    report = {}

    for i, col in enumerate(_columns):
        col_data = _tensor[:, i]
        nan_mask = torch.isnan(col_data)
        valid    = col_data[~nan_mask]

        entry = {
            "missing_count": int(nan_mask.sum().item()),
            "missing_pct":   round(nan_mask.sum().item() / n_rows * 100, 2),
        }
        if valid.numel() > 0:
            entry.update({
                "count":  valid.numel(),
                "min":    round(valid.min().item(), 4),
                "max":    round(valid.max().item(), 4),
                "mean":   round(valid.mean().item(), 4),
                "median": round(valid.median().item(), 4),
                "std":    round(valid.std().item(), 4),
            })
        report[col] = entry

    return json.dumps(report, indent=2)


@tool
def analyze_missing_values(dummy: str = "") -> str:
    """
    Analyzes missing (NaN) values in ALL columns (numeric + text).
    Returns count and percentage of missing per column.
    """
    if _df is None:
        return "No dataset loaded yet."

    result = {}
    n = len(_df)
    for col in _df.columns:
        cnt = int(_df[col].isna().sum())
        result[col] = {
            "missing_count": cnt,
            "missing_pct":   round(cnt / n * 100, 2),
        }
    return json.dumps(result, indent=2)


@tool
def impute_missing_data(method: str = "mean") -> str:
    """
    Fills missing values in numeric columns using 'mean', 'median', or 'zero'.
    Updates the tensor in-place. Returns a confirmation message.
    """
    global _tensor
    if _tensor is None:
        return "No dataset loaded yet."

    method = method.strip().lower()
    if method not in ("mean", "median", "zero"):
        return "Invalid method. Choose 'mean', 'median', or 'zero'."

    imputed = _tensor.clone()
    for i in range(_tensor.shape[1]):
        col  = _tensor[:, i]
        mask = torch.isnan(col)
        if mask.any():
            fill = (col[~mask].mean() if method == "mean"
                    else col[~mask].median() if method == "median"
                    else torch.tensor(0.0))
            imputed[mask, i] = fill

    _tensor = imputed
    return f"Missing values filled using method='{method}'."


@tool
def generate_missing_value_chart(dummy: str = "") -> str:
    """
    Generates a horizontal bar chart of missing value % for all columns.
    Green < 5%, Amber 5-20%, Red > 20%.
    Saves to charts/missing_values.png and returns the file path.
    """
    if _df is None:
        return "No dataset loaded yet."

    os.makedirs(_charts_dir, exist_ok=True)
    n = len(_df)
    cols  = list(_df.columns)
    pcts  = [round(_df[c].isna().sum() / n * 100, 2) for c in cols]
    clrs  = ["#f87171" if p > 20 else "#fbbf24" if p > 0 else "#34d399" for p in pcts]

    fig, ax = plt.subplots(figsize=(9, max(3, len(cols) * 0.45)))
    bars = ax.barh(cols, pcts, color=clrs, edgecolor="#1e1e2e")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Values per Column", fontsize=13, fontweight="bold")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8.5, color=TEXT_CLR)
    _apply_dark(fig, ax)
    plt.tight_layout()
    out = os.path.join(_charts_dir, "missing_values.png")
    fig.savefig(out, dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    return f"Missing value chart saved to: {out}"


@tool
def generate_distribution_charts(dummy: str = "") -> str:
    """
    Generates a seaborn histplot (with KDE) for every numeric column —
    exactly like: sns.histplot(df['col'], kde=True).
    Mean and median lines overlaid. Saves one PNG per column.
    Returns the list of saved paths.
    """
    if _df is None or not _columns:
        return "No dataset loaded yet."

    os.makedirs(_charts_dir, exist_ok=True)
    saved = []

    for i, col in enumerate(_columns):
        col_data = _df[col].dropna()
        if len(col_data) == 0:
            continue

        # --- seaborn histplot with KDE (like untitled6.py) ---
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(col_data, kde=True, ax=ax,
                     color=PALETTE[i % len(PALETTE)], alpha=0.75)

        # Overlay mean and median using PyTorch values
        t = torch.tensor(col_data.values, dtype=torch.float32)
        mean_v = t.mean().item()
        med_v  = t.median().item()
        ax.axvline(mean_v, color="#f59e0b", linewidth=1.8, linestyle="--",
                   label=f"Mean: {mean_v:.2f}")
        ax.axvline(med_v,  color="#34d399", linewidth=1.8, linestyle=":",
                   label=f"Median: {med_v:.2f}")
        ax.legend(fontsize=8)
        ax.set_title(f"Distribution: {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        _apply_dark(fig, ax)
        plt.tight_layout()

        out = os.path.join(_charts_dir, f"hist_{col}.png")
        fig.savefig(out, dpi=120, facecolor=DARK_BG)
        plt.close(fig)
        saved.append(out)

    return f"Distribution histograms saved for: {', '.join(_columns)}"


@tool
def generate_boxplot_charts(dummy: str = "") -> str:
    """
    Generates a seaborn boxplot for ALL numeric columns showing spread,
    IQR, median and outlier dots — like: sns.boxplot(data=df).
    Saves to charts/boxplot_all.png and returns the file path.
    """
    if _df is None or not _columns:
        return "No dataset loaded yet."

    os.makedirs(_charts_dir, exist_ok=True)
    numeric_df = _df[_columns]

    fig, ax = plt.subplots(figsize=(max(8, len(_columns) * 1.5), 5))
    sns.boxplot(data=numeric_df, ax=ax,
                palette=PALETTE[:len(_columns)],
                flierprops=dict(marker="o", markerfacecolor="#f87171",
                                markersize=4, alpha=0.6))
    ax.set_title("Box Plots — Spread & Outlier Detection", fontsize=13, fontweight="bold")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Value")
    plt.xticks(rotation=30, ha="right")
    _apply_dark(fig, ax)
    plt.tight_layout()

    out = os.path.join(_charts_dir, "boxplot_all.png")
    fig.savefig(out, dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    return f"Box plot saved to: {out}"


@tool
def detect_outliers_report(dummy: str = "") -> str:
    """
    Detects outliers in every numeric column using IQR method via PyTorch.
    Returns Q1, Q3, IQR, fences, outlier count and values per column.
    """
    if _tensor is None:
        return "No dataset loaded yet."

    report = {}
    for i, col in enumerate(_columns):
        valid = _tensor[:, i]
        valid = valid[~torch.isnan(valid)]
        if valid.numel() < 4:
            report[col] = {"note": "Not enough data"}
            continue

        q1  = valid.quantile(0.25).item()
        q3  = valid.quantile(0.75).item()
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr

        mask     = (valid < lo) | (valid > hi)
        out_vals = valid[mask].tolist()

        report[col] = {
            "q1": round(q1, 4), "q3": round(q3, 4), "iqr": round(iqr, 4),
            "lower_fence": round(lo, 4), "upper_fence": round(hi, 4),
            "outlier_count":  int(mask.sum().item()),
            "outlier_values": [round(v, 4) for v in out_vals[:10]],
        }

    return json.dumps(report, indent=2)


@tool
def generate_eda_charts(plot_type: str) -> str:
    """
    Generates a chart and saves it.
    Supported plot_type values:
      - 'bar'      : seaborn barplot of column means
      - 'scatter'  : seaborn scatterplot of first two numeric columns
      - 'heatmap'  : seaborn heatmap of Pearson correlation matrix
                     (like: sns.heatmap(df.corr(), annot=True))
    Returns the saved file path.
    """
    if _df is None or not _columns:
        return "No dataset loaded yet."

    os.makedirs(_charts_dir, exist_ok=True)
    plot_type = plot_type.strip().lower()

    if plot_type not in ("bar", "scatter", "heatmap"):
        return "Invalid plot_type. Choose 'bar', 'scatter', or 'heatmap'."

    numeric_df = _df[_columns]
    fig, ax = plt.subplots(figsize=(9, 5))

    if plot_type == "bar":
        # Use PyTorch means; seaborn barplot for display
        means = {c: round(_tensor[:, i][~torch.isnan(_tensor[:, i])].mean().item(), 4)
                 for i, c in enumerate(_columns)}
        sns.barplot(x=list(means.keys()), y=list(means.values()),
                    ax=ax, palette=PALETTE)
        ax.set_title("Column Means — Bar Chart", fontsize=13, fontweight="bold")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Mean Value")
        plt.xticks(rotation=30, ha="right")

    elif plot_type == "scatter":
        if len(_columns) < 2:
            plt.close(fig)
            return "Need at least 2 numeric columns for scatter plot."
        # exactly like: sns.scatterplot(data=df, x='col1', y='col2')
        sns.scatterplot(data=_df, x=_columns[0], y=_columns[1],
                        ax=ax, color=PALETTE[0], alpha=0.7)
        ax.set_title(f"Scatter: {_columns[0]} vs {_columns[1]}",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.2)

    elif plot_type == "heatmap":
        # exactly like: sns.heatmap(df.corr(), annot=True)
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax,
                    cmap="coolwarm", vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="#1e1e2e",
                    annot_kws={"size": 8})
        ax.set_title("Correlation Heatmap (Pearson)", fontsize=13, fontweight="bold")

    _apply_dark(fig, ax)
    plt.tight_layout()
    out = os.path.join(_charts_dir, f"{plot_type}_chart.png")
    fig.savefig(out, dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    return f"Chart saved to: {out}"


@tool
def calculate_descriptive_stats(column_name: str) -> str:
    """
    Calculates descriptive statistics for a single named column using PyTorch.
    Input: the exact column name as a string.
    """
    if _tensor is None:
        return "No dataset loaded yet."
    if column_name not in _columns:
        return f"Column '{column_name}' not found. Numeric columns: {_columns}"

    i    = _columns.index(column_name)
    data = _tensor[:, i]
    data = data[~torch.isnan(data)]

    if data.numel() == 0:
        return f"Column '{column_name}' has no valid numeric values."

    stats = {
        "column": column_name,
        "count":  data.numel(),
        "min":    round(data.min().item(), 4),
        "max":    round(data.max().item(), 4),
        "mean":   round(data.mean().item(), 4),
        "median": round(data.median().item(), 4),
        "std":    round(data.std().item(), 4),
    }
    return json.dumps(stats, indent=2)




@tool
def generate_countplot(column_name: str) -> str:
    """
    Generates a seaborn countplot showing the frequency of each category
    in a column — like: sns.countplot(data=df, x='column').
    The LLM should call this for categorical columns (text, low-cardinality).
    Input: the column name as a string.
    Saves chart to charts/count_{column_name}.png and returns the path.
    """
    if _df is None:
        return "No dataset loaded yet."
    if column_name not in _df.columns:
        return f"Column '{column_name}' not found. Available: {list(_df.columns)}"

    os.makedirs(_charts_dir, exist_ok=True)

    # Count unique values; skip if too many (> 30) — not useful as countplot
    n_unique = _df[column_name].nunique()
    if n_unique > 30:
        return f"Column '{column_name}' has {n_unique} unique values — too many for a countplot."

    fig, ax = plt.subplots(figsize=(max(7, n_unique * 0.7), 4))
    order = _df[column_name].value_counts().index.tolist()
    sns.countplot(data=_df, x=column_name, ax=ax, order=order, palette=PALETTE)
    ax.set_title(f"Count Distribution: {column_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel(column_name)
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    _apply_dark(fig, ax)
    plt.tight_layout()
    out = os.path.join(_charts_dir, f"count_{column_name}.png")
    fig.savefig(out, dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    return f"Countplot saved to: {out}"


@tool
def generate_barplot_comparison(params_json: str) -> str:
    """
    Generates a seaborn barplot comparing a numeric column's mean grouped by a categorical column.
    Like: sns.barplot(data=df, x='cat_col', y='num_col')
    The LLM should use this to discover patterns like 'average salary by department'.
    Input: JSON string with keys 'cat_col' (categorical) and 'num_col' (numeric).
    Example: {"cat_col": "Sex", "num_col": "Fare"}
    Saves to charts/bar_{cat_col}_vs_{num_col}.png and returns the path.
    """
    if _df is None:
        return "No dataset loaded yet."
    try:
        params   = json.loads(params_json)
        cat_col  = params["cat_col"]
        num_col  = params["num_col"]
    except Exception:
        return "Invalid input. Provide JSON like: {\"cat_col\": \"Sex\", \"num_col\": \"Fare\"}"

    if cat_col not in _df.columns:
        return f"Column '{cat_col}' not found."
    if num_col not in _df.columns:
        return f"Column '{num_col}' not found."

    os.makedirs(_charts_dir, exist_ok=True)

    n_unique = _df[cat_col].nunique()
    order = _df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(max(7, n_unique * 0.9), 5))
    sns.barplot(data=_df, x=cat_col, y=num_col, ax=ax,
                order=order, palette=PALETTE, errorbar="sd")
    ax.set_title(f"Mean {num_col} by {cat_col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(cat_col)
    ax.set_ylabel(f"Mean {num_col}")
    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    _apply_dark(fig, ax)
    plt.tight_layout()
    fname = f"bar_{cat_col}_vs_{num_col}.png".replace(" ", "_")
    out   = os.path.join(_charts_dir, fname)
    fig.savefig(out, dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    return f"Barplot comparison saved to: {out}"


@tool
def generate_scatter_comparison(params_json: str) -> str:
    """
    Generates a seaborn scatterplot between any two columns chosen by the LLM.
    Like: sns.scatterplot(data=df, x='carat', y='price')
    The LLM should use this to compare any two numeric columns that may be correlated.
    Optionally color points by a third categorical column ('hue_col').
    Input: JSON string with keys 'x_col', 'y_col', and optionally 'hue_col'.
    Example: {"x_col": "Age", "y_col": "Fare", "hue_col": "Survived"}
    Saves to charts/scatter_{x_col}_vs_{y_col}.png and returns the path.
    """
    if _df is None:
        return "No dataset loaded yet."
    try:
        params  = json.loads(params_json)
        x_col   = params["x_col"]
        y_col   = params["y_col"]
        hue_col = params.get("hue_col")
    except Exception:
        return "Invalid input. Provide JSON like: {\"x_col\": \"Age\", \"y_col\": \"Fare\"}"

    for c in [x_col, y_col, hue_col]:
        if c and c not in _df.columns:
            return f"Column '{c}' not found. Available: {list(_df.columns)}"

    os.makedirs(_charts_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter_kw = dict(data=_df, x=x_col, y=y_col, ax=ax, alpha=0.65, s=30)
    if hue_col:
        scatter_kw["hue"] = hue_col
        scatter_kw["palette"] = PALETTE
    else:
        scatter_kw["color"] = PALETTE[0]

    sns.scatterplot(**scatter_kw)
    title = f"Scatter: {x_col} vs {y_col}"
    if hue_col:
        title += f" (colored by {hue_col})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    _apply_dark(fig, ax)
    plt.tight_layout()
    fname = f"scatter_{x_col}_vs_{y_col}.png".replace(" ", "_")
    out   = os.path.join(_charts_dir, fname)
    fig.savefig(out, dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    return f"Scatter comparison saved to: {out}"


@tool
def generate_boxplot_by_category(params_json: str) -> str:
    """
    Generates a seaborn boxplot of a numeric column grouped by a categorical column —
    exactly like: sns.boxplot(data=df, x='Pclass', y='Age')
    The LLM should use this to compare distributions across groups (e.g., survival by class).
    Useful for finding whether categories have significantly different value ranges.
    Input: JSON string with keys 'cat_col' and 'num_col'.
    Example: {"cat_col": "Pclass", "num_col": "Fare"}
    Saves to charts/boxcat_{cat_col}_vs_{num_col}.png and returns the path.
    """
    if _df is None:
        return "No dataset loaded yet."
    try:
        params  = json.loads(params_json)
        cat_col = params["cat_col"]
        num_col = params["num_col"]
    except Exception:
        return "Invalid input. Provide JSON like: {\"cat_col\": \"Pclass\", \"num_col\": \"Fare\"}"

    if cat_col not in _df.columns:
        return f"Column '{cat_col}' not found."
    if num_col not in _df.columns:
        return f"Column '{num_col}' not found."

    os.makedirs(_charts_dir, exist_ok=True)

    n_unique = _df[cat_col].nunique()
    order    = sorted(_df[cat_col].dropna().unique().tolist(), key=str)

    fig, ax = plt.subplots(figsize=(max(7, n_unique * 1.1), 5))
    sns.boxplot(data=_df, x=cat_col, y=num_col, ax=ax,
                order=order, palette=PALETTE,
                flierprops=dict(marker="o", markerfacecolor="#f87171",
                                markersize=4, alpha=0.6))
    ax.set_title(f"{num_col} Distribution by {cat_col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(cat_col)
    ax.set_ylabel(num_col)
    ax.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=30, ha="right")
    _apply_dark(fig, ax)
    plt.tight_layout()
    fname = f"boxcat_{cat_col}_vs_{num_col}.png".replace(" ", "_")
    out   = os.path.join(_charts_dir, fname)
    fig.savefig(out, dpi=120, facecolor=DARK_BG)
    plt.close(fig)
    return f"Box-by-category chart saved to: {out}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — imported by agent.py
# ─────────────────────────────────────────────────────────────────────────────
EDA_TOOLS = [
    get_dataframe_info,
    generate_eda_summary,
    analyze_missing_values,
    impute_missing_data,
    generate_missing_value_chart,
    generate_distribution_charts,
    generate_boxplot_charts,
    detect_outliers_report,
    generate_eda_charts,
    calculate_descriptive_stats,
    # ── LLM-driven smart comparison tools ──────────────────────────────────
    generate_countplot,
    generate_barplot_comparison,
    generate_scatter_comparison,
    generate_boxplot_by_category,
]


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test — python tools.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    csv = (
        "age,salary,score,dept\n"
        "25,50000,88.5,eng\n"
        "30,,92.0,mkt\n"
        "22,45000,,eng\n"
        "35,60000,75.0,hr\n"
        "28,52000,81.0,eng\n"
        "55,95000,91.0,hr\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, encoding="utf-8") as f:
        f.write(csv)
        p = f.name

    print(load_csv_into_tensor(p))
    print(get_dataframe_info.invoke(""))
    print(generate_eda_summary.invoke(""))
    print(analyze_missing_values.invoke(""))
    print(generate_missing_value_chart.invoke(""))
    print(generate_distribution_charts.invoke(""))
    print(generate_boxplot_charts.invoke(""))
    print(detect_outliers_report.invoke(""))
    print(generate_eda_charts.invoke("heatmap"))
    print(generate_eda_charts.invoke("bar"))
    print(generate_eda_charts.invoke("scatter"))
    print(generate_countplot.invoke("dept"))
    print(generate_barplot_comparison.invoke('{"cat_col":"dept","num_col":"salary"}'))
    print(generate_scatter_comparison.invoke('{"x_col":"age","y_col":"salary"}'))
    print(generate_boxplot_by_category.invoke('{"cat_col":"dept","num_col":"score"}'))
    os.unlink(p)

