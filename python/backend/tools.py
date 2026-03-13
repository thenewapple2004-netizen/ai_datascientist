"""
backend/tools.py — Data processing & visualization tools for the AI agent.
All chart output goes to python/charts/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Global DataFrame — shared with agent.py via tools._df
_df = None

_BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(_BASE, "charts")


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> str:
    global _df
    _df = pd.read_csv(csv_path)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return f"Loaded {_df.shape[0]} rows and {_df.shape[1]} columns."


# ─── Info ─────────────────────────────────────────────────────────────────────

def get_dataframe_info(df: pd.DataFrame) -> str:
    if df is None:
        return "Error: No dataframe loaded."
    buf = [f"Data shape: {df.shape} ---> (rows, columns)", "Column Details:"]
    top_values = []
    for col in df.columns:
        vals = df[col].dropna().unique()
        sample = ", ".join(map(str, vals[:5]))
        top_values.append(sample + (" ..." if len(vals) > 5 else ""))

    summary_df = pd.DataFrame({
        "Data Type":     df.dtypes,
        "Missing Count": df.isnull().sum(),
        "Missing %":     (df.isnull().mean() * 100).round(1),
        "Unique Values": df.nunique(),
        "Top Distinct":  top_values,
    })
    buf.append(summary_df.to_string())
    return "\n".join(buf)


# ─── Feature Engineering ──────────────────────────────────────────────────────

from sklearn.preprocessing import LabelEncoder, StandardScaler


def drop_useless_columns(df: pd.DataFrame, columns: list) -> tuple:
    df = df.copy()
    to_drop = [c for c in columns if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        return df, f"Dropped {len(to_drop)} column(s): {', '.join(to_drop)}"
    return df, "No columns dropped."


def impute_missing_values(df: pd.DataFrame, fill_dict: dict) -> tuple:
    report = []
    df = df.copy()
    for col, strategy in fill_dict.items():
        if col not in df.columns:
            continue
        s = str(strategy).strip().lower()
        if s == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
            report.append(f"Filled '{col}' with mean")
        elif s == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            report.append(f"Filled '{col}' with median")
        elif s == "mode":
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
            report.append(f"Filled '{col}' with mode")
        elif s == "drop":
            df = df.dropna(subset=[col])
            report.append(f"Dropped rows with missing '{col}'")
        else:
            try:
                df[col] = df[col].fillna(float(strategy))
            except (ValueError, TypeError):
                df[col] = df[col].fillna(strategy)
            report.append(f"Filled '{col}' with value: {strategy}")
    return df, "\n".join(report) if report else "No missing values imputed."


def encode_categorical_features(df: pd.DataFrame, columns: list) -> tuple:
    report = []
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        le   = LabelEncoder()
        mask = df[col].notnull()
        df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
        df[col] = pd.to_numeric(df[col], errors="coerce")
        report.append(f"Encoded categorical '{col}' to integers")
    return df, "\n".join(report) if report else "No categorical columns encoded."


def scale_numeric_features(df: pd.DataFrame, columns: list) -> tuple:
    df = df.copy()
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if valid_cols:
        scaler = StandardScaler()
        df[valid_cols] = scaler.fit_transform(df[valid_cols])
        return df, f"Scaled {len(valid_cols)} column(s) with StandardScaler: {', '.join(valid_cols)}"
    return df, "No columns scaled."


def binarize_numeric_features(df: pd.DataFrame, columns: list = None, threshold: float = 0) -> tuple:
    if columns is None:
        columns = list(df.select_dtypes(include=["int64", "float64"]).columns)
    df = df.copy()
    extraction_log = []
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        before_min  = df[col].min()
        before_max  = df[col].max()
        before_mean = round(df[col].mean(), 2)
        n_above     = int((df[col] > threshold).sum())
        new_col     = f"{col}_bin"
        df[new_col] = (df[col] > threshold).astype(int)
        pct_positive = round(n_above / len(df) * 100, 1) if len(df) > 0 else 0
        extraction_log.append({
            "original":     col,
            "new_col":      new_col,
            "threshold":    threshold,
            "before_min":   before_min,
            "before_max":   before_max,
            "before_mean":  before_mean,
            "pct_positive": pct_positive,
            "zeros":        int(len(df) - n_above),
            "ones":         n_above,
        })
    return df, extraction_log


# ─── Visualization ────────────────────────────────────────────────────────────

def _save(fig, name: str) -> str:
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _clean_white_style(fig, ax):
    """Apply consistent clean white theme to a chart."""
    bg = "white"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors="#333333", labelsize=9)
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")
    ax.title.set_color("black")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.8)
    ax.grid(False) # Removed grid lines as requested


def plot_histogram(df: pd.DataFrame, column_name: str, suffix: str = "") -> str:
    if column_name not in df.columns:
        return f"Error: Column '{column_name}' not found."
    data = df[column_name].dropna()
    if len(data) == 0:
        return f"Error: Column '{column_name}' has no non-null values."

    fig, ax = plt.subplots(figsize=(9, 5))
    _clean_white_style(fig, ax)
    n_unique = data.nunique()

    if n_unique <= 3:
        vc   = data.value_counts().sort_index()
        bars = ax.bar([str(v) for v in vc.index], vc.values,
                      color=["#818cf8", "#34d399", "#f472b6"][:len(vc)], 
                      edgecolor="none", width=1.0, alpha=0.85) # Filled gap, no edge lines
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom", fontsize=9, color="black")
        ax.set_title(f"Value Counts — {column_name}", fontsize=11, pad=10)
    else:
        has_var = data.std() > 1e-9
        try:
            sns.histplot(data, kde=has_var, ax=ax, color="#818cf8", alpha=0.7, edgecolor="none", binwidth=None, shrink=1.0)
        except Exception:
            sns.histplot(data, kde=False, ax=ax, color="#818cf8", alpha=0.7, edgecolor="none", binwidth=None, shrink=1.0)
        mean_v, med_v, std_v, skew_v = data.mean(), data.median(), data.std(), data.skew()
        ax.axvline(mean_v, color="#f87171", linestyle="--", linewidth=1.8, label=f"Mean   {mean_v:.3f}")
        ax.axvline(med_v,  color="#34d399", linestyle="--", linewidth=1.8, label=f"Median {med_v:.3f}")
        ax.legend(fontsize=9, framealpha=0.2, labelcolor="black")
        skew_label = ("highly " if abs(skew_v) > 1 else "") + (
            "right-skewed" if skew_v > 0.5 else "left-skewed" if skew_v < -0.5 else "symmetric"
        )
        ax.set_title(
            f"Distribution of {column_name}\nn={len(data):,}  |  std={std_v:.3f}  |  skew={skew_v:.2f}  ({skew_label})",
            fontsize=10, pad=10,
        )
    ax.set_xlabel(column_name)
    ax.set_ylabel("Count")
    fig.tight_layout()
    suf = f"_{suffix}" if suffix else ""
    return _save(fig, f"hist_{column_name}{suf}.png")


def plot_countplot(df: pd.DataFrame, x_col: str, hue_col: str = None, suffix: str = "") -> str:
    if x_col not in df.columns:
        return f"Error: Column '{x_col}' not found."
    fig, ax = plt.subplots(figsize=(8, 5))
    _clean_white_style(fig, ax)
    palette = sns.color_palette("cool", n_colors=df[x_col].nunique())
    if hue_col and hue_col in df.columns:
        sns.countplot(data=df, x=x_col, hue=hue_col, ax=ax)
    else:
        sns.countplot(data=df, x=x_col, ax=ax, palette=palette)
    ax.set_title(f"Countplot — {x_col}", fontsize=11, pad=10)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    suf = f"_{suffix}" if suffix else ""
    return _save(fig, f"count_{x_col}{suf}.png")


def plot_scatterplot(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None, suffix: str = "") -> str:
    if x_col not in df.columns or y_col not in df.columns:
        return f"Error: Required columns not found."
    fig, ax = plt.subplots(figsize=(8, 5))
    _clean_white_style(fig, ax)
    if hue_col and hue_col in df.columns:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, alpha=0.7)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color="#818cf8", alpha=0.6)
    ax.set_title(f"Scatter — {x_col} vs {y_col}", fontsize=11, pad=10)
    fig.tight_layout()
    suf = f"_{suffix}" if suffix else ""
    return _save(fig, f"scatter_{x_col}_{y_col}{suf}.png")


def plot_boxplot(df: pd.DataFrame, x_col: str, y_col: str, suffix: str = "") -> str:
    if x_col not in df.columns or y_col not in df.columns:
        return f"Error: Required columns not found."
    fig, ax = plt.subplots(figsize=(8, 5))
    _clean_white_style(fig, ax)
    sns.boxplot(data=df, x=x_col, y=y_col, ax=ax,
                palette="cool", linewidth=1.2, flierprops=dict(markerfacecolor="#f87171", alpha=0.5))
    ax.set_title(f"Boxplot — {x_col} vs {y_col}", fontsize=11, pad=10)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    suf = f"_{suffix}" if suffix else ""
    return _save(fig, f"box_{x_col}_{y_col}{suf}.png")


def plot_correlation_heatmap(df: pd.DataFrame, suffix: str = "") -> str:
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]
    if numeric_df.empty:
        return "No numeric columns to correlate."

    n = numeric_df.shape[1]
    figsize = (max(12, n * 0.9), max(9, n * 0.7))
    fig, ax = plt.subplots(figsize=figsize)
    _clean_white_style(fig, ax)

    corr      = numeric_df.corr().fillna(0)
    annot     = n <= 25
    font_size = max(5, 11 - (n // 4))

    sns.heatmap(
        corr, annot=annot, fmt=".2f",
        annot_kws={"size": font_size}, cmap="coolwarm",
        ax=ax, linewidths=0, linecolor="none",
        cbar_kws={"shrink": 0.8}, square=True
    )
    title_map = {
        "":        "Initial EDA — Correlation Heatmap (Original Data)",
        "cleaned": "Feature Engineering — Correlation Heatmap (Cleaned Data)",
    }
    ax.set_title(title_map.get(suffix, f"Correlation Heatmap ({suffix})"), fontsize=13, pad=14, color="black")
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor", fontsize=font_size + 1)
    plt.yticks(rotation=0, fontsize=font_size + 1)
    fig.tight_layout()
    fname = f"heatmap_correlation{'_' + suffix if suffix else ''}.png"
    return _save(fig, fname)
