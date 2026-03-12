import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

# Global DataFrame accessible by agent.py via tools._df
_df = None
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")

def load_csv(csv_path: str):
    """Loads the CSV into the global dataframe."""
    global _df
    _df = pd.read_csv(csv_path)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return f"Loaded {_df.shape[0]} rows and {_df.shape[1]} columns."

def get_dataframe_info(df: pd.DataFrame) -> str:
    """Returns shape, column names, data types, missing values, and unique counts."""
    if df is None:
        return "Error: No dataframe loaded."
    buf = []
    buf.append(f"Data shape: {df.shape} ---> (rows, columns)")
    top_values = []
    for col in df.columns:
        vals = df[col].dropna().unique()
        if len(vals) <= 10:
            top_values.append(", ".join(map(str, vals)))
        else:
            top_values.append(", ".join(map(str, vals[:5])) + " ...")

    summary_df = pd.DataFrame({
        'Data Type':     df.dtypes,
        'Missing Count': df.isnull().sum(),
        'Missing %':     (df.isnull().mean() * 100).round(1),
        'Unique Values': df.nunique(),
        'Top Distinct':  top_values
    })
    buf.append("Column Details:")
    buf.append(summary_df.to_string())
    return "\n".join(buf)

from sklearn.preprocessing import LabelEncoder, StandardScaler

def impute_missing_values(df: pd.DataFrame, fill_dict: dict) -> tuple:
    report = []
    df = df.copy()
    for col, strategy in fill_dict.items():
        if col not in df.columns:
            continue
        s = str(strategy).strip().lower()
        if s == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
            report.append(f"Filled missing '{col}' with mean")
        elif s == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            report.append(f"Filled missing '{col}' with median")
        elif s == "mode":
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
            report.append(f"Filled missing '{col}' with mode")
        elif s == "drop":
            df = df.dropna(subset=[col])
            report.append(f"Dropped rows with missing '{col}'")
        else:
            try:
                df[col] = df[col].fillna(float(strategy))
            except (ValueError, TypeError):
                df[col] = df[col].fillna(strategy)
            report.append(f"Filled missing '{col}' with value: {strategy}")
    return df, "\n".join(report) if report else "No missing values imputed."

def encode_categorical_features(df: pd.DataFrame, columns: list) -> tuple:
    report = []
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        # Only fit on non-nulls to preserve NaN (or just fill first)
        mask = df[col].notnull()
        df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
        df[col] = pd.to_numeric(df[col], errors='coerce')
        report.append(f"Encoded categorical '{col}' to integers")
    return df, "\n".join(report) if report else "No categorical columns encoded."

def scale_numeric_features(df: pd.DataFrame, columns: list) -> tuple:
    report = []
    df = df.copy()
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if valid_cols:
        scaler = StandardScaler()
        df[valid_cols] = scaler.fit_transform(df[valid_cols])
        report.append(f"Scaled {len(valid_cols)} column(s) using StandardScaler: {', '.join(valid_cols)}")
    return df, "\n".join(report) if report else "No columns scaled."

def drop_useless_columns(df: pd.DataFrame, columns: list) -> tuple:
    report = []
    df = df.copy()
    to_drop = [c for c in columns if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        report.append(f"Dropped {len(to_drop)} column(s): {', '.join(to_drop)}")
    return df, "\n".join(report) if report else "No columns dropped."


def binarize_numeric_features(df: pd.DataFrame, columns: list = None, threshold: float = 0) -> tuple:
    """
    Convert numeric columns to binary (1/0) based on a threshold.
    Default threshold=0: any value > 0 → 1, else → 0.

    Args:
        df: DataFrame to transform
        columns: list of column names (default: all int64/float64 columns)
        threshold: value to threshold on (default 0)

    Returns:
        (modified_df, extraction_log list of dicts with before/after stats)
    """
    if columns is None:
        columns = list(df.select_dtypes(include=['int64', 'float64']).columns)

    df = df.copy()
    extraction_log = []
    for col in columns:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Before stats
        before_min  = df[col].min()
        before_max  = df[col].max()
        before_mean = round(df[col].mean(), 2)
        n_nonzero   = int((df[col] > threshold).sum())

        # Binarize
        new_col = f"{col}_bin"
        df[new_col] = (df[col] > threshold).astype(int)
        pct_positive = round(n_nonzero / len(df) * 100, 1) if len(df) > 0 else 0

        extraction_log.append({
            "original":      col,
            "new_col":       new_col,
            "threshold":     threshold,
            "before_min":    before_min,
            "before_max":    before_max,
            "before_mean":   before_mean,
            "pct_positive":  pct_positive,
            "zeros":         int(len(df) - n_nonzero),
            "ones":          n_nonzero,
        })

    return df, extraction_log

def plot_histogram(df: pd.DataFrame, column_name: str, suffix: str = "") -> str:
    """Generates an annotated histogram with KDE, mean/median markers, and stats."""
    if column_name not in df.columns:
        return f"Error: Column '{column_name}' not found in dataset."

    data = df[column_name].dropna()
    if len(data) == 0:
        return f"Error: Column '{column_name}' has no non-null values."

    n_unique = data.nunique()
    fig, ax = plt.subplots(figsize=(9, 5))

    if n_unique <= 3:
        # Binary / near-binary column — a bar chart is clearer than a histogram
        vc = data.value_counts().sort_index()
        bars = ax.bar([str(v) for v in vc.index], vc.values,
                      color=['#818cf8', '#34d399', '#f472b6'][:len(vc)], edgecolor='none', alpha=0.85)
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=9, color='white')
        ax.set_title(f"Value Counts — {column_name}  (binary/categorical)", fontsize=11)
        ax.set_xlabel(column_name)
        ax.set_ylabel("Count")
    else:
        # Continuous column — histogram + KDE + mean/median markers
        has_variance = data.std() > 1e-9
        try:
            sns.histplot(data, kde=has_variance, ax=ax,
                         color='#818cf8', alpha=0.7, edgecolor='none')
        except Exception:
            sns.histplot(data, kde=False, ax=ax, color='#818cf8', alpha=0.7, edgecolor='none')

        mean_v   = data.mean()
        median_v = data.median()
        std_v    = data.std()
        skew_v   = data.skew()

        ax.axvline(mean_v,   color='#f87171', linestyle='--', linewidth=1.8,
                   label=f"Mean   {mean_v:.3f}")
        ax.axvline(median_v, color='#34d399', linestyle='--', linewidth=1.8,
                   label=f"Median {median_v:.3f}")
        ax.legend(fontsize=9, framealpha=0.3)

        skew_label = "symmetric"
        if abs(skew_v) > 1:
            skew_label = "highly right-skewed" if skew_v > 0 else "highly left-skewed"
        elif abs(skew_v) > 0.5:
            skew_label = "right-skewed" if skew_v > 0 else "left-skewed"

        note = " (StandardScaler normalized)" if abs(mean_v) < 0.5 and abs(std_v - 1) < 0.3 else ""
        ax.set_title(
            f"Distribution of {column_name}{note}\n"
            f"n={len(data):,}  |  std={std_v:.3f}  |  skew={skew_v:.2f}  ({skew_label})",
            fontsize=10
        )
        ax.set_xlabel(column_name)
        ax.set_ylabel("Count")

    fig.tight_layout()
    suf  = f"_{suffix}" if suffix else ""
    path = os.path.join(CHARTS_DIR, f"hist_{column_name}{suf}.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path

def plot_scatterplot(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None, suffix: str = "") -> str:
    """Generates a bivariate scatterplot."""
    if x_col not in df.columns or y_col not in df.columns:
        return f"Error: Required columns '{x_col}' and/or '{y_col}' not found in dataset."
    fig, ax = plt.subplots(figsize=(8, 5))
    if hue_col and hue_col in df.columns:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f"Scatterplot: {x_col} vs {y_col}")
    suf = f"_{suffix}" if suffix else ""
    path = os.path.join(CHARTS_DIR, f"scatter_{x_col}_{y_col}{suf}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_countplot(df: pd.DataFrame, x_col: str, hue_col: str = None, suffix: str = "") -> str:
    """Generates a countplot for a categorical variable."""
    if x_col not in df.columns:
        return f"Error: Column '{x_col}' not found in dataset."
    fig, ax = plt.subplots(figsize=(8, 5))
    if hue_col and hue_col in df.columns:
        sns.countplot(data=df, x=x_col, hue=hue_col, ax=ax)
    else:
        sns.countplot(data=df, x=x_col, ax=ax)
    ax.set_title(f"Countplot of {x_col}")
    plt.xticks(rotation=45)
    suf = f"_{suffix}" if suffix else ""
    path = os.path.join(CHARTS_DIR, f"count_{x_col}{suf}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_boxplot(df: pd.DataFrame, x_col: str, y_col: str, suffix: str = "") -> str:
    """Generates a multivariate boxplot."""
    if x_col not in df.columns or y_col not in df.columns:
        return f"Error: Required columns '{x_col}' and/or '{y_col}' not found in dataset."
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=x_col, y=y_col, palette='Set1', ax=ax)
    ax.set_title(f"Boxplot: {x_col} vs {y_col}")
    plt.xticks(rotation=45)
    suf = f"_{suffix}" if suffix else ""
    path = os.path.join(CHARTS_DIR, f"box_{x_col}_{y_col}{suf}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_correlation_heatmap(df: pd.DataFrame, suffix: str = "") -> str:
    """Generates a correlation heatmap for all numeric columns."""
    import numpy as np
    numeric_df = df.select_dtypes(include='number')
    
    # Drop zero-variance columns to avoid NaN in correlations
    if not numeric_df.empty:
        numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

    if numeric_df.empty:
        return "No numeric columns to correlate."
        
    fig, ax = plt.subplots(figsize=(18, 14))
    corr = numeric_df.corr().fillna(0)
    
    # Annotate only when few enough columns fit without overlap
    _n = corr.shape[0]
    _annot = _n <= 30
    _font_size = max(5, 12 - (_n // 3))
    sns.heatmap(corr, annot=_annot, fmt=".2f",
                annot_kws={"size": _font_size}, cmap='coolwarm', ax=ax)
    title_map = {
        "": "Initial EDA — Correlation Heatmap (Original Data)",
        "cleaned": "Feature Engineering — Correlation Heatmap (Cleaned Data)",
    }
    title = title_map.get(suffix, f"Correlation Heatmap ({suffix})")
    ax.set_title(title, fontsize=13, pad=12)
    
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.yticks(rotation=0)
    fig.tight_layout()
    
    fname = f"heatmap_correlation{'_' + suffix if suffix else ''}.png"
    path = os.path.join(CHARTS_DIR, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path