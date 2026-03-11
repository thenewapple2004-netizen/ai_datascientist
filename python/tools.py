import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

_df = None
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")

def load_csv(csv_path: str):
    """Loads the CSV into the global dataframe."""
    global _df
    _df = pd.read_csv(csv_path)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return f"Loaded {_df.shape[0]} rows and {_df.shape[1]} columns."

def get_dataframe_info(df: pd.DataFrame) -> str:
    """Returns the shape, column names, data types and missing values."""
    buf = []
    buf.append(f"Data shape: {df.shape} ---> (rows, columns)")
    buf.append("Columns and Data Types:")
    buf.append(df.dtypes.to_string())
    buf.append("Missing Values:")
    buf.append(df.isnull().sum().to_string())
    return "\n".join(buf)

def plot_histogram(df: pd.DataFrame, column_name: str) -> str:
    """Generates a histogram with KDE for a continuous variable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column_name].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {column_name}")
    path = os.path.join(CHARTS_DIR, f"hist_{column_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_scatterplot(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None) -> str:
    """Generates a bivariate scatterplot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if hue_col and hue_col in df.columns:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(f"Scatterplot: {x_col} vs {y_col}")
    path = os.path.join(CHARTS_DIR, f"scatter_{x_col}_{y_col}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_countplot(df: pd.DataFrame, x_col: str, hue_col: str = None) -> str:
    """Generates a countplot for a categorical variable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if hue_col and hue_col in df.columns:
        sns.countplot(data=df, x=x_col, hue=hue_col, ax=ax)
    else:
        sns.countplot(data=df, x=x_col, ax=ax)
    ax.set_title(f"Countplot of {x_col}")
    plt.xticks(rotation=45)
    path = os.path.join(CHARTS_DIR, f"count_{x_col}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_boxplot(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    """Generates a multivariate boxplot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x=x_col, y=y_col, palette='Set1', ax=ax)
    ax.set_title(f"Boxplot: {x_col} vs {y_col}")
    plt.xticks(rotation=45)
    path = os.path.join(CHARTS_DIR, f"box_{x_col}_{y_col}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def get_correlation_matrix(df: pd.DataFrame) -> str:
    """Returns the correlation matrix for all numeric columns."""
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        return "No numeric columns found."
    return numeric_df.corr().to_string()

def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Generates a correlation heatmap for all numeric columns."""
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        return "No numeric columns to correlate."
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    path = os.path.join(CHARTS_DIR, "heatmap_correlation.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
