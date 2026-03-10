import os
import pandas as pd
import seaborn as sns

_df = None
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")

def load_csv(csv_path: str):
    """Loads the CSV into the global dataframe."""
    global _df
    _df = pd.read_csv(csv_path)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return f"Loaded {_df.shape[0]} rows and {_df.shape[1]} columns."

def get_dataframe_info(df: pd.DataFrame) -> str:
    """Returns the shape, column names, and data types of the dataframe."""
    buf = []
    buf.append(f"Data shape: {df.shape} ---> (rows, columns)")
    buf.append("Columns and Data Types:")
    buf.append(df.dtypes.to_string())
    buf.append("Missing Values:")
    buf.append(df.isnull().sum().to_string())
    return "\n".join(buf)

def plot_histogram(df: pd.DataFrame, column_name: str):
    """Generates a histogram."""
    plot = sns.histplot(df[column_name].dropna(), kde=True)
    plot.set_title(f"Distribution of {column_name}")
    path = os.path.join(CHARTS_DIR, f"hist_{column_name}.png")
    plot.figure.savefig(path, bbox_inches="tight")
    plot.figure.clf()
    return path

def plot_scatterplot(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None):
    """Generates a scatterplot."""
    if hue_col and hue_col in df.columns:
        plot = sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
    else:
        plot = sns.scatterplot(data=df, x=x_col, y=y_col)
    plot.set_title(f"Scatterplot: {x_col} vs {y_col}")
    path = os.path.join(CHARTS_DIR, f"scatter_{x_col}_{y_col}.png")
    plot.figure.savefig(path, bbox_inches="tight")
    plot.figure.clf()
    return path

def plot_countplot(df: pd.DataFrame, x_col: str, hue_col: str = None):
    """Generates a countplot."""
    if hue_col and hue_col in df.columns:
        plot = sns.countplot(data=df, x=x_col, hue=hue_col)
    else:
        plot = sns.countplot(data=df, x=x_col)
    plot.set_title(f"Countplot of {x_col}")
    plot.figure.autofmt_xdate()
    path = os.path.join(CHARTS_DIR, f"count_{x_col}.png")
    plot.figure.savefig(path, bbox_inches="tight")
    plot.figure.clf()
    return path

def plot_boxplot(df: pd.DataFrame, x_col: str, y_col: str):
    """Generates a boxplot."""
    plot = sns.boxplot(data=df, x=x_col, y=y_col, palette='Set1')
    plot.set_title(f"Boxplot: {x_col} vs {y_col}")
    plot.figure.autofmt_xdate()
    path = os.path.join(CHARTS_DIR, f"box_{x_col}_{y_col}.png")
    plot.figure.savefig(path, bbox_inches="tight")
    plot.figure.clf()
    return path

def get_correlation_matrix(df: pd.DataFrame) -> str:
    """Returns the correlation matrix for all numeric columns."""
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty: return "No numeric columns found."
    return numeric_df.corr().to_string()

def plot_correlation_heatmap(df: pd.DataFrame):
    """Generates a correlation heatmap."""
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty: return None
    plot = sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plot.set_title("Correlation Heatmap")
    path = os.path.join(CHARTS_DIR, "heatmap_correlation.png")
    plot.figure.savefig(path, bbox_inches="tight")
    plot.figure.clf()
    return path
