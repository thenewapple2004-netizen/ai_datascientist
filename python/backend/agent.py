"""
backend/agent.py — LangGraph agent orchestration with token tracking.
Phases: read → eda → fe → final
"""
import os
import glob
import json

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

from . import tools
from .tools import load_csv
from .llm import get_llm

# ─── Tool Definitions ─────────────────────────────────────────────────────────

@tool
def tool_get_dataframe_info(dummy: str = "") -> str:
    """Returns dataset info: shape, column names, data types, missing %, unique counts. ALWAYS CALL THIS FIRST."""
    return tools.get_dataframe_info(tools._df)


@tool
def tool_plot_correlation_heatmap(suffix: str = "") -> str:
    """Generates and saves a correlation heatmap. Use suffix='' for initial, suffix='cleaned' after feature engineering."""
    return tools.plot_correlation_heatmap(tools._df, suffix=suffix)


@tool
def tool_plot_histogram(column_name: str, suffix: str = "") -> str:
    """Generates a univariate histogram with KDE for a continuous column."""
    return tools.plot_histogram(tools._df, column_name, suffix)


@tool
def tool_plot_countplot(column_name: str, suffix: str = "") -> str:
    """Generates a univariate countplot for a categorical column."""
    return tools.plot_countplot(tools._df, column_name, suffix=suffix)


@tool
def tool_plot_scatterplot(x_col: str, y_col: str, suffix: str = "") -> str:
    """Generates a bivariate scatterplot comparing two continuous columns."""
    return tools.plot_scatterplot(tools._df, x_col, y_col, suffix=suffix)


@tool
def tool_plot_boxplot(x_col: str, y_col: str, suffix: str = "") -> str:
    """Generates a multivariate boxplot: x_col=categorical grouping, y_col=numeric."""
    return tools.plot_boxplot(tools._df, x_col, y_col, suffix)


@tool
def tool_impute_missing_values(operations_json: str) -> str:
    """
    Impute missing values. Pass JSON: { "col1": "mean", "col2": "median", "col3": "mode", "col4": "drop" }
    """
    if tools._df is None:
        return "Error: No dataframe loaded."
    try:
        ops = json.loads(operations_json)
    except Exception:
        return "Error: Invalid JSON"
    new_df, report = tools.impute_missing_values(tools._df, ops)
    tools._df = new_df
    return report


@tool
def tool_encode_categorical_features(columns_json: str) -> str:
    """
    Encode categorical text features into integers (Label Encoding).
    Pass a JSON array: ["col1", "col2"]
    """
    if tools._df is None:
        return "Error: No dataframe loaded."
    try:
        cols = json.loads(columns_json)
    except Exception:
        return "Error: Invalid JSON"
    if not isinstance(cols, list):
        cols = [cols]
    new_df, report = tools.encode_categorical_features(tools._df, cols)
    tools._df = new_df
    return report


@tool
def tool_scale_numeric_features(columns_json: str) -> str:
    """
    Scale numeric features using StandardScaler. Pass a JSON array: ["col1", "col2"]
    """
    if tools._df is None:
        return "Error: No dataframe loaded."
    try:
        cols = json.loads(columns_json)
    except Exception:
        return "Error: Invalid JSON"
    if not isinstance(cols, list):
        cols = [cols]
    new_df, report = tools.scale_numeric_features(tools._df, cols)
    tools._df = new_df
    return report


@tool
def tool_drop_useless_columns(columns_json: str) -> str:
    """
    Drop noisy or useless columns. Pass a JSON array: ["col1", "col2", "col3"]
    """
    if tools._df is None:
        return "Error: No dataframe loaded."
    try:
        cols = json.loads(columns_json)
    except Exception:
        return "Error: Invalid JSON"
    if not isinstance(cols, list):
        cols = [cols]
    new_df, report = tools.drop_useless_columns(tools._df, cols)
    tools._df = new_df
    return report


@tool
def tool_binarize_features(columns_json: str = "") -> str:
    """
    Convert numeric columns to binary (1/0) features.

    Pass either per-column thresholds:
      { "Age": 18, "Fare": 50 }
    Or a global threshold with column list:
      { "columns": ["Age", "Fare"], "threshold": 0 }

    Returns a summary of each binarized column.
    """
    if tools._df is None:
        return "Error: No dataframe loaded."
    try:
        args = json.loads(columns_json) if columns_json.strip() else {}
    except Exception:
        args = {}

    log_all = []
    if isinstance(args, dict) and args and not any(k in args for k in ["columns", "threshold"]):
        for col, thresh in args.items():
            try:
                new_df, log = tools.binarize_numeric_features(tools._df, columns=[col], threshold=float(thresh))
                tools._df = new_df
                log_all.extend(log)
            except Exception:
                pass
    else:
        cols      = args.get("columns", None)
        threshold = float(args.get("threshold", 0))
        new_df, log = tools.binarize_numeric_features(tools._df, columns=cols, threshold=threshold)
        tools._df = new_df
        log_all.extend(log)

    if not log_all:
        return "No numeric columns found to binarize."

    lines = [
        "Feature Extraction (Binarization) Complete:",
        f"{'Original':<25} {'New Col':<30} {'Threshold':>10} {'Min':>8} {'Max':>8} {'Mean':>8} {'% Positive':>12} {'Zeros':>8} {'Ones':>8}",
        "-" * 120,
    ]
    for r in log_all:
        lines.append(
            f"{r['original']:<25} {r['new_col']:<30} {r['threshold']:>10} "
            f"{r['before_min']:>8} {r['before_max']:>8} {r['before_mean']:>8} "
            f"{r['pct_positive']:>11}% {r['zeros']:>8} {r['ones']:>8}"
        )
    lines.append(f"\nNew shape: {tools._df.shape[0]} rows × {tools._df.shape[1]} columns.")
    return "\n".join(lines)


# ─── Tool Sets ────────────────────────────────────────────────────────────────

EDA_TOOLS = [
    tool_get_dataframe_info,
    tool_plot_correlation_heatmap,
    tool_plot_histogram,
    tool_plot_countplot,
    tool_plot_scatterplot,
    tool_plot_boxplot,
]

FE_TOOLS = [
    tool_get_dataframe_info,
    tool_plot_correlation_heatmap,
    tool_impute_missing_values,
    tool_binarize_features,
    tool_encode_categorical_features,
    tool_scale_numeric_features,
    tool_drop_useless_columns,
]

FINAL_TOOLS = [
    tool_get_dataframe_info,
    tool_plot_correlation_heatmap,
    tool_plot_histogram,
    tool_plot_countplot,
    tool_plot_scatterplot,
    tool_plot_boxplot,
]

# ─── Phase Prompts ────────────────────────────────────────────────────────────

PHASE1_PROMPT = """\
You are an expert AI Data Scientist. A CSV dataset has just been uploaded.

PHASE 1 — READ DATA, DECIDE IF YOU NEED CLARIFICATION FIRST:
1. Call `tool_get_dataframe_info` — MANDATORY first step.
2. Study the actual columns, their data types, missing %, and unique value counts.
3. Decide: do you have genuine ambiguity that requires user input?

ASK QUESTIONS ONLY IF TRULY NEEDED (e.g., unclear target variable, multiple groupings).
DO NOT ASK IF: dataset has obvious target column, dataset is small/simple.

IF ASKING: output ONLY this JSON — no intro text, no explanation:
{
  "clarification_needed": true,
  "questions": [
    {
      "id": 1,
      "question": "<specific question using REAL column names>",
      "hint": "<why this changes the analysis>",
      "options": ["<real_col_1>", "<real_col_2>", "<real_col_3>", "No preference"]
    }
  ]
}

IF NOT ASKING: output exactly this and nothing else:
READY_TO_ANALYZE
"""

PHASE2_PROMPT = """\
You are an expert AI Data Scientist performing exploratory data analysis.

PHASE 2 — GENERATE INITIAL EDA CHARTS (in order):

DATA SCIENCE GUARDRAILS:
- NEVER plot a zero-variance column (only 1 unique value)
- NEVER plot ID/index columns (unique values = total rows)
- NEVER use a categorical column with >20 unique values for boxplot/countplot

YOUR TASKS (in order):
1. `tool_plot_correlation_heatmap` — always first
2. Generate the BEST 5-6 charts that reveal initial patterns:
   - 2-3 Univariate charts (histograms/countplots of most important numerical/categorical features)
   - 2 Bivariate charts (scatterplots of interesting relationships)
   - 1 Multivariate chart (boxplot showing groups)

After all charts, write:
EDA_COMPLETE: <one sentence summary of the most important pattern found>
"""

PHASE_FE_PROMPT = """\
You are an expert AI Data Scientist performing Feature Engineering.

MANDATORY PIPELINE — FOLLOW THIS EXACT ORDER:

STEP 1 — READ THE DATA (ALWAYS FIRST)
Call `tool_get_dataframe_info` immediately.

STEP 2 — ASK CLARIFICATION (before any cleaning, only if genuinely ambiguous)
If clarification needed, output ONLY this JSON then STOP:
{
  "clarification_needed": true,
  "questions": [
    {
      "id": 1,
      "question": "<specific question using REAL column names>",
      "hint": "<how user's choice changes the cleaning step>",
      "options": ["<option_A>", "<option_B>", "<option_C>"]
    }
  ]
}

STEP 3 — EXECUTE ALL 5 FE OPERATIONS IN SEQUENCE (if no clarification needed):

3a. `tool_drop_useless_columns` — Drop ID cols (unique==rows), zero-variance, >60% missing. Call ONCE with ALL cols.
3b. `tool_impute_missing_values` — Fill all remaining missing values (mean/median for numeric, mode for text).
3c. `tool_binarize_features` — Binarize meaningful numeric cols (e.g., Age>18). Skip if not meaningful.
3d. `tool_encode_categorical_features` — Encode ALL remaining text/object columns.
3e. `tool_scale_numeric_features` — StandardScaler on ALL numeric columns. Run LAST.

STEP 4 — GENERATE CLEANED HEATMAP
Call: `tool_plot_correlation_heatmap(suffix="cleaned")`

STEP 5 — REPORT
Output EXACTLY:
FE_COMPLETE:
• <bullet per action performed>
"""

PHASE_FINAL_PROMPT = """\
You are an expert AI Data Scientist generating final analysis charts on a cleaned dataset.

PHASE FINAL — FINAL ANALYSIS CHARTS:
1. Call `tool_get_dataframe_info` to understand the cleaned dataset.
2. Do NOT generate another heatmap (already done in Feature Engineering).
3. Generate 5-6 COMPREHENSIVE charts that provide deep insight:
   - 2 optimized univariate charts (focus on target distribution and key features)
   - 2 optimized bivariate charts (focus on relationships with the potential target)
   - 2 optimized multivariate charts (complex interactions, outlier detection via boxplots)

CRITICAL: Pass `suffix="final"` for EVERY chart you generate.
CRITICAL: Only plot columns that CURRENTLY exist per `tool_get_dataframe_info`.

After all charts, output:
FINAL_COMPLETE: <3 concise bullet points of key insights>
"""

# ─── Agent Factory ────────────────────────────────────────────────────────────

def create_agent(phase: str = "eda"):
    prompts   = {"read": PHASE1_PROMPT, "eda": PHASE2_PROMPT, "fe": PHASE_FE_PROMPT, "final": PHASE_FINAL_PROMPT}
    tool_sets = {"read": EDA_TOOLS, "eda": EDA_TOOLS, "fe": FE_TOOLS, "final": FINAL_TOOLS}
    return create_react_agent(
        get_llm(),
        tools=tool_sets.get(phase, EDA_TOOLS),
        prompt=prompts.get(phase, PHASE2_PROMPT),
    )


# ─── Token Counter ────────────────────────────────────────────────────────────

def _count_tokens(messages: list) -> dict:
    """Parse usage_metadata from AIMessages to count tokens."""
    input_tokens  = 0
    output_tokens = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "usage_metadata", None):
            meta = msg.usage_metadata
            input_tokens  += meta.get("input_tokens", 0)
            output_tokens += meta.get("output_tokens", 0)
    return {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  input_tokens + output_tokens,
    }


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def run_analysis_graph(csv_path: str, messages: list, phase: str = "read") -> dict:
    """
    Run the LangGraph agent for a specific pipeline phase.
    Returns messages, chart_paths, token_usage, and the last answer.
    """
    load_csv(csv_path)
    agent  = create_agent(phase=phase)
    result = agent.invoke({"messages": messages})

    last_msg    = result["messages"][-1].content
    token_usage = _count_tokens(result["messages"])
    token_usage["phase"] = phase

    # Persist modified dataframe to disk only on completion (not on MCQ pause)
    if phase == "fe" and tools._df is not None and "FE_COMPLETE" in last_msg:
        tools._df.to_csv(csv_path, index=False)
    elif phase == "final" and tools._df is not None:
        tools._df.to_csv(csv_path, index=False)

    charts_dir = tools.CHARTS_DIR
    paths = sorted(glob.glob(os.path.join(charts_dir, "*.png"))) if os.path.isdir(charts_dir) else []

    return {
        "messages":    result["messages"],
        "chart_paths": paths,
        "success":     True,
        "answer":      last_msg,
        "token_usage": token_usage,
    }
