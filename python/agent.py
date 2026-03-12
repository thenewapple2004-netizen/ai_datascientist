import os
import json
import glob
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import tools
from tools import load_csv
from llm import get_llm

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
    Impute missing values in the dataset.
    Pass a JSON string:
    { "col1": "mean", "col2": "median", "col3": "mode", "col4": "drop", "col5": 0 }
    """
    if tools._df is None: return "Error: No dataframe loaded."
    import json
    try: ops = json.loads(operations_json)
    except Exception: return "Error: Invalid JSON"
    new_df, report = tools.impute_missing_values(tools._df, ops)
    tools._df = new_df
    return report

@tool
def tool_encode_categorical_features(columns_json: str) -> str:
    """
    Encode categorical text features into integers (Label Encoding).
    Pass a JSON string with a list of columns:
    ["col1", "col2"]
    """
    if tools._df is None: return "Error: No dataframe loaded."
    import json
    try: cols = json.loads(columns_json)
    except Exception: return "Error: Invalid JSON"
    if not isinstance(cols, list): cols = [cols]
    new_df, report = tools.encode_categorical_features(tools._df, cols)
    tools._df = new_df
    return report

@tool
def tool_scale_numeric_features(columns_json: str) -> str:
    """
    Scale numeric features using StandardScaler (Standardization).
    Pass a JSON string with a list of numeric columns to scale:
    ["col1", "col2"]
    """
    if tools._df is None: return "Error: No dataframe loaded."
    import json
    try: cols = json.loads(columns_json)
    except Exception: return "Error: Invalid JSON"
    if not isinstance(cols, list): cols = [cols]
    new_df, report = tools.scale_numeric_features(tools._df, cols)
    tools._df = new_df
    return report

@tool
def tool_drop_useless_columns(columns_json: str) -> str:
    """
    Drop specific noisy or useless columns from the dataset.
    You MUST pass a JSON string with a LIST or ARRAY of multiple columns to drop in bulk:
    ["col1", "col2", "col3"]
    """
    if tools._df is None: return "Error: No dataframe loaded."
    import json
    try: cols = json.loads(columns_json)
    except Exception: return "Error: Invalid JSON"
    if not isinstance(cols, list): cols = [cols]
    new_df, report = tools.drop_useless_columns(tools._df, cols)
    tools._df = new_df
    return report

@tool
def tool_binarize_features(columns_json: str = "") -> str:
    """
    Feature Extraction: Convert numeric columns to binary features.

    For each numeric column, creates a NEW column named  <col>_bin  where:
      1 = value is above the threshold
      0 = value is at or below the threshold

    Pass a JSON string either as a global threshold:
    {
      "columns":   ["col_a", "col_b"],
      "threshold": 0
    }
    OR (Preferred) as a dictionary mapping specific columns to their own custom thresholds:
    {
      "col_a": 50,
      "col_b": 2.5
    }

    Returns a summary table showing each column: original range, threshold, % positive.
    """
    if tools._df is None:
        return "Error: No dataframe loaded."
    import json
    try:
        args = json.loads(columns_json) if columns_json.strip() else {}
    except Exception:
        args = {}

    log_all = []
    if isinstance(args, dict) and len(args) > 0 and not any(k in args for k in ["columns", "threshold"]):
        for col, thresh in args.items():
            try:
                t = float(thresh)
                new_df, log = tools.binarize_numeric_features(tools._df, columns=[col], threshold=t)
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

    lines = ["Feature Extraction (Binarization) Complete:",
             f"{'Original':<25} {'New Col':<30} {'Threshold':>10} {'Min':>8} {'Max':>8} {'Mean':>8} {'% Positive':>12} {'Zeros':>8} {'Ones':>8}"]
    lines.append("-" * 120)
    for r in log_all:
        lines.append(
            f"{r['original']:<25} {r['new_col']:<30} {r['threshold']:>10} "
            f"{r['before_min']:>8} {r['before_max']:>8} {r['before_mean']:>8} "
            f"{r['pct_positive']:>11}% {r['zeros']:>8} {r['ones']:>8}"
        )
    lines.append(f"\nNew shape: {tools._df.shape[0]} rows × {tools._df.shape[1]} columns.")
    return "\n".join(lines)

# ─── Tool Lists ───────────────────────────────────────────────────────────────

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
2. Study the actual columns, their data types, missing %, and unique value counts (cardinality).
3. Decide: do you have genuine ambiguity that requires user input to produce the best analysis?

ASK QUESTIONS ONLY IF TRULY NEEDED:
- Too many numeric columns and unclear which is the target/outcome variable
- Multiple categorical groupings and unclear which comparison is most relevant
- Purpose of analysis is unclear and different choices would produce completely different charts

DO NOT ASK IF:
- Dataset has an obvious target column (price, salary, target, label, survived, churn, etc.)
- Dataset is small/simple and the best visualizations are obvious
- All columns are numeric → correlations + distributions are clearly the best starting point

IF ASKING: output ONLY this JSON — no intro text, no explanation before or after it:
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

IF NOT ASKING: output exactly this text and nothing else:
READY_TO_ANALYZE
"""

PHASE2_PROMPT = """\
You are an expert AI Data Scientist performing exploratory data analysis.

PHASE 2 — GENERATE INITIAL EDA CHARTS:
Based on the dataset and any user preferences received, generate these charts in order:

DATA SCIENCE GUARDRAILS:
- NEVER plot a zero-variance column (only 1 unique value)
- NEVER plot ID/index columns (unique values = total rows)
- NEVER use a categorical column with >20 unique values for boxplot/countplot (unreadable)
- Prefer columns with meaningful variance and domain relevance

YOUR TASKS (in order):
1. `tool_plot_correlation_heatmap` — always first, reveals feature relationships
2. ONE best univariate: `tool_plot_histogram` (continuous) OR `tool_plot_countplot` (categorical)
3. ONE best bivariate: `tool_plot_scatterplot` (two continuous with strongest pattern)
4. ONE best multivariate: `tool_plot_boxplot` (valid categorical vs numeric)

After all 4 charts, write:
EDA_COMPLETE: <one sentence summary of the most important pattern found>
"""

PHASE_FE_PROMPT = """\
You are an expert AI Data Scientist performing Feature Engineering.

Initial EDA has been completed. Your job now is to clean and engineer the data in a strict, ordered pipeline.

═══════════════════════════════════════════════════
MANDATORY PIPELINE — FOLLOW THIS EXACT ORDER
═══════════════════════════════════════════════════

──────────────────────────────────────────────────
STEP 1 — READ THE DATA (ALWAYS FIRST)
──────────────────────────────────────────────────
Call `tool_get_dataframe_info` immediately. Study ALL columns: their data types,
missing %, unique value counts, and sample values.

──────────────────────────────────────────────────
STEP 2 — ASK CLARIFICATION FIRST (before any cleaning)
──────────────────────────────────────────────────
After reading the data, decide: Is there a column whose handling is genuinely ambiguous?
(e.g., a text column that could be either dropped or encoded; a column with 40% missing
that could be either imputed or dropped).

DO NOT ask about:
- Obviously useless ID columns (unique count == total rows) → always drop
- Zero-variance columns (only 1 unique value) → always drop
- Clearly categorical columns with text → always encode

IF clarification is needed → Output ONLY the JSON below, then STOP completely.
Do NOT call any cleaning tools before outputting this JSON:
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

IF no clarification is needed (or the user has already answered your questions in the
conversation history) → Skip this step entirely and proceed to STEP 3 now.

──────────────────────────────────────────────────
STEP 3 — EXECUTE ALL 5 FE OPERATIONS IN SEQUENCE
──────────────────────────────────────────────────
Apply ALL of the following in EXACT ORDER. Do NOT skip any step. Do NOT stop early.
Incorporate the user's MCQ answers (if any) when choosing columns.

3a. `tool_drop_useless_columns` — Compile ALL useless columns into ONE list and call ONCE:
    • ID Rule: unique_count == total_rows (pure row index — not useful as a feature)
    • Zero-Variance Rule: unique_count <= 1 (constant column, no information)
    • Sparsity Rule: missing% > 60% (too sparse to impute reliably)
    If NO columns match any rule, call with an empty list [].

3b. `tool_impute_missing_values` — Fill ALL remaining missing values:
    • Numeric columns → use "mean" or "median" (prefer median for skewed distributions)
    • Text / categorical columns → use "mode"
    • If there are NO missing values after dropping, skip this tool.

3c. `tool_binarize_features` — Create binary indicator columns for meaningful numerics:
    • Only binarize when it conveys real-world meaning (e.g., Age > 18, Income > 50000)
    • Use domain-sensible thresholds; do NOT binarize every numeric column blindly
    • If no meaningful binarization applies, skip this tool.

3d. `tool_encode_categorical_features` — Encode ALL remaining text/object columns to integers.
    • Label-encode every column whose dtype is still object/string.
    • If no categorical columns remain, skip this tool.

3e. `tool_scale_numeric_features` — Apply StandardScaler to ALL numeric columns.
    • Run this LAST, after encoding, so all columns are numeric.
    • If no numeric columns remain, skip this tool.

──────────────────────────────────────────────────
STEP 4 — GENERATE CLEANED CORRELATION HEATMAP
──────────────────────────────────────────────────
After all 5 operations above are done, call:
  `tool_plot_correlation_heatmap(suffix="cleaned")`
This produces the final cleaned-data correlation heatmap.

──────────────────────────────────────────────────
STEP 5 — REPORT COMPLETION
──────────────────────────────────────────────────
Output EXACTLY the following (no extra text before it):
FE_COMPLETE:
• <list each operation performed, one bullet per action>
• Examples: "Dropped 2 columns: PassengerId, Cabin", "Imputed 'Age' with median",
  "Binarized 'Age' → Age_bin (threshold=18)", "Encoded 'Sex', 'Embarked'",
  "Scaled 8 numeric columns with StandardScaler"
"""

PHASE_FINAL_PROMPT = """\
You are an expert AI Data Scientist generating final analysis charts on a cleaned dataset.

Feature engineering has been applied. The data is now clean and encoded.

PHASE FINAL — FEATURE EXTRACTION & VISUAL ANALYSIS:
1. Call `tool_get_dataframe_info` to understand the cleaned dataset.
2. Skip generating another heatmap — it was already produced during Feature Engineering.
3. Generate the BEST 3 charts that reveal the most important patterns in the cleaned data:
   - ONE optimized univariate chart (histogram of the most relevant numeric column)
   - ONE optimized bivariate chart (scatter of the two most strongly correlated numeric columns)
   - ONE optimized multivariate chart (boxplot of the most informative category vs numeric pair)

CRITICAL REQUIREMENT: For EVERY chart you generate in step 3, you MUST pass `suffix="final"`.
For example: `tool_plot_histogram(column_name="price", suffix="final")`
If you do not pass suffix="final", the charts will overwrite the initial EDA and the user won't see them!

DATA SCIENCE GUARDRAILS:
- Skip ID cols, zero-variance, high-cardinality cats.
- CRITICAL: You MUST ONLY plot columns that CURRENTLY exist in the dataset as reported by `tool_get_dataframe_info`. Do NOT attempt to plot columns that were dropped during Feature Engineering (like 'cut', 'color', 'clarity' etc. if they are missing).

After generating all charts, output:
FINAL_COMPLETE: <3 concise bullet points of the key insights from the cleaned data>
"""

# ─── Agent Factory ────────────────────────────────────────────────────────────

def create_agent(phase: str = "eda"):
    prompts = {
        "read":  PHASE1_PROMPT,
        "eda":   PHASE2_PROMPT,
        "fe":    PHASE_FE_PROMPT,
        "final": PHASE_FINAL_PROMPT,
    }
    tool_sets = {
        "read":  EDA_TOOLS,
        "eda":   EDA_TOOLS,
        "fe":    FE_TOOLS,
        "final": FINAL_TOOLS,
    }
    prompt   = prompts.get(phase, PHASE2_PROMPT)
    tool_set = tool_sets.get(phase, EDA_TOOLS)
    return create_react_agent(get_llm(), tools=tool_set, prompt=prompt)


def run_analysis_graph(csv_path: str, messages: list, phase: str = "read") -> dict:
    """
    Runs the LangGraph agent for a specific pipeline phase.

    Phases:
      "read"  → Phase 1: read data, ask MCQ if needed
      "eda"   → Phase 2: generate initial EDA charts
      "fe"    → Phase FE: feature engineering decisions + apply + heatmap
      "final" → Phase Final: final charts on cleaned data
    """
    load_csv(csv_path)
    agent  = create_agent(phase=phase)
    result = agent.invoke({"messages": messages})

    last_msg = result["messages"][-1].content

    # For FE phase: ONLY persist to disk when FE is fully complete.
    # If the agent paused to ask MCQ (clarification_needed JSON), do NOT save
    # partial tool-call side-effects — the next run will start from the clean CSV.
    if phase == "fe" and tools._df is not None:
        if "FE_COMPLETE" in last_msg:
            tools._df.to_csv(csv_path, index=False)
        # else: MCQ pause — leave disk untouched so next run starts fresh

    elif phase == "final" and tools._df is not None:
        tools._df.to_csv(csv_path, index=False)

    charts_dir = os.path.join(os.path.dirname(__file__), "charts")
    paths = sorted(glob.glob(os.path.join(charts_dir, "*.png"))) if os.path.isdir(charts_dir) else []

    return {
        "messages":    result["messages"],
        "chart_paths": paths,
        "success":     True,
        "answer":      last_msg,
    }