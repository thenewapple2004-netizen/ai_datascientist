import os
import glob
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import tools
from tools import load_csv
from llm import get_llm



@tool
def tool_get_dataframe_info(dummy: str = "") -> str:
    """Returns dataset info: shape, column names, types, missing values. ALWAYS CALL THIS FIRST."""
    return tools.get_dataframe_info(tools._df)

@tool
def tool_plot_correlation_heatmap(dummy: str = "") -> str:
    """Generates and saves a correlation heatmap of all numeric columns. Call this to understand feature relationships."""
    return tools.plot_correlation_heatmap(tools._df)

@tool
def tool_plot_histogram(column_name: str) -> str:
    """Generates a univariate histogram with KDE for a single continuous column."""
    return tools.plot_histogram(tools._df, column_name)

@tool
def tool_plot_countplot(column_name: str) -> str:
    """Generates a univariate countplot for a single categorical column."""
    return tools.plot_countplot(tools._df, column_name)

@tool
def tool_plot_scatterplot(x_col: str, y_col: str) -> str:
    """Generates a bivariate scatterplot comparing two continuous columns."""
    return tools.plot_scatterplot(tools._df, x_col, y_col)

@tool
def tool_plot_boxplot(x_col: str, y_col: str) -> str:
    """Generates a multivariate boxplot: x_col = categorical grouping column, y_col = numeric column to compare."""
    return tools.plot_boxplot(tools._df, x_col, y_col)

EDA_TOOLS = [
    tool_get_dataframe_info,
    tool_plot_correlation_heatmap,
    tool_plot_histogram,
    tool_plot_countplot,
    tool_plot_scatterplot,
    tool_plot_boxplot,
]



PHASE1_PROMPT = """\
You are an expert AI Data Scientist. A CSV dataset has just been uploaded.

PHASE 1 — YOUR ONLY JOB RIGHT NOW:
1. Call `tool_get_dataframe_info` to understand the dataset.
2. Based on what you learn (columns, types, data), identify any ambiguities you have.
3. If you have ANY questions you need from the user before deciding what to visualize — ask them ALL NOW in one response.
4. If you are fully confident and have no questions, say exactly: "READY_TO_ANALYZE"

CRITICAL RULES:
- DO NOT call any chart tools yet. Charts come in Phase 2.
- DO NOT generate any charts now.
- If you have clarification questions, return them in this EXACT JSON format (nothing else outside it):
{{
  "clarification_needed": true,
  "questions": [
    {{
      "id": 1,
      "question": "Which column should be the primary focus of the analysis?",
      "hint": "This helps the AI choose the most relevant comparisons.",
      "options": ["Column_A", "Column_B", "Column_C", "No preference"]
    }},
    {{
      "id": 2,
      "question": "What type of patterns interest you most?",
      "hint": "Guides which chart types will be prioritized.",
      "options": ["Distributions", "Relationships between columns", "Category comparisons", "Show everything important"]
    }}
  ]
}}
- Include 1–3 questions, each with 3–5 options.
- If you are fully confident and need NO clarification: output exactly the text: READY_TO_ANALYZE
"""


PHASE2_PROMPT = """\
You are an expert AI Data Scientist. You have already read the dataset info and received the user's preferences.

PHASE 2 — GENERATE THE BEST VISUAL EDA:
Based on the dataset features and the user's answers, now generate EXACTLY these charts:
1. `tool_plot_correlation_heatmap` — always run this first to see feature relationships
2. ONE best univariate chart: use `tool_plot_histogram` for continuous columns OR `tool_plot_countplot` for categorical
3. ONE best bivariate chart: use `tool_plot_scatterplot` for two continuous columns that show the strongest pattern
4. ONE best multivariate chart: use `tool_plot_boxplot` comparing the best categorical vs numeric column pair

STRICT RULES:
- Choose the MOST INSIGHTFUL columns based on the data info and user preferences.
- Do NOT generate redundant or obvious charts.
- After generating all 4 charts, write a concise 3-bullet-point summary of what was found.
- That final summary is your LAST output. Do not say anything after it.
"""

def create_agent(phase: int = 2):
    prompt = PHASE1_PROMPT if phase == 1 else PHASE2_PROMPT
    return create_react_agent(get_llm(), tools=EDA_TOOLS, prompt=prompt)

def run_analysis_graph(csv_path: str, messages: list, phase: int = 1) -> dict:
    """
    Runs LangGraph agent.
    phase=1 → reads data, may ask MCQ clarification questions
    phase=2 → generates all charts using user answers as context
    """
    load_csv(csv_path)
    agent = create_agent(phase=phase)
    result = agent.invoke({"messages": messages})

    charts_dir = os.path.join(os.path.dirname(__file__), "charts")
    paths = []
    if os.path.isdir(charts_dir):
        paths = sorted(glob.glob(os.path.join(charts_dir, "*.png")))

    last_msg = result["messages"][-1].content

    return {
        "messages": result["messages"],
        "chart_paths": paths,
        "success": True,
        "answer": last_msg,
    }
