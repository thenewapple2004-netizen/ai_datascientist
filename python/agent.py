"""
agent.py — LangChain Agent Controller + Tool Registry & Executor

Architecture role (matches diagram):
  User Query
      ↓
  LangChain Agent Controller  (this file)
      ↓
  LLM (OpenAI G-Functions / ReAct)  ← from llm.py
      ↓  ↑  (loop until done)
  Tool Registry & Executor  ← EDA_TOOLS from tools.py
      ↓
  Final result dict → Report Generator (next step)

Uses: LangGraph's create_react_agent (modern replacement for AgentExecutor)
"""

import os
import glob

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm
from tools import EDA_TOOLS, load_csv_into_tensor


# ---------------------------------------------------------------------------
# System prompt — tells the LLM its role in the agent loop
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a senior AI Data Scientist. Your job is to explore a CSV dataset and
discover the MOST INTERESTING patterns, relationships, and anomalies in it.

You have a rich set of chart tools. YOU decide which ones to call and with which
columns — based on what you learn about the data.

Decision rules you MUST follow:
1. Always start with get_dataframe_info to learn column names and dtypes.
2. Run generate_eda_summary to see distributions and missing data.
3. Identify:
   - Numeric columns  → use generate_distribution_charts, generate_boxplot_charts,
                         detect_outliers_report, generate_scatter_comparison, generate_eda_charts(heatmap)
   - Categorical cols → use generate_countplot for each one
   - Categorical + Numeric pairs → use generate_barplot_comparison and generate_boxplot_by_category
     for the most MEANINGFUL combinations (e.g. survival rate by class, price by cut)
4. For scatter/comparison charts, choose the pair with the HIGHEST expected correlation
   or the most domain-relevant relationship — do NOT just pick the first two columns.
5. For hue in scatter, use the most meaningful binary/categorical column.
6. Generate generate_eda_charts(heatmap) to see ALL correlations.
7. Generate generate_missing_value_chart to visualise data quality.

Be intelligent. Think like a data scientist — choose the charts that tell the best story.
"""

# ---------------------------------------------------------------------------
# Auto-EDA query — LLM drives the analysis, picks columns intelligently
# ---------------------------------------------------------------------------
AUTO_EDA_QUERY = (
    "Perform a COMPLETE intelligent EDA on this dataset.\n\n"
    "Phase 1 — Understand the data:\n"
    "  • Call get_dataframe_info to see all columns, dtypes, and sample rows.\n"
    "  • Call generate_eda_summary to get statistics for all numeric columns.\n"
    "  • Call analyze_missing_values to check data quality.\n\n"
    "Phase 2 — Visual data quality:\n"
    "  • Call generate_missing_value_chart.\n\n"
    "Phase 3 — Distributions & outliers (generate for ALL numeric columns):\n"
    "  • Call generate_distribution_charts.\n"
    "  • Call generate_boxplot_charts.\n"
    "  • Call detect_outliers_report.\n\n"
    "Phase 4 — Correlation map:\n"
    "  • Call generate_eda_charts with plot_type='heatmap'.\n\n"
    "Phase 5 — YOU decide the best comparisons based on the data:\n"
    "  • For each categorical column with <= 15 unique values, call generate_countplot.\n"
    "  • Pick the 2-3 most meaningful (cat, num) pairs and call generate_barplot_comparison.\n"
    "  • Pick the 2-3 most meaningful (cat, num) pairs and call generate_boxplot_by_category.\n"
    "  • Pick the most correlated or interesting (x, y) numeric pair and call "
    "generate_scatter_comparison — optionally add the most relevant categorical column as hue_col.\n\n"
    "After ALL charts are generated, write 3 bullet points naming the top patterns you found."
)



# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def create_agent():
    """
    Builds and returns the LangGraph ReAct agent.

    Components wired here:
      - LLM         : ChatOpenAI (gpt-4o) from llm.py
      - Tools       : All 6 EDA tools from tools.py (Tool Registry)
      - Agent type  : ReAct via LangGraph's create_react_agent
    """
    llm = get_llm()

    # create_react_agent is the modern LangGraph way to build
    # an OpenAI function-calling ReAct agent with a tool registry
    agent = create_react_agent(
        model=llm,
        tools=EDA_TOOLS,           # Tool Registry
        prompt=SYSTEM_PROMPT,      # System-level instructions
    )
    return agent


# ---------------------------------------------------------------------------
# Main entry point — called by Streamlit frontend (next step)
# ---------------------------------------------------------------------------
def run_analysis(query: str, csv_path: str, chat_history: list = None) -> dict:
    """
    Full pipeline:
      1. Load CSV into PyTorch tensor (via tools.py)
      2. Run the ReAct agent loop with the user's query
      3. Return a structured result dict for the Report Generator

    Args:
        query        : Natural language question from the user
        csv_path     : Absolute path to the uploaded CSV file
        chat_history : Optional prior conversation messages (list of dicts)

    Returns:
        {
          "answer"      : str  — LLM's final natural-language answer
          "chart_paths" : list — Paths of any chart images generated
          "steps"       : list — Tool calls made during the agent loop
          "success"     : bool
          "error"       : str | None
        }
    """
    result = {
        "answer": "",
        "chart_paths": [],
        "steps": [],
        "success": False,
        "error": None,
    }

    try:
        # Step 1: Load CSV into the PyTorch tensor store in tools.py
        load_status = load_csv_into_tensor(csv_path)
        print(f"[Agent] Data load: {load_status}")

        # Step 2: Build the agent
        agent = create_agent()

        # Step 3: Build the message list for the agent
        messages = [HumanMessage(content=query)]

        # Step 4: Run the ReAct loop (agent streams through thoughts + tool calls)
        print(f"\n[Agent] Running query: {query}")
        print("-" * 50)

        final_response = agent.invoke({"messages": messages})

        # Step 5: Extract the final AI answer (last message in the thread)
        all_messages = final_response.get("messages", [])
        
        # Collect tool call steps for transparency
        for msg in all_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    step_info = f"Tool called: {tc['name']} | Input: {tc['args']}"
                    result["steps"].append(step_info)
                    print(f"  [Tool] {step_info}")

        # The last message is always the AI's final answer
        final_msg = all_messages[-1]
        result["answer"] = final_msg.content
        result["success"] = True

        # Step 6: Collect chart paths generated during the run
        charts_dir = os.path.join(os.path.dirname(__file__), "charts")
        if os.path.isdir(charts_dir):
            result["chart_paths"] = sorted(glob.glob(os.path.join(charts_dir, "*.png")))

    except Exception as e:
        result["error"] = str(e)
        result["answer"] = f"Agent encountered an error: {e}"
        print(f"[Agent ERROR] {e}")

    return result


# ---------------------------------------------------------------------------
# Quick self-test (run: python agent.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    # Create a sample CSV
    csv_content = (
        "age,salary,score,years_exp\n"
        "25,50000,88.5,2\n"
        "30,,92.0,5\n"
        "22,45000,,1\n"
        "35,60000,75.0,8\n"
        "28,52000,81.0,3\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        tmp_csv = f.name

    print("=" * 60)
    print("TEST 1: Dataset overview + missing values")
    print("=" * 60)
    out = run_analysis(
        query="Give me a complete overview of this dataset, check for missing values, and generate a bar chart.",
        csv_path=tmp_csv,
    )
    print("\n--- FINAL ANSWER ---")
    print(out["answer"])
    print("\n--- Steps taken ---")
    for s in out["steps"]:
        print(" •", s)
    print("Charts:", out["chart_paths"])

    print("\n" + "=" * 60)
    print("TEST 2: Specific column stats")
    print("=" * 60)
    out2 = run_analysis(
        query="What are the min, max, and mean of the salary column?",
        csv_path=tmp_csv,
    )
    print("\n--- FINAL ANSWER ---")
    print(out2["answer"])

    os.unlink(tmp_csv)
