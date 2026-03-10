import os
import glob
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import tools
from tools import load_csv
from llm import get_llm

def run_analysis(csv_path: str) -> dict:
    """
    Full pipeline using pure LangChain LCEL (Directed Acyclic Graph):
      1. Load data
      2. Get context (dataframe info + correlation matrix)
      3. Create LangChain prompt expecting JSON
      4. Call LLM sequentially
      5. Parse output and generate charts
    """
    result = {
        "success": False,
        "answer": "",
        "steps": [],
        "chart_paths": [],
    }

    try:

        load_status = load_csv(csv_path)
        print(f"[Agent] Data load: {load_status}")
        df = tools._df

        df_info = tools.get_dataframe_info(df)
        corr_matrix = tools.get_correlation_matrix(df)
        result["steps"].append("Computed Dataframe Info and Correlation Matrix")

        template = """
You are a senior AI Data Scientist. Your objective is to perform a visual EDA on the uploaded dataset.
Instead of bombarding the user with all possible graphs, YOUR JOB IS TO FIND THE SINGLE BEST, MOST IMPORTANT PATTERN for each category.

DATAFRAME INFO:
{df_info}

CORRELATION MATRIX for NUMERICAL COLUMNS:
{corr_matrix}

Pick EXACTLY ONE graph for each category that reveals the most important story in this dataset.
Provide your plan strictly in the following JSON format:
{{
  "best_univariate_chart": {{"type": "histogram", "column": "col_name"}},
  "best_bivariate_chart": {{"type": "scatterplot", "x": "col1", "y": "col2"}},
  "best_multivariate_chart": {{"type": "boxplot", "x": "cat_col", "y": "num_col"}},
  "summary": "3 brief bullet points about why you chose these specific relationships."
}}
NOTE: For categorical variables, use type "countplot". For continuous, use "histogram".
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["df_info", "corr_matrix"]
        )

        llm = get_llm()
        llm_json = llm.bind(response_format={"type": "json_object"})
        parser = JsonOutputParser()

        chain = prompt | llm_json | parser

        print("\n[Agent] Running sequential LCEL chain for top 3 charts...")
        output = chain.invoke({"df_info": df_info, "corr_matrix": corr_matrix})

        result["steps"].append("LLM decided on the 3 best analysis charts")

        tools.plot_correlation_heatmap(df)
        result["steps"].append("Plotted correlation heatmap")

        uni = output.get("best_univariate_chart", {})
        if uni and uni.get("column") in df.columns:
            if uni.get("type") == "countplot":
                tools.plot_countplot(df, uni["column"])
            else:
                tools.plot_histogram(df, uni["column"])

        bi = output.get("best_bivariate_chart", {})
        if bi and bi.get("x") in df.columns and bi.get("y") in df.columns:
            tools.plot_scatterplot(df, bi["x"], bi["y"])

        multi = output.get("best_multivariate_chart", {})
        if multi and multi.get("x") in df.columns and multi.get("y") in df.columns:
            tools.plot_boxplot(df, multi["x"], multi["y"])
        
        result["steps"].append("Plotted exclusively the top Univariate, Bivariate, and Multivariate patterns")



        result["answer"] = output.get("summary", "Done running charts.")
        result["success"] = True

        charts_dir = os.path.join(os.path.dirname(__file__), "charts")
        if os.path.isdir(charts_dir):
            result["chart_paths"] = sorted(glob.glob(os.path.join(charts_dir, "*.png")))

    except Exception as e:
        import traceback
        result["answer"] = f"Runtime Error: {str(e)}\n\n{traceback.format_exc()}"
        result["success"] = False

    return result
