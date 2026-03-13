# AI Data Scientist

An AI-powered Exploratory Data Analysis (EDA) and Feature Engineering dashboard built with Streamlit and LangGraph. Upload a CSV file and the agent autonomously analyzes your data, generates visualizations, cleans and engineers features, and delivers a downloadable processed dataset — with human-in-the-loop clarification at each stage.

---

## Features

- **Automated EDA** — correlation heatmaps, histograms, scatter plots, box plots, and count plots generated automatically
- **Intelligent Feature Engineering** — drop useless columns, impute missing values, binarize features, encode categoricals, and scale numerics
- **4-Phase Agentic Pipeline** — LangGraph DAG orchestrates analysis in structured phases with clear outputs
- **Human-in-the-Loop MCQ** — agent pauses to ask clarifying questions when data is ambiguous (e.g., identifying target variables)
- **Multi-Stage Downloads** — export cleaned CSV at any pipeline stage
- **Dark-themed UI** — responsive Streamlit interface with styled HTML components and chart preview grids

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent Orchestration | LangGraph |
| LLM Framework | LangChain |
| LLM Model | OpenAI GPT-4o-mini |
| Data Processing | pandas, numpy, scikit-learn |
| Visualization | matplotlib, seaborn |
| Language | Python 3.x |

---

## Project Structure

```
ai_datascientist/
├── python/
│   ├── app.py          # Main Streamlit web application (UI + session state)
│   ├── agent.py        # LangGraph agent definition and pipeline phases
│   ├── llm.py          # OpenAI ChatGPT client configuration
│   ├── tools.py        # 11 LLM-accessible data processing & visualization tools
│   ├── charts/         # Generated PNG charts (created at runtime)
│   └── .env            # Environment variables (not committed)
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd ai_datascientist
```

### 2. Create and activate a virtual environment

```bash
cd python
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit langchain langgraph langchain-openai openai pandas numpy scikit-learn matplotlib seaborn python-dotenv
```

### 4. Configure environment variables

Create `python/.env`:

```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Usage

```bash
cd python
streamlit run app.py
```

Open `http://localhost:8501` in your browser, then upload a CSV file (max 10 MB) via the sidebar to begin.

---

## Pipeline

The agent runs a 4-phase LangGraph workflow:

### Phase 1 — Read
- Reads dataset metadata (shape, column types, missing %, unique counts)
- Determines if clarification is needed and asks MCQ questions if so
- Output: `READY_TO_ANALYZE` or a clarification question set

### Phase 2 — EDA (Exploratory Data Analysis)
Generates 4 core charts in sequence:
1. Correlation heatmap
2. Univariate histogram or count plot
3. Bivariate scatter plot
4. Multivariate box plot

Output: `EDA_COMPLETE: <insight summary>`

### Phase 3 — Feature Engineering
Executes a mandatory 5-step pipeline in order:
1. **Drop useless columns** — removes ID columns, zero-variance columns, high-sparsity columns (>60% missing)
2. **Impute missing values** — mean/median for numeric, mode for categorical
3. **Binarize features** — creates binary indicators from numeric columns
4. **Encode categoricals** — label encoding for text columns
5. **Scale numerics** — StandardScaler normalization

Output: `FE_COMPLETE: <operations list>`

### Phase 4 — Final Analysis
- Generates 3 optimized charts on the cleaned, engineered dataset
- Highlights patterns revealed after feature engineering

Output: `FINAL_COMPLETE: <3 key insights>`

---

## Available Tools (11)

| Tool | Description |
|---|---|
| `tool_get_dataframe_info` | Dataset shape, dtypes, missing %, unique counts |
| `tool_plot_correlation_heatmap` | Seaborn correlation matrix heatmap |
| `tool_plot_histogram` | Univariate distribution with KDE and mean/median markers |
| `tool_plot_countplot` | Categorical value counts bar chart |
| `tool_plot_scatterplot` | Bivariate continuous relationship scatter plot |
| `tool_plot_boxplot` | Categorical vs numeric multivariate box plot |
| `tool_drop_useless_columns` | Bulk removal of specified columns |
| `tool_impute_missing_values` | JSON-configured imputation strategies per column |
| `tool_binarize_features` | Create binary indicator columns with custom thresholds |
| `tool_encode_categorical_features` | Label encoding for categorical columns |
| `tool_scale_numeric_features` | StandardScaler normalization |

---

## UI Overview

The interface has 3 tabs:

| Tab | Content |
|---|---|
| **Initial EDA** | Data preview table, MCQ clarification cards, EDA charts |
| **Feature Engineering** | Pre-FE statistics, operations log, post-FE comparison, cleaned heatmap |
| **Final Analysis** | Final charts, binary feature summary, cleaned dataset download |

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o-mini access | Yes |
