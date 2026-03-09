import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools import EDA_TOOLS

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def get_llm(temperature: float = 0, model: str = "gpt-4o") -> ChatOpenAI:
    """
    Returns a bare ChatOpenAI instance.
    Use this when you don't need tool-calling (e.g. simple Q&A).
    """
    return ChatOpenAI(model=model, temperature=temperature)


def get_llm_with_tools(temperature: float = 0, model: str = "gpt-4o") -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance with the EDA tools pre-bound.
    The agent loop uses this so the LLM can call tools autonomously
    via OpenAI Function-Calling / ReAct.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    return llm.bind_tools(EDA_TOOLS)


if __name__ == "__main__":
    # --- Test 1: bare LLM ---
    llm = get_llm()
    print("=== Bare LLM ===")
    response = llm.invoke("Hello, are you ready for data analysis?")
    print("Response:", response.content)

    # --- Test 2: LLM with tools bound ---
    print("\n=== LLM with EDA Tools bound ===")
    llm_tools = get_llm_with_tools()
    print("Tools registered:", [t.name for t in EDA_TOOLS])
    print("LLM is ready for the Agent Loop.")
