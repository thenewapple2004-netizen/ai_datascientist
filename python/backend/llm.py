"""
backend/llm.py — LLM client configuration
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_BASE, ".env"))


def get_llm(temperature: float = 0, model: str = "gpt-4o-mini") -> ChatOpenAI:
    """Returns a configured ChatOpenAI instance."""
    return ChatOpenAI(model=model, temperature=temperature)
