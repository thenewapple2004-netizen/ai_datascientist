import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def get_llm(temperature: float = 0, model: str = "gpt-4o-mini") -> ChatOpenAI:
    """Returns a bare ChatOpenAI instance."""
    return ChatOpenAI(model=model, temperature=temperature)
