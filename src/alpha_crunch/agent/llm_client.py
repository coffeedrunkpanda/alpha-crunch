import os
import json
import requests
from typing import List, Dict
from dotenv import load_dotenv

# env variables
load_dotenv()
MODAL_KEY = os.getenv("MODAL_KEY")
MODAL_SECRET = os.getenv("MODAL_SECRET")
MODAL_ENDPOINT_URL = os.getenv("ALPHA_CRUNCH_URL")


# Request headers
headers = {
    "Content-Type": "application/json",
    "Modal-Key": MODAL_KEY,
    "Modal-Secret": MODAL_SECRET,
}

def ask_finance_llm(messages: List[Dict[str,str]],
                    max_new_tokens:int = 512,
                    temperature:float = 0.2) -> str:
    """Client to call the Modal LLM endpoint."""

    payload = {
        "messages": messages,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }

    response = requests.post(
        f"{MODAL_ENDPOINT_URL}/generate",
        headers=headers,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    answer = response.json()["response"]
    return answer

def classify_intent(messages: List[Dict[str,str]]) -> str:

    answer= ask_finance_llm(messages=messages,
                            max_new_tokens=10,
                            temperature=0.01)
    
    clean_intent = answer.lower().strip().strip('."\'\n')

    if "analyst" in clean_intent:
        return "analyst"
    
    # otherwise, fallback to rag
    return "rag"