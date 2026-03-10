import os
import json
import requests
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

# TODO: change the prompt template

def ask_finance_llm(question: str,
                    context: str | None = None) -> str:
    """
    Calls Finance LLM served on Modal.
    """

    prompt_template = (
        "Use the following context to answer the question.\n\n"
        "Context: {context}\n\n"
        "Question: {question}"
    )

    if context:
        content = prompt_template.format(context=context, question=question)
    else:
        content = question

    payload = {
        "message": content,
        "max_new_tokens": 256,
        "temperature": 0.2,
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
