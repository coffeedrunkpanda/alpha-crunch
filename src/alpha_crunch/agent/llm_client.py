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


def get_intent_prompt(question: str) -> str:
    # We use a strict few-shot prompt format
    return f"""You are a strict routing assistant. Your job is to classify the user's question into one of two categories: "rag" or "analyst".

Categories:
- "rag": Use this when the user asks for specific numbers, earnings, SEC data, or historical facts about a specific company (e.g., Apple's revenue, Tesla's Q3 filings).
- "analyst": Use this when the user asks for general definitions, financial concepts, or explanations that do not require looking up a specific company's document (e.g., what is inflation, what is a 10-K).

Output EXACTLY one word: either "rag" or "analyst". Do not output anything else.

Examples:
Question: What was Microsoft's revenue in 2023?
Category: rag

Question: What is a P/E ratio?
Category: analyst

Question: Did Amazon beat earnings expectations last quarter?
Category: rag

Question: {question}
Category:"""


def classify_intent(question: str) -> str:
    
    # Prompt for routing
    prompt = get_intent_prompt(question)

    payload = {
        "message": prompt,
        "max_new_tokens": 15,
        "temperature": 0.01,
    }

    response = requests.post(
        f"{MODAL_ENDPOINT_URL}/generate",
        headers=headers,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    response = response.json()["response"]
    clean_intent = response.lower().strip().strip('."\'\n')

    if "analyst" in clean_intent:
        return "analyst"
    
    # otherwise, fallback to rag
    return "rag"