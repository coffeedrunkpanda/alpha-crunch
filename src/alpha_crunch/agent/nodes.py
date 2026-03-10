from alpha_crunch.agent.llm_client import ask_finance_llm
from alpha_crunch.agent.state import AgentState

def llm_node (state: AgentState) -> dict:

    answer = ask_finance_llm(
        question=state["query"],
        context=state["retrieved_context"]
    )

    return {"final_answer": answer}
    