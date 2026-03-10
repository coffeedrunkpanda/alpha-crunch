from alpha_crunch.agent.llm_client import ask_finance_llm, classify_intent
from alpha_crunch.agent.state import AgentState

def llm_node (state: AgentState) -> dict:

    answer = ask_finance_llm(
        question=state["query"],
        context=state["retrieved_context"]
    )

    return {"final_answer": answer}
    
def intent_node(state: AgentState) -> dict:
    """
    Determines if the query requires RAG (document lookup) or can be 
    answered directly by the analyst's general knowledge.
    """

    print("--- ROUTING: Classifying Intent ---")
    query = state["query"]
    
    # Call router function
    intent = classify_intent(query)
    print(f"--- ROUTING: Intent classified as '{intent}' ---")
    
    # Return the state update
    return {"intent": intent}


def route_by_intent(state: AgentState) -> str:
    """
    Reads the intent from state and returns the name of the next node.
    """
    intent = state["intent"]
    
    if intent == "rag":
        print("--- ROUTING TO: rag_node (currently mocked to llm_node) ---")
        # return "rag_node"  <-- We will uncomment this in Phase 4
        return "llm_node"
    
    elif intent == "analyst":
        print("--- ROUTING TO: llm_node ---")
        return "llm_node"
        
    # Fallback just in case
    return "llm_node"