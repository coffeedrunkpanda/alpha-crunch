from alpha_crunch.agent.llm_client import ask_finance_llm, classify_intent
from alpha_crunch.agent.state import AgentState
from alpha_crunch.agent.prompts import (SYSTEM_PROMPT,
                                        rag_user_prompt,
                                        analyst_user_prompt,
                                        INTENT_SYSTEM_PROMPT)

def llm_node (state: AgentState) -> dict:

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if state.messages and len(state.messages) > 0:
        for msg in state.messages:
            # Check LangChain message type to assign the correct role string
            role = "user" if msg.type == "human" else "assistant"
            messages.append({
                "role": role, 
                "content": str(msg.content)
            }) 

    if state.retrieved_context:
        user_prompt = {"role": "user",
                       "content": rag_user_prompt.format(context = state.retrieved_context,
                                                         question = state.query)}

    else:
        user_prompt = {"role": "user",
                       "content": analyst_user_prompt.format(question = state.query)}

    messages.append(user_prompt)

    final_answer = ask_finance_llm(messages= messages)

    return {"final_answer": final_answer}
    
def intent_node(state: AgentState) -> dict:
    """
    Determines if the query requires RAG (document lookup) or can be 
    answered directly by the analyst's general knowledge.
    """

    # Build message list using a system prompt and examples to simulate a conversation history
    messages = [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        # Example 1
        {"role": "user", "content": "What was Microsoft's revenue in 2023?"},
        {"role": "assistant", "content": "rag"},
        # Example 2
        {"role": "user", "content": "What is a P/E ratio?"},
        {"role": "assistant", "content": "analyst"},
        # Example 3
        {"role": "user", "content": "Did Amazon beat earnings expectations last quarter?"},
        {"role": "assistant", "content": "rag"},
        # The actual query we need to classify
        {"role": "user", "content": state.query}       
        ]

    # Call router function
    intent = classify_intent(messages)
    print(f"--- ROUTING: Intent classified as '{intent}' ---")
    
    # Return the state update
    return {"intent": intent}


def route_by_intent(state: AgentState) -> str:
    """
    Reads the intent from state and returns the name of the next node.
    """
    intent = state.intent
    
    if intent == "rag":
        print("--- ROUTING TO: rag_node ---")
        return "rag_node"
    
    elif intent == "analyst":
        print("--- ROUTING TO: llm_node ---")
        return "llm_node"
        
    # Fallback just in case
    return "llm_node"