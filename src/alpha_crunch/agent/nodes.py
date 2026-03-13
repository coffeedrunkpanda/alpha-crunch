# alpha-crunch/src/alpha_crunch/agent/nodes.py
from langchain_core.messages import AIMessage, HumanMessage

from alpha_crunch.agent.llm_client import ask_finance_llm, classify_intent
from alpha_crunch.agent.state import AgentState
from alpha_crunch.agent.prompts import (SYSTEM_PROMPT,
                                        rag_user_prompt,
                                        analyst_user_prompt,
                                        INTENT_SYSTEM_PROMPT)

from alpha_crunch.agent.tools import get_dataset_help

# Adapter Design Pattern
def _format_chat_messages(state: AgentState):
# injects the temporary context (System Prompt + RAG chunks), and outputs the exact JSON structure Mistral 7B expects.

    formated_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    current_query = state.messages[-1].content

    if state.messages and len(state.messages) > 1:
        for msg in state.messages[: -1]:
            # Check LangChain message type to assign the correct role string
            role = "user" if msg.type == "human" else "assistant"
            formated_messages.append({
                "role": role, 
                "content": str(msg.content)
            }) 

    if state.retrieved_context and state.intent == "rag":
        latest_content = rag_user_prompt.format(
            context = state.retrieved_context,
            question = current_query)

    else:
        latest_content = analyst_user_prompt.format(question = current_query)

    formated_messages.append({"role": "user", "content": latest_content})
    return formated_messages

def llm_node (state: AgentState) -> dict:
    print("\n--- 🚨 DEBUG STATE (LLM NODE)---")
    print(f"Number of messages: {len(state.messages)}")
    if state.messages:
        print(f"Type of last message: {type(state.messages[-1])}")
        print(f"Content of last message: {state.messages[-1].content}")
    else:
        print("WARNING: state.messages is EMPTY!")
    print("----------------------\n")

    formated_messages = _format_chat_messages(state)
    print("----------------------\n")
    print(formated_messages)
    print("----------------------\n")
    final_answer = ask_finance_llm(messages= formated_messages, temperature= 0.4)

    return {"messages": [AIMessage(content=final_answer)]}
    
def intent_node(state: AgentState) -> dict:
    """
    Determines if the query requires RAG (document lookup) or can be 
    answered directly by the analyst's general knowledge.
    """
    current_query = state.messages[-1].content

    # Keyword fallback (fast, no LLM)
    help_keywords = ["help", "cutoff", "info"]
    if any(word in current_query for word in help_keywords):
        print("--- FAST ROUTE: Help keywords detected ---")
        return {"intent": "help"}


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
        # Example 4
        {"role": "user", "content": "What companies do you have data for? Help!"},
        {"role": "assistant", "content": "help"},
        # The actual query we need to classify
        {"role": "user", "content": current_query},       
        ]

    # Call router function
    intent = classify_intent(messages)
    print(f"--- ROUTING: Intent classified as '{intent}' ---")
    
    if intent == "analyst":
        return {"intent": intent, "retrieved_context": None}
    
    elif intent == "rag":
        return {"intent": intent}
    
    else:  # rag or fallback
        return {"intent": "help"}

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
    
    elif intent == "help":
        print("--- ROUTING TO: help_node ---")
        return "help_node"
    
    # Fallback just in case
    return "help"

def help_node(state: AgentState) -> dict:
    """Direct tool invocation."""
    # from alpha_crunch.agent.tools import get_dataset_help
    
    # CORRECT: Use .invoke() (takes dict or input matching tool args)
    tool_result = get_dataset_help.invoke({})  # {} for no-input tool
    
    # ALTERNATIVE: .run("") if simple str input
    # tool_result = get_dataset_help.run("")
    
    print("--- HELP TOOL INVOKED ---")
    return {
        "messages": [AIMessage(content=str(tool_result))],  # str() for safety
        "intent": None
    }