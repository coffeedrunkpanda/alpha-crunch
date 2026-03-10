from alpha_crunch.agent.graph import alphaCrunch_agent
from alpha_crunch.agent.state import AgentState
from alpha_crunch.agent.rag_node import rag_node

def run(query: str):
    # This initial state triggers the execution
    initial_state = {
        "query": query,
        "intent": None,
        "retrieved_context": None,
        "tool_output": None,
        "final_answer": None,
        "messages": []
    }

    print(f"Submitting query: {query}")
    # .invoke() is what actually runs the nodes and triggers the prints
    result = alphaCrunch_agent.invoke(initial_state)
    
    return result

def test_rag(query: str):

    # Mocking the LangGraph state
    initial_state = {
        "query": query,
        "intent": "rag",
        "retrieved_context": "",
        "tool_output": None,
        "final_answer": None,
        "messages": []
    }
    # Run the node
    new_state_data = rag_node(initial_state)
    
    print("\n=== THE FORMATTED CONTEXT SENT TO THE LLM ===")
    print(new_state_data["retrieved_context"])

if __name__ == "__main__":
    # result = run("What is an SEC 10-K filing?")
    # result = run("What was Apple's total revenue in Q3 2023?")
    # print(result["final_answer"])

    result = test_rag("What are costco's main supply chain risk factors?")
    print("\n--- Final State ---")
    # print(result)



    