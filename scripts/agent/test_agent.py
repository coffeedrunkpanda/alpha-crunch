from alpha_crunch.agent.graph import alphaCrunch_agent

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

if __name__ == "__main__":
    # result = run("What is an SEC 10-K filing?")
    result = run("What was Apple's total revenue in Q3 2023?")

    print("\n--- Final State ---")
    print(result["final_answer"])