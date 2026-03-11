from alpha_crunch.agent.graph import alphaCrunch_agent
from alpha_crunch.agent.state import AgentState
from alpha_crunch.agent.rag_node import rag_node

def run(query: str):
    # This initial state triggers the execution
    initial_state = AgentState(query=query)

    print(f"Submitting query: {query}")
    # .invoke() is what actually runs the nodes and triggers the prints
    result = alphaCrunch_agent.invoke(initial_state)
    
    return result


if __name__ == "__main__":

    # result = run("What is an SEC 10-K filing?")
    # result = run("What are Apple's main supply chain risks?")
    # result = run("Did JPM beat earnings?")
    # result = run("What was Apple's total revenue in Q3 2023?")
    result = run("What are costco's main supply chain risk factors?")

    # Print the final result
    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])


