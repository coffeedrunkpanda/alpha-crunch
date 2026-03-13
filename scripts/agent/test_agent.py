from alpha_crunch.agent.graph import alphaCrunch_agent
from alpha_crunch.agent.state import AgentState
from langchain_core.messages import HumanMessage

def run(query: str):
    # This initial state triggers the execution
    initial_state = AgentState(messages=[HumanMessage(content=query)])

    print(f"Submitting query: {query}")

    # .invoke() is what actually runs the nodes and triggers the prints
    final_state = alphaCrunch_agent.invoke(initial_state, config={"configurable": {"thread_id": "session_2"}})

    return final_state


if __name__ == "__main__":

    # result = run("What is an SEC 10-K filing?")
    # result = run("What are Apple's main supply chain risks?")
    # result = run("Did JPM beat earnings?")
    # result = run("What was Apple's total revenue in Q3 2023?")
    result = run("What are costco's main supply chain risk factors?")

    # Print the final result
    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])


