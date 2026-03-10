from langgraph.graph import StateGraph, END
from alpha_crunch.agent.state import AgentState
from alpha_crunch.agent.nodes import llm_node

def build_graph():
    
    # Create graph
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("llm_node", llm_node)


    # Entry point for the agent
    graph.set_entry_point("llm_node")

    # Exit edge
    graph.add_edge("llm_node", END)

    # Compile/validate the graph
    return graph.compile()

alphaCrunch_agent = build_graph()
