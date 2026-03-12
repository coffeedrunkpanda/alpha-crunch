from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from alpha_crunch.agent.state import AgentState
from alpha_crunch.agent.nodes import llm_node, intent_node, route_by_intent
from alpha_crunch.agent.rag_node import rag_node

# START -> intent_node -> (conditional route) -> rag_node / llm_node -> END
def build_graph():
    
    # Persist memmory after reaching END: handles sesion's memory natively
    # TODO: Change this to a persistent checkpoint saving the memory in a db PostgresSaver or MongoDBSaver
    checkpointer = MemorySaver()

    # Create graph
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("llm_node", llm_node)
    graph.add_node("intent_node", intent_node)
    graph.add_node("rag_node", rag_node)

    # Entry point for the agent
    graph.add_edge(START, "intent_node")

    # Intent node
    graph.add_conditional_edges(source="intent_node", # starting node
                                path=route_by_intent, # next node
                                path_map={ "llm_node": "llm_node",  # If router says "llm_node", go to "llm_node"
                                           "rag_node": "rag_node",
                                          })
    
    # Exit edge
    graph.add_edge("rag_node", "llm_node")
    graph.add_edge("llm_node", END)

    print("🚨 IN BUILD GRAPH")
    # Compile/validate the graph
    return graph.compile(checkpointer=checkpointer)

alphaCrunch_agent = build_graph()
