import gradio as gr
from gradio import themes

from langchain_core.messages import HumanMessage
from alpha_crunch.agent.state import AgentState
from alpha_crunch.agent.graph import alphaCrunch_agent

# We define a constant thread ID for our local testing session.
# In a real deployed app with multiple users, you would generate a unique UUID per user session.
SESSION_ID = "gradio_local_test_1"

def chat_interface(user_message: str, history: list):
    """
    Generator function that connects Gradio to LangGraph.
    It yields `gr.ChatMessage` objects to stream intermediate thoughts,
    and finally yields the actual LLM string.
    """
    
    # 1. Initialize the LangGraph State
    # We pass the user's input as a HumanMessage. LangGraph's checkpointer (MemorySaver)
    # will automatically pull the previous messages for this SESSION_ID and append this new one.

    inputs = AgentState(messages=[HumanMessage(content=user_message)])
    config = {"configurable": {"thread_id": SESSION_ID}}

    # 2. Yield an initial "Thought" message to show the user we are starting
    # The 'metadata' dictionary creates the nice collapsible accordion in Gradio
    thought_msg = gr.ChatMessage(
        role="assistant", 
        content="Analyzing query...", 
        metadata={"title": "Agent Pipeline Process", "status": "pending"}
    )
    yield thought_msg

    # 3. Stream the Graph execution
    # stream_mode="updates" yields a dictionary every time a node finishes executing.
    # This is perfect for XAI (Explainable AI) to show the user our routing logic.
    accumulated_thoughts = ""
    for output in alphaCrunch_agent.stream(inputs, config=config, stream_mode="updates"):
        
        # Check if the intent node just ran
        if "intent_node" in output:
            intent = output["intent_node"].get("intent", "unknown")
            accumulated_thoughts += f"➡️ **Routing:** Query classified as `{intent}`.\n"
            thought_msg.content = accumulated_thoughts
            yield thought_msg
            
        # Check if the RAG node just ran
        elif "rag_node" in output:
            context = output["rag_node"].get("retrieved_context", "")
            accumulated_thoughts += "➡️ **Retrieval:** Successfully queried ChromaDB for SEC 10-K filings.\n"
            
            # Optional: You could even display snippets of the context in the thought bubble!
            # accumulated_thoughts += f"```text\n{context[:200]}...\n```\n"
            
            thought_msg.content = accumulated_thoughts
            yield thought_msg

    # 4. Finalize the Thought Process
    thought_msg.metadata["status"] = "done"
    yield thought_msg

    # 5. Retrieve the Final LLM Answer
    # We query the checkpointer for the final state of the graph
    final_state = alphaCrunch_agent.get_state(config).values
    
    # The LLM's answer is the last message in the list
    final_text = final_state["messages"][-1].content
    
    # 6. Yield the final text as a standard message
    yield gr.ChatMessage(role="assistant", content=final_text)


# ==========================================
# Gradio UI Configuration (Gradio 6.0+)
# ==========================================
# 1. Remove theme from here
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 AlphaCrunch: Automated Investment Analyst")
    gr.Markdown("Ask financial questions. RAG is powered by local ChromaDB and S&P 500 SEC 10-K filings.")
    
    # 2. Remove type="messages"
    chatbot = gr.ChatInterface(
        fn=chat_interface,
        textbox=gr.Textbox(placeholder="Ask about Costco's risk factors or Apple's revenue..."),
    )

if __name__ == "__main__":
    print("Starting Gradio Server...")
    demo.launch(theme=themes.Soft(), share=False)

