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

custom_css = """
/* 1. Loomy Dark Mesh Gradient */
div {
    background: transparent;
}

body, .gradio-container {
    background-color: #0b0914 !important; 
    background-image: 
        radial-gradient(circle at 80% 0%, rgba(79, 172, 254, 0.4) 0px, transparent 20%),
        radial-gradient(circle at 0% 0%, rgba(200, 100, 60, 0.3) 0px, transparent 30%), 
        radial-gradient(circle at 80% 100%, rgba(161, 140, 209, 0.3) 0px, transparent 50%),
        radial-gradient(circle at 0% 100%, rgba(200, 50, 100, 0.4) 0px, transparent 50%)    
        !important;
    color: #e2e8f0 !important; 
}

* {
    transition: 0.2s ease-in-out 0s;    
}

/* 2. Dark Glassmorphism for the Chat Window */
/* We target the custom class we will define below */
.my-chat-window {
    background: rgba(30, 41, 59, 0.3) !important; /* Solid slate gray */
    backdrop-filter: blur(28px) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important; 
    box-shadow: 0 15px 50px 0 rgba(0, 0, 0, 0.8) !important; 
}

.gr-group  {
    box-shadow: 0 15px 50px 0 rgba(0, 0, 0, 0.8);
    border-radius: 10px;
}

/* 3. The Input Box Container (Matching the Chat Window!) */
.my-input-container {
    border-radius: 10px;
    background: rgba(30, 41, 59, 0.3) !important; /* Solid slate gray */
    backdrop-filter: blur(28px) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Make sure the text area itself doesn't have its own weird background */
.my-input-container textarea {
    background: rgba(15, 15, 15, 0.8);
    backdrop-filter: blur(28px) !important;
}

.my-input-container textarea:disabled {
    opacity: 0;
}

.my-input-container textarea:focus {
    background: rgba(10, 10, 10, 0.8);
}

.gr-group:focus-within {
    outline: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0px 0px 25px 2px rgba(255, 255, 255, 0.2) !important; 
    border-radius: 10px;
}

/* 4. Headers: Anta Font */
h1, h2, h3, .form-label {
    font-family: 'Anta', sans-serif !important;
    color: #f8fafc !important;
    letter-spacing: 1px;
}
h1 {
    text-align: center;
    text-shadow: 0 0 15px rgba(79, 172, 254, 0.4); 
    font-weight: 400;
}

/* 5. Typewriter Chat Text */
.message-wrap, .chatbot, textarea, input {
    font-family: 'Montserrat', monospace !important;
    font-size: 15px !important;
    color: #cbd5e1 !important; 
}

/* Subtle dark tint for user bubble */
.message.user {
    background: rgba(77, 145, 227, 0.2) !important; 
    border: 1px solid rgba(255, 255, 255, 0.05) !important; /* Exact same border */
}

.message.bot {
    background: rgba(232, 79, 149, 0.2)  !important; 
    border: #6d28d9 !important; /* Exact same border */
}

"""

# Theme adjusted for the purple/slate aesthetic
custom_theme = themes.Soft(
    font=[themes.GoogleFont("Anta"), themes.GoogleFont("Montserrat"), "sans-serif"],
    primary_hue="violet",  
    neutral_hue="slate"
)

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 AlphaCrunch: Automated Investment Analyst")
    gr.Markdown("<div style='text-align: center;'>Ask financial questions. RAG is powered by local ChromaDB and S&P 500 SEC 10-K filings.</div>")
    chatbot = gr.ChatInterface(
        fn=chat_interface,
        chatbot=gr.Chatbot(elem_classes="my-chat-window", show_label=False),
        textbox=gr.Textbox(
            placeholder="Ask about Costco's risk factors or Apple's revenue...", 
            container=False, # Hides Gradios internal label container
            elem_classes="my-input-container"
        ),
    )

if __name__ == "__main__":
    print("Starting Gradio Server...")
    demo.launch(theme=custom_theme,
                css = custom_css,
                share=False)
