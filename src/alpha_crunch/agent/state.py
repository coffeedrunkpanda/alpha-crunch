from typing import TypedDict, Optional, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    query: str                          # User's question — set once, never changed
    intent: Optional[str]               # "rag" | "price" | "sentiment" | "direct"
    retrieved_context: Optional[str]    # RAG output — None until rag_node runs
    tool_output: Optional[str]          # yfinance/sentiment — None until tool nodes run
    final_answer: Optional[str]         # Finance LLM's response
    messages: List[BaseMessage]         # Conversation history (future multi-turn)