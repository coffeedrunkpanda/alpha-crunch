from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import BaseMessage

class AgentState(BaseModel):
    query: str                                                # User's question — set once, never changed
    intent: str | None = None                                 # "rag" | "analyst"
    retrieved_context: str | None = None                      # RAG output — None until tool nodes run
    tool_output: str | None = None                            # yfinance/sentiment — None until tool nodes run
    final_answer: str | None = None                           # Finance LLM's response
    messages: List[BaseMessage] = Field(default_factory=list) # Conversation history (future multi-turn)