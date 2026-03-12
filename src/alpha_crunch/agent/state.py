from pydantic import BaseModel, Field
from typing import List, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(BaseModel):
    intent: str | None = None                  # "rag" | "analyst"
    retrieved_context: str | None = None       # RAG output — None until tool nodes run
    tool_output: str | None = None             # yfinance/sentiment — None until tool nodes run
    messages: Annotated[
        List[BaseMessage],add_messages
        ] = Field(default_factory=list)       # Conversation history (future multi-turn) using a reducer pattern