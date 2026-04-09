# src/alpha_crunch/vector_db/contracts.py

# protocol/interface for vector search
# shared request/response models

from typing import Protocol, Any
from langchain_core.documents import Document

class VectorSearch(Protocol):
    def search(self, query: str, k: int = 3, filter: dict [str, Any] | None = None) -> list[Document]:
        pass
