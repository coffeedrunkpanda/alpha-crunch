# src/alpha_crunch/vector_db/pinecone_provider.py

from typing import Any
from langchain_core.documents import Document
from alpha_crunch.vector_db.contracts import VectorSearch

class PineconeProvider(VectorSearch):

    def __init__(self):
        self.pinecone_client = None

    def search(self, query: str, k: int = 3, filter: dict [str, Any] | None = None) -> list[Document]:
        raise NotImplementedError("Pinecone provider is not implemented yet.")
