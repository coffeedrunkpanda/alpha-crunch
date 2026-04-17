# src/alpha_crunch/vector_db/chroma_provider.py

import torch
from typing import Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from alpha_crunch.agent.config import CHROMA_PATH, CHROMA_EMBEDDING_MODEL
from alpha_crunch.vector_db.contracts import VectorSearch

class ChromaProvider(VectorSearch):
    def __init__(self, embedding_model: str = CHROMA_EMBEDDING_MODEL):

        self.embedding_model = embedding_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try: 
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            raise ValueError(f"Error initializing embeddings: {e}")

        try:
            self.chroma_client = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=self.embeddings
            )
        except Exception as e:
            raise ValueError(f"Error initializing Chroma client: {e}")

    def search(self, query: str, k: int = 3, filter: dict [str, Any] | None = None) -> list[Document]:

        if filter:
            retriever = self.chroma_client.as_retriever(search_kwargs={
                "k": k,
                "filter": filter
            })

        else:
            retriever = self.chroma_client.as_retriever(search_kwargs = {
                "k": k
            })

        return retriever.invoke(query)