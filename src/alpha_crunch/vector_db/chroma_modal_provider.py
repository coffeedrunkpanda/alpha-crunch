# src/alpha_crunch/vector_db/chroma_modal_provider.py

import os
import httpx

from dotenv import load_dotenv
from typing import Any
from langchain_core.documents import Document

from alpha_crunch.vector_db.contracts import VectorSearch

load_dotenv()

MODAL_KEY = os.getenv("MODAL_KEY")
MODAL_SECRET = os.getenv("MODAL_SECRET")
VECTOR_DB_URL = os.getenv("ALPHA_CRUNCH_VECTOR_DB_URL")

class ChromaModalProvider(VectorSearch):
    def __init__(self):

        assert MODAL_KEY is not None, "MODAL_KEY must be set at .env file."
        assert MODAL_SECRET is not None, "MODAL_SECRET must be set at .env file."
        assert VECTOR_DB_URL is not None, "VECTOR_DB_URL must be set at .env file."
        
        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Modal-Key": MODAL_KEY,
            "Modal-Secret": MODAL_SECRET,
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{VECTOR_DB_URL}/health",
                headers=self._headers,
            )
            response.raise_for_status()
            print (response.json())
            print("✅ ChromaModalProvider initialized")

        with httpx.Client(timeout=120.0) as client:
            response = client.get(
                f"{VECTOR_DB_URL}/ready",
                headers=self._headers,
            )
            response.raise_for_status()
            print (response.json())
            self.ready = True if response.json()["ready"] else False
            
            if not self.ready:
                print("ChromaModalProvider is not ready")
            

    def search(self, query: str, k: int = 3, filter: dict [str, Any] | None = None) -> list[Document]:

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{VECTOR_DB_URL}/query",
                headers=self._headers,
                json={"query": query, "k": k, "filter": filter},
            )

            response.raise_for_status()
            print (response.json())
            return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in response.json()]
