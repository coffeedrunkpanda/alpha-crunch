# src/alpha_crunch/vector_db/chroma_modal_provider.py

import os
import httpx
from http import HTTPStatus

from dotenv import load_dotenv
from typing import Any
from langchain_core.documents import Document

from alpha_crunch.vector_db.contracts import VectorSearch

load_dotenv()

MODAL_KEY = os.getenv("MODAL_KEY")
MODAL_SECRET = os.getenv("MODAL_SECRET")
VECTOR_DB_URL = os.getenv("ALPHA_CRUNCH_VECTOR_DB_URL")

class ChromaModalProvider(VectorSearch):
    def __init__(self,
                 modal_key: str | None = MODAL_KEY,
                 modal_secret: str | None = MODAL_SECRET,
                 vector_db_url: str | None = VECTOR_DB_URL,
                 timeout: float = 120.0):

        assert modal_key is not None, "MODAL_KEY must be set at .env file."
        assert modal_secret is not None, "MODAL_SECRET must be set at .env file."
        assert vector_db_url is not None, "VECTOR_DB_URL must be set at .env file."
        
        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Modal-Key": modal_key,
            "Modal-Secret": modal_secret,
        }
        
        self._vector_db_url = vector_db_url
        self._timeout = timeout
        self.ready = False
        self.health = False

        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(
                f"{self._vector_db_url}/health",
                headers=self._headers,
            )
 
            self.health = True if response.status_code == HTTPStatus.OK else False
            response.raise_for_status()
            print(response.json())
            
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(
                f"{self._vector_db_url}/ready",
                headers=self._headers,
            )

            self.ready = True if response.status_code == HTTPStatus.OK else False
            response.raise_for_status()
            print(response.json())


    def search(self, query: str, k: int = 3, filter: dict[str, Any] | None = None) -> list[Document]:

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._vector_db_url}/query",
                headers=self._headers,
                json={"query": query, "k": k, "filter": filter},
            )

            response.raise_for_status()
            print (response.json())
            return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in response.json()]
