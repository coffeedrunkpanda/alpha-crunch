# src/alpha_crunch/vector_db/factory.py

# returns provider based on env/config (VECTOR_DB_PROVIDER)

from typing import Literal
from alpha_crunch.vector_db.contracts import VectorSearch
from alpha_crunch.vector_db.chroma_provider import ChromaProvider
from alpha_crunch.vector_db.pinecone_provider import PineconeProvider
from alpha_crunch.agent.config import VECTOR_DB_PROVIDER
from functools import lru_cache

@lru_cache(maxsize=1)
def get_vector_db_provider(vector_db_provider: Literal["chroma", "pinecone"] = VECTOR_DB_PROVIDER) -> VectorSearch:
    if vector_db_provider == "chroma":
        return ChromaProvider()
    elif vector_db_provider == "pinecone":
        # return PineconeProvider()
        raise NotImplementedError("Pinecone provider is not implemented yet. Set VECTOR_DB_PROVIDER=chroma in the config.")

    else:
        raise ValueError(f"Invalid vector database provider: {vector_db_provider}")