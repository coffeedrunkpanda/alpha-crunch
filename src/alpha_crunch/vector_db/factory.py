# src/alpha_crunch/vector_db/factory.py

from functools import lru_cache
from alpha_crunch.vector_db.types import VectorDBProvider
from alpha_crunch.vector_db.contracts import VectorSearch
from alpha_crunch.vector_db.chroma_provider import ChromaProvider
from alpha_crunch.vector_db.chroma_modal_provider import ChromaModalProvider
from alpha_crunch.vector_db.pinecone_provider import PineconeProvider
from alpha_crunch.agent.config import VECTOR_DB_PROVIDER

@lru_cache(maxsize=1)
def get_vector_db_provider(vector_db_provider: VectorDBProvider = VECTOR_DB_PROVIDER) -> VectorSearch:
    
    match vector_db_provider:
        case VectorDBProvider.CHROMA:
            return ChromaProvider()
        case VectorDBProvider.CHROMA_MODAL:
            return ChromaModalProvider()
        case VectorDBProvider.PINECONE:
            # return PineconeProvider()
            raise NotImplementedError("Pinecone provider is not implemented yet. Set VECTOR_DB_PROVIDER as (chroma, chroma-modal, pinecone) in the config.")
