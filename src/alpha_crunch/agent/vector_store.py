import torch
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from alpha_crunch.agent.config import CHROMA_PATH

@lru_cache(maxsize=1) # Only load once (singleton)
def get_chroma_retriever(k_text_chunks: int = 3, embedding_model:str = "all-mpnet-base-v2"):
    """Initializes the ChromaDB connection and returns a retriever object."""

    # 1. Should use the same embedding model used to create the Chroma DB
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Connect to the existing Chroma Database
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # Convert vectorstore into a LangChain retriever interface
    return vectorstore.as_retriever(search_kwargs={"k": k_text_chunks})
