import os
import torch
from pathlib import Path
from functools import lru_cache

# Modern LangChain imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from alpha_crunch.agent.state import AgentState
from alpha_crunch.agent.config import CHROMA_PATH

# TODO: check if this is really loading only once
@lru_cache(maxsize=1) # Only load once (singleton)
def get_chroma_retriever(chroma_path:str, k_text_chunks: int = 3, embedding_model:str = "all-mpnet-base-v2"):
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
        persist_directory=chroma_path,
        embedding_function=embeddings
    )
    
    # Convert vectorstore into a LangChain retriever interface
    return vectorstore.as_retriever(search_kwargs={"k": k_text_chunks})

def rag_node(state: AgentState) -> dict:
    """
    The LangGraph node function for RAG.
    It takes the current state, uses the query to search ChromaDB,
    and updates the state with the formatted retrieved context.
    """

    query = state.query
    print(f"--- RAG NODE: Searching for '{query}' ---")
    
    # Get the retriever and fetch documents
    retriever = get_chroma_retriever(CHROMA_PATH)
    docs = retriever.invoke(query)
    
    if not docs:
        print("--- RAG NODE: No documents found ---")
        return {"retrieved_context": "No specific context was found in the database."}
    
    # Format the retrieved documents nicely so the LLM can easily read them
    formatted_context = ""
    for i, doc in enumerate(docs):
        # Extract the metadata we attached during ingestion
        company = doc.metadata.get('company', 'Unknown')
        year = doc.metadata.get('date', 'Unknown')[:4] # Grab just the YYYY
        item_type = doc.metadata.get('item_type', 'Unknown')
        
        # Format: [APPLE - 2021 (item_1A)] The text goes here...
        formatted_context += f"[{company} - {year} ({item_type})] {doc.page_content}\n\n"
    
    print(f"--- RAG NODE: Retrieved {len(docs)} chunks successfully ---")
    print(f"--- RAG NODE: Docs:  ---")

    print(formatted_context)

    # In LangGraph, returning a dictionary updates the state
    return {"retrieved_context": formatted_context}
