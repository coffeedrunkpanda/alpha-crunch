import os
import re
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

import torch

# Modern LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def standardize_company_name(name: str) -> str:
    """Standardizes company names by making them uppercase and removing suffixes."""
    name = str(name).upper()
    name = re.sub(r'[.,]', '', name)
    name = re.sub(r'\b(INC|CORP|CORPORATION|LLC|LTD|COMPANY|CO)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def build_vector_database(chroma_path: str, max_rows: int = 500):
    """
    Pulls SEC 10-K data from Hugging Face, cleans the metadata, chunks the text, 
    and ingests it into a local ChromaDB instance.
    """

    if not os.path.isdir(chroma_path):
        os.makedirs(chroma_path, exist_ok=True)

    print(f"Loading dataset from Hugging Face (Max rows: {max_rows})...")
    dataset = load_dataset("jlohding/sp500-edgar-10k", split=f"train[:{max_rows}]")
    df = dataset.to_pandas()

    # Clean the metadata
    df['clean_company'] = df['company'].apply(standardize_company_name)

    # Item 1 (Business), Item 1A (Risk Factors), and Item 7 (Management's Discussion and Analysis).
    text_items = ['item_1', 'item_1A', 'item_7']
    docs = []

    # Restructure data into LangChain Documents
    for index, row in df.iterrows():
        for item in text_items:
            text_content = row[item]
            
            if not text_content or len(str(text_content).strip()) < 50:
                continue
                
            metadata = {
                "company": row['clean_company'],
                "original_name": row['company'],
                "cik": row['cik'],
                "date": str(row['date']),
                "item_type": item
            }
            
            docs.append(Document(page_content=str(text_content), metadata=metadata))

    if not docs:
        print("No documents found to ingest. Exiting.")
        return

    # Chunk the massive text into smaller semantic pieces
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunked_docs = text_splitter.split_documents(docs)
    print(f"Created {len(chunked_docs)} chunks from {len(docs)} sections.")

    # Initialize local embedding model and ingest into ChromaDB
    print("Initializing embedding model and saving to ChromaDB...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing embedding model on: {device.upper()}")

    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True, 'batch_size': 128}
    
    # TODO: upgrade to more precise embeddings bge-large-en-v1.5 or all-mpnet-base-v2
    # really bad: all-MiniLM-L6-v2
    embeddings = HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=chroma_path
    )

    print(f"Ingestion complete! Database saved to {chroma_path}")
