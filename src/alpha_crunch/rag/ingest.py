import os
import re
import json
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
import torch
import pandas as pd
from datetime import datetime, timezone

# Modern LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

COMPANY_ALIASES = {
    "GOOGLE": "ALPHABET",
    "COMPUTER ASSOCIATES": "CA",
    "FB": "META PLATFORMS",
    "FACEBOOK": "META PLATFORMS",
    "CVS": "CVS CAREMARK",
    "HP": "HEWLETT PACKARD",
    "IBM": "INTERNATIONAL BUSINESS MACHINES",
    "GE": "GENERAL ELECTRIC",
    "GM": "GENERAL MOTORS",
    "JPM": "J P MORGAN CHASE",
    "JP MORGAN": "J P MORGAN CHASE",
    "JPMORGAN CHASE": "J P MORGAN CHASE",
    "COSTCO": "COSTCO WHOLESALE",
    "UNITED TECHNOLOGIES": "RAYTHEON TECHNOLOGIES",
    "WAL MART STORES": "WALMART",
    "WALMART STORES": "WALMART",
    "LILLY ELI": "ELI LILLY",
}

ITEM_COLUMNS = [
    "item_1",   # Business
    "item_1A",  # Risk Factors
    "item_5",   # Market for Common Equity
    "item_6",   # Selected Financial Data
    "item_7",   # MD&A
    "item_7A",  # Market Risk
    "item_8",   # Financial Statements 
]

ITEM_DESCRIPTIONS = {
    "item_1": "Business: Overview of operations, products, services, and markets.",
    "item_1A": "Risk Factors: Detailed disclosure of potential risks to the company.",
    "item_5": "Market for Common Equity: Information on stock performance and dividends.",
    "item_6": "Selected Financial Data: Summarized financial data for the past five years.",
    "item_7": "Management's Discussion and Analysis (MD&A): Management's perspective on financial condition and results.",
    "item_7A": "Quantitative and Qualitative Disclosures About Market Risk: Risks related to interest rates, foreign currency, etc.",
    "item_8": "Financial Statements and Supplementary Data: Audited financial statements, including balance sheets and income statements.",
}

def standardize_company_name(name: str) -> str:
    """Standardizes company names by making them uppercase, removing suffixes, and stripping SEC formatting."""
    name = str(name).upper()
    
    # 1. Remove standard SEC state abbreviations (/DE/, \DE\, /NEW/, /MN, /CA/)
    name = re.sub(r'[/\\][A-Z]{2,3}[/\\]?', '', name)
    
    # 2. Remove punctuation (commas, periods)
    name = re.sub(r'[.,]', '', name)
    
    # 3. Remove common corporate suffixes
    suffixes = r'\b(INC|CORP|CORPORATION|LLC|LTD|COMPANY|CO|GROUP|HOLDINGS|COM)\b'
    name = re.sub(suffixes, '', name)
    
    # 4. Strip trailing garbage (ampersands, slashes, dashes, or single stray letters at the very end)
    # This loop ensures that if it ends with " &", it removes it. If it then ends with a space, it removes it.
    # It cleans "RAYTHEON /" -> "RAYTHEON" and "TIFFANY &" -> "TIFFANY"
    name = re.sub(r'[\s&/\-]+$', '', name)
    
    # Run it one more time just in case there was a double space like "TIFFANY  &"
    name = re.sub(r'[\s&/\-]+$', '', name)
    
    # 5. Strip leading/trailing whitespaces and extra internal spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def build_vector_database(chroma_path: str, embedding_model:str = "all-mpnet-base-v2"):
    """
    Pulls SEC 10-K data from Hugging Face, cleans the metadata, chunks the text, 
    and ingests it into a local ChromaDB instance.
    """

    if not os.path.isdir(chroma_path):
        os.makedirs(chroma_path, exist_ok=True)

    print(f"Loading dataset from Hugging Face...")
    ds = load_dataset("jlohding/sp500-edgar-10k")
    df = ds["train"].to_pandas()
    df_top_companies = get_top_companies(df)

    companies = df_top_companies['clean_company'].dropna().unique().tolist()
    companies.sort(key=len, reverse=True)
    print("Unique companies: ", len(companies))


    save_company_names_json(chroma_path, df_top_companies)
    save_corpus_description_json(chroma_path, df_top_companies)

    docs = []

    # Restructure data into LangChain Documents
    for index, row in df_top_companies.iterrows():
        for item in ITEM_COLUMNS:
            text_content = row[item]
            
            if not text_content or len(str(text_content).strip()) < 50:
                continue
                
            metadata = {
                "company": row['clean_company'],
                "original_name": row['company'],
                "cik": row['cik_int'],
                "date": str(row['date']),
                "year": str(row['date'].year),
                "item_type": item,
                "ticker": row["ticker"]
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
    encode_kwargs = {'normalize_embeddings': True, 'batch_size': 64}
    
    # TODO: upgrade to more precise embeddings bge-large-en-v1.5 or all-mpnet-base-v2
    # really bad: all-MiniLM-L6-v2
    embeddings = HuggingFaceEmbeddings(
        model_name= embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=chroma_path
    )

    print(f"Ingestion complete! Database saved to {chroma_path}")

def save_company_names_json(chroma_path: str, df: pd.DataFrame):
    # Save to a config or data folder, not the source code folder
    TARGET_JSON_FILE = Path(chroma_path).parent / "company_registry.json"

    companies = df['clean_company'].dropna().unique().tolist()
    companies.sort(key=len, reverse=True)
    print("Unique companies: ", len(companies))

    # Ensure directory exists
    TARGET_JSON_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(TARGET_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(companies, f, indent=4) # indent=4 makes it beautifully formatted!
        
    print(f"✅ Successfully wrote company registry to: {TARGET_JSON_FILE}")

def get_top_companies(df: pd.DataFrame):
    top_50_ticker_to_cik = {
        "NVDA": "0001045810",
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "AMZN": "0001018724",
        "GOOGL": "0001652044",
        "GOOG": "0001652044",
        "META": "0001326801",
        "AVGO": "0001730168",
        "TSLA": "0001318605",
        "BRK.B": "0001067983",
        "WMT": "0000104169",
        "LLY": "0000059478",
        "JPM": "0000019617",
        "XOM": "0000034088",
        "V": "0001403161",
        "JNJ": "0000200406",
        "ORCL": "0001341439",
        "MU": "0000723125",
        "MA": "0001141391",
        "COST": "0000909832",
        "ABBV": "0001551152",
        "NFLX": "0001065280",
        "CVX": "0000093410",
        "PLTR": "0001321655",
        "PG": "0000080424",
        "HD": "0000354950",
        "BAC": "0000070858",
        "KO": "0000021344",
        "GE": "0000040545",
        "AMD": "0000002488",
        "CAT": "0000018230",
        "CSCO": "0000858877",
        "MRK": "0000310158",
        "RTX": "0000101829",
        "AMAT": "0000006951",
        "LRCX": "0000707549",
        "PM": "0001413329",
        "UNH": "0000731766",
        "MS": "0000895421",
        "GS": "0000886982",
        "TMUS": "0001283699",
        "INTC": "0000050863",
        "IBM": "0000051143",
        "MCD": "0000063908",
        "WFC": "0000072971",
        "LIN": "0001707925",
        "PEP": "0000077476",
        "VZ": "0000732717",
        "AXP": "0000004962",
    }

    int_ciks = [int(item) for item in top_50_ticker_to_cik.values()]
    tickers = list(top_50_ticker_to_cik.keys())  # Convert to list
    ticker_to_cik = dict(zip(tickers, int_ciks))
    cik_to_ticker = dict(zip(int_ciks, tickers))

    # Safe int conversion (handles str/NaN)
    df['cik_int'] = pd.to_numeric(df['cik'], errors='coerce').fillna(-1).astype(int)

    # Filter FIRST (only top 50 CIKs)
    df_top50 = df[df['cik_int'].isin(int_ciks)].copy()

    # Now safe to map ticker (all cik_int are in cik_to_ticker)
    df_top50['ticker'] = df_top50['cik_int'].map(cik_to_ticker)

    # standardize the names
    df_top50['clean_company'] = df_top50['company'].apply(standardize_company_name)
    df_top50['clean_company'] = df_top50['clean_company'].apply(lambda x: COMPANY_ALIASES[x] if x in COMPANY_ALIASES else x)

    return df_top50

def save_corpus_description_json(chroma_path: str, df: pd.DataFrame):
    """
    Save a high-level description of what went into this Chroma DB.
    """

    description_path = Path(chroma_path).parent / "corpus_description.json"

    years = sorted(df.date.dt.year.dropna().unique().tolist())
    companies = df["clean_company"].dropna().unique().tolist()

    description = {
        "source": {
            "dataset": "jlohding/sp500-edgar-10k",
            "dataset_description": (
                "Annual reports (Form 10-K) for historical S&P 500 "
                "constituents from 2010–2022, pulled from SEC EDGAR."
            ),
            "ingestion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "coverage": {
            "years": {
                "min_year": int(min(years)) if years else None,
                "max_year": int(max(years)) if years else None,
                "included_years": years,
            },
            "items_included": ITEM_COLUMNS,
            "companies_count": len(companies),
            "item_descriptions": ITEM_DESCRIPTIONS,
            
        },
        "metadata_schema": {
            "company": "Standardized company name",
            "original_name": "Original company name from dataset",
            "cik": "SEC Central Index Key",
            "date": "Filing date (string, ISO format)",
            "year": "Filing year (int)",
            "item_type": "10-K item column name, e.g. item_1A",
            "ticker": "asset/stock identification"
        },
        "chunking": {
            "strategy": "RecursiveCharacterTextSplitter",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separators": ["\\n\\n", "\\n", ".", " ", ""],
        },
    }

    description_path.parent.mkdir(parents=True, exist_ok=True)
    with open(description_path, "w", encoding="utf-8") as f:
        json.dump(description, f, indent=4)

    print(f"📝 Wrote corpus description to: {description_path}")
