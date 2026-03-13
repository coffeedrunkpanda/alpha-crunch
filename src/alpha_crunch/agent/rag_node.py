import re
from alpha_crunch.agent.state import AgentState
from alpha_crunch.agent.config import COMPANY_ALIASES, COMPANY_REGISTRY
from alpha_crunch.agent.vector_store import get_chroma_retriever
from alpha_crunch.agent.tools import CORPUS_INFO, COMPANIES

# TODO: Add fallback to Gemini or openai if no company was found.
# this my happen in case of misspelling or aliases for the companies, 
# or companies that are not available in the db, currently.

def rag_node(state: AgentState) -> dict:
    """
    The LangGraph node function for RAG.
    It takes the current state, uses the query to search ChromaDB,
    and updates the state with the formatted retrieved context.
    """

    print("\n--- 🚨 DEBUG STATE ---")
    print(f"Number of messages: {len(state.messages)}")
    if state.messages:
        print(f"Type of last message: {type(state.messages[-1])}")
        print(f"Content of last message: {state.messages[-1].content}")
    else:
        print("WARNING: state.messages is EMPTY!")
    print("----------------------\n")


    current_query = state.messages[-1].content

    print(f"--- RAG NODE: Searching for '{current_query}' ---")

    target_company = extract_target_company(current_query)
    print(f"--- RAG NODE: Extracted Entity -> {target_company} ---")
    
    # Get the retriever and fetch documents
    retriever = get_chroma_retriever()
    
    if target_company != "NONE":
        print(f"--- RAG NODE: Applying Strict Metadata Filter for {target_company} ---")
        retriever.search_kwargs = {
            "k": 3, 
            "filter": {"company": target_company} 
        }

    else:
        # Fallback: if no company was found, do a broad semantic search
        print("--- RAG NODE: No specific company detected. Doing broad search. ---")
        retriever.search_kwargs = {"k": 3}
    
    # 4. Invoke the search
    docs = retriever.invoke(current_query)
        
    if not docs:
        disclaimer = f"**Disclaimer**: No data found. Dataset covers {CORPUS_INFO['coverage']['years']['min_year']}-{CORPUS_INFO['coverage']['years']['max_year']} for {len(COMPANIES)} S&P 500 companies (see /help)."
        return {"retrieved_context": disclaimer}
    
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

    disclaimer = f"""**RAG CONTEXT DISCLAIMER** (Dataset: {CORPUS_INFO['source']['dataset_description']}):
- Coverage: {CORPUS_INFO['coverage']['years']['min_year']}-{CORPUS_INFO['coverage']['years']['max_year']} ({len(COMPANIES)} companies: see /help).
- Items: {', '.join(CORPUS_INFO['coverage']['items_included'])}.
- Query focused on: {target_company if target_company != 'NONE' else 'broad search'}.
- Cite sources precisely. Say 'No data post-2022' if needed.

**Retrieved Docs ({len(docs)} chunks):**
"""
    full_context = disclaimer + formatted_context
    
    print(f"--- RAG NODE: Context w/ disclaimer ({len(full_context)} chars) ---")

    # In LangGraph, returning a dictionary updates the state
    return {"retrieved_context": full_context}

def extract_target_company(query: str) -> str:
    """Extracts and formats the company name from the user's query."""

    # First using company's list
    """Hybrid Extractor with Alias Resolution."""
    normalized_query = query.upper().replace("'", "").replace('"', "")
    normalized_query = re.sub(r'[.,?]', '', normalized_query)
    
    # 1. Check for Aliases FIRST
    # We iterate through our known aliases to see if the user typed one
    for alias, official_name in COMPANY_ALIASES.items():
        pattern = r'\b' + re.escape(alias) + r'S?\b'
        if re.search(pattern, normalized_query):
            print(f"--- RAG NODE: Resolved Alias '{alias}' -> '{official_name}' ---")
            return official_name


    # 2. Proceed to standard Exact Matching
    sorted_names = sorted(COMPANY_REGISTRY, key=len, reverse=True)
    
    for company in sorted_names:
        pattern = r'\b' + re.escape(company) + r'S?\b'
        
        if re.search(pattern, normalized_query):
            # Short-name safety check (e.g., 'CA', '3M')
            if len(company) <= 3:
                strict_pattern = r'\b' + re.escape(company) + r'S?\b'
                if not re.search(strict_pattern, query.replace("'", "").replace('"', "")):
                    continue 
            return company

    return "NONE"
