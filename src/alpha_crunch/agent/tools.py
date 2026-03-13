# alpha-crunch/src/alpha_crunch/agent/tools.py
import json
from pathlib import Path
from alpha_crunch.agent.config import CHROMA_PATH

from langchain_core.tools import tool
from typing import Dict, List

DATA_PATH = Path(CHROMA_PATH).parent

# Load corpus info

with open(DATA_PATH / "corpus_description.json", "r") as f:
    CORPUS_INFO = json.load(f)

# Load companies
with open(DATA_PATH / "company_registry.json", "r") as f:
    COMPANIES = json.load(f)

@tool
def get_dataset_help() -> str:
    """Get overview of available companies, data coverage, items, and knowledge cutoff."""
    help_text = f"""
Dataset: {CORPUS_INFO['source']['dataset_description']}
Cutoff: Data covers {CORPUS_INFO['coverage']['years']['min_year']}-{CORPUS_INFO['coverage']['years']['max_year']} (ingested {CORPUS_INFO['source']['ingestion_timestamp_utc']}).
Companies ({len(COMPANIES)} available): {', '.join(COMPANIES)}
The available information is correspondent to the items {CORPUS_INFO['coverage']['items_included']} of the (Form 10-K).
Available 10-K Items:
"""
    for item, desc in CORPUS_INFO['coverage']['item_descriptions'].items():
        help_text += f"- {item}: {desc}\n"
    help_text += f"\nMetadata: {', '.join(CORPUS_INFO['metadata_schema'].keys())}\nAsk about specific companies/items from {CORPUS_INFO['coverage']['years']['min_year']}-{CORPUS_INFO['coverage']['years']['max_year']}!"
    return help_text


TOOLS = [get_dataset_help]