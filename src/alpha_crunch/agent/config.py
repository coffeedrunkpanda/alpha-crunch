import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CHROMA_PATH = str(DATA_DIR/ "chroma_db")
COMPANIES_JSON_FILE = DATA_DIR / "company_registry.json"

# TODO: add embedding and k chunks here
# TODO: Verify if I need any additional companies

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
    "JP MORGAN": "J P MORGAN CHASE"
}

# STATIC DATA LOADING (Runs once on import)
try:
    with open(COMPANIES_JSON_FILE, "r", encoding="utf-8") as f:
        SP500_CLEAN_NAMES = tuple(json.load(f))
    COMPANY_REGISTRY = sorted(SP500_CLEAN_NAMES, key=len, reverse=True)

except FileNotFoundError:
    print(f"⚠️ WARNING: {COMPANIES_JSON_FILE} not found.")
    COMPANY_REGISTRY = tuple()
