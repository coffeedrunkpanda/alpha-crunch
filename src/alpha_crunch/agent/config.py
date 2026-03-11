from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHROMA_PATH = str(PROJECT_ROOT / "data" / "chroma_db")

# TODO: add embedding and k chunks here