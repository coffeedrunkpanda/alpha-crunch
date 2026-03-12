from pathlib import Path

# Adjust this import based on your actual folder structure
# Assuming this script is running from the root or a scripts folder
from alpha_crunch.rag.ingest import build_vector_database, save_company_names_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_PATH = str(PROJECT_ROOT / "data" / "chroma_db")

print(PROJECT_ROOT)
print(CHROMA_PATH)


def main():
    # Dynamically find the project root
    print(f"Targeting Database Path: {CHROMA_PATH}")
    
    # Run the ingestion function
    build_vector_database(
        chroma_path=CHROMA_PATH, 
    )

if __name__ == "__main__":
    main()
