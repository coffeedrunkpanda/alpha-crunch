# Run: 
# python scripts/deploy/vector_db-upload-data-to-modal.py

import modal

vol = modal.Volume.from_name("alpha-crunch-vector-db", create_if_missing=True)

with vol.batch_upload() as batch:
    batch.put_directory(
        "data/chroma_db/",
        "/chroma_db/"
    )
    print("✅ uploaded chroma_db")
    
    batch.put_file(
        "data/company_registry.json",
        "/company_registry.json"
    )
    print("✅ uploaded company_registry.json")

    batch.put_file(
        "data/corpus_description.json",
        "/corpus_description.json"    
    )
    print("✅ uploaded corpus_description.json")

print("✅ Saved Vector DB data in the volume")

