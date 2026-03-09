# Run: 
# python scripts/deploy/finance_llm-upload_adapter.py

import modal

vol = modal.Volume.from_name("alpha-crunch-adapter", create_if_missing=True)

with vol.batch_upload() as batch:
    batch.put_file(
        "outputs/finance-llm-adapter/adapter_config.json",
        "/adapter/adapter_config.json"
    )
    print("✅ uploaded adapter_config.json")
    
    batch.put_file(
        "outputs/finance-llm-adapter/adapter_model.safetensors",
        "/adapter/adapter_model.safetensors"
    )
    print("✅ uploaded adapter_model.safetensors")


print("✅ Saved LoRA Adapter in the volume")

