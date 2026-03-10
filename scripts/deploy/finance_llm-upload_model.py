# Run once: python scripts/deploy/finance_llm_upload.py
import os
import modal

base_volume = modal.Volume.from_name("alpha-crunch-models", create_if_missing=True)

# ── Download base model (HF → volume, runs in Modal cloud) ────────
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub")
)

app = modal.App("alpha-crunch-upload")

@app.function(
    image=download_image,
    volumes={"/models": base_volume},
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def upload_base_model():
    from huggingface_hub import snapshot_download

    model_path = "/models/mistral-7b"

    if os.path.exists(model_path) and os.listdir(model_path):
        print("✅ Base model already in volume — skipping")
        return

    print("⬇️ Downloading Mistral 7B (~14GB)...")
    snapshot_download(
        "mistralai/Mistral-7B-Instruct-v0.2",
        local_dir=model_path,
        token=os.environ["HF_TOKEN"],
    )
    base_volume.commit()
    print("✅ Base model saved to volume")


@app.local_entrypoint()
def main():

    # Base model: runs in Modal cloud (snapshot_download)
    print("🚀 Uploading base model to Modal cloud...")
    upload_base_model.remote()

    print("🎉 All uploads complete — ready to deploy")
