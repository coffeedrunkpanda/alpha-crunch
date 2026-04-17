"""
Modal ASGI app: Chroma vector search API.

Deploy: modal deploy scripts/deploy/vector_db-api.py

Health: GET /health  (returns immediately; does not wait for embeddings/Chroma)
Ready: GET /ready   (loads model + Chroma; use to verify volumes)
Query: POST /query
"""

import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Optional

import modal
import torch
from fastapi import FastAPI
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

APP_NAME = "alpha-crunch-vector-db"
VECTOR_DB_MOUNT = "/data"
# Must match upload layout: data/chroma_db -> /data/chroma_db on the volume
CHROMA_SUBDIR = "chroma_db"
EMBEDDINGS_MOUNT = "/embeddings"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "pydantic",
        "fastapi[standard]",
        "langchain-core",
        "langchain-chroma",
        "chromadb",
        "langchain-community",
        "langchain-huggingface",
        "langchain-text-splitters",
        "huggingface-hub",
        "sentence-transformers", 
    )
)

vector_db_volume = modal.Volume.from_name("alpha-crunch-vector-db")
embeddings_volume = modal.Volume.from_name("alpha-crunch-embeddings", create_if_missing=True)


class VectorDBRequest(BaseModel):
    query: str
    k: int = 3
    filter: Optional[dict] = None


@app.function(
    image=image,
    timeout=60 * 20,
    scaledown_window=15 * 60,
    volumes={
        VECTOR_DB_MOUNT: vector_db_volume,
        EMBEDDINGS_MOUNT: embeddings_volume,
    },
)
@modal.asgi_app(requires_proxy_auth=True)
def serve_vector_db_api():
    embedding_model_name = "all-mpnet-base-v2"
    embedding_dir = os.path.join(EMBEDDINGS_MOUNT, embedding_model_name)
    chroma_persist = os.path.join(VECTOR_DB_MOUNT, CHROMA_SUBDIR)

    state: dict[str, Any] = {
        "embedding_model": None,
        "vector_store_chroma": None,
        "init_error": None,
    }
    init_lock = threading.Lock()

    def _ensure_initialized() -> None:
        """Load embedding model + Chroma once (first /query or /ready)."""
        with init_lock:
            if state["vector_store_chroma"] is not None:
                return
            if state["init_error"] is not None:
                raise RuntimeError(state["init_error"])

            try:
                if not os.path.isdir(embedding_dir) or not os.listdir(embedding_dir):
                    print("⬇️ Downloading embedding model from Hugging Face...")
                    os.makedirs(embedding_dir, exist_ok=True)
                    snapshot_download(
                        repo_id=f"sentence-transformers/{embedding_model_name}",
                        local_dir=embedding_dir,
                        token=os.environ.get("HF_TOKEN"),
                    )
                    embeddings_volume.commit()
                    print("✅ Embedding model downloaded")

                print("loading embedding model...")
                state["embedding_model"] = HuggingFaceEmbeddings(
                    model_name=embedding_dir,
                    model_kwargs={
                        "device": "cuda" if torch.cuda.is_available() else "cpu",
                        "local_files_only": True,
                    },
                    encode_kwargs={"normalize_embeddings": True},
                )
                print("✅ Embedding model loaded")

                if not os.path.isdir(chroma_persist):
                    raise FileNotFoundError(
                        f"Chroma persist directory missing: {chroma_persist}. "
                        "Upload data/chroma_db to the alpha-crunch-vector-db volume."
                    )

                print("Creating vector store...")
                state["vector_store_chroma"] = Chroma(
                    embedding_function=state["embedding_model"],
                    persist_directory=chroma_persist,
                )
                print("✅ Vector store created")
            except Exception as e:
                state["init_error"] = str(e)
                raise

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Empty lifespan so /health responds without waiting for model + Chroma load.
        yield

    web_app = FastAPI(lifespan=lifespan)

    @web_app.get("/health")
    def health():
        return {"status": "ok"}

    @web_app.get("/ready")
    def ready():
        try:
            _ensure_initialized()
            return {"ready": True}
        except Exception as e:
            return {"ready": False, "error": str(e)}

    @web_app.post("/query")
    def query(request: VectorDBRequest):
        _ensure_initialized()
        vs = state["vector_store_chroma"]
        if request.filter:
            retriever = vs.as_retriever(
                search_kwargs={"k": request.k, "filter": request.filter},
            )
        else:
            retriever = vs.as_retriever(search_kwargs={"k": request.k})
        docs = retriever.invoke(request.query)
        return [
            {"page_content": d.page_content, "metadata": dict(d.metadata)}
            for d in docs
        ]

    return web_app
