# Run: modal deploy scripts/deploy/finance_llm_modal.py
import os
import modal
import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager

from pydantic import BaseModel
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


APP_NAME = "alpha-crunch-finance-llm"
MODEL_MOUNT = "/models"
ADAPTER_MOUNT = "/adapters"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers==5.2.0",
        "accelerate",
        "peft",
        "fastapi[standard]",
        "pydantic",
        "sentencepiece",
        "safetensors",
    )
)

adapter_volume = modal.Volume.from_name("alpha-crunch-adapter")
base_volume = modal.Volume.from_name("alpha-crunch-models")

class ChatRequest(BaseModel):
    messages: List[Dict[str,str]]
    max_new_tokens: int = 256
    temperature: float = 0.2

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 20,
    scaledown_window=15 * 60,
    volumes={
        MODEL_MOUNT: base_volume,
        ADAPTER_MOUNT: adapter_volume,
    },
)

@modal.asgi_app(requires_proxy_auth=True)
def serve():
    model = None
    tokenizer = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal model, tokenizer

        print("⬇️ Loading tokenizer from volume...")
        tokenizer = AutoTokenizer.from_pretrained(
            f"{MODEL_MOUNT}/mistral-7b",   # ← from volume, not HF
            local_files_only=True,          # ← never hits HF
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        print("✅ Tokenizer loaded")

        print("⬇️ Loading base model from volume...")
        base_model = AutoModelForCausalLM.from_pretrained(
            f"{MODEL_MOUNT}/mistral-7b",   # ← from volume, not HF
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,          # ← never hits HF
        )
        print("✅ Base model loaded")

        print("⬇️ Loading LoRA adapter from volume...")
        model = PeftModel.from_pretrained(
            base_model,
            f"{ADAPTER_MOUNT}/adapter",
        )
        model.eval()
        print("✅ Adapter loaded — ready to serve")

        yield

    web_app = FastAPI(lifespan=lifespan)

    @web_app.get("/health")
    def health():
        return {"status": "ok"}

    @web_app.post("/generate")
    def generate(req: ChatRequest):

        prompt = tokenizer.apply_chat_template(
            req.messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                do_sample=req.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {"response": text}

    return web_app
