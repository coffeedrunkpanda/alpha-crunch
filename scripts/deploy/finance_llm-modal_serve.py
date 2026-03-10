# scripts/deploy/finance_llm_modal.py
import os
import modal
import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager

from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


APP_NAME = "alpha-crunch-finance-llm"
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

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

base_volume = modal.Volume.from_name("alpha-crunch-models", create_if_missing=True)
adapter_volume = modal.Volume.from_name("alpha-crunch-adapter", create_if_missing=True)

class ChatRequest(BaseModel):
    message: str
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
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.asgi_app(requires_proxy_auth=True)
def serve():
    model = None
    tokenizer = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal model, tokenizer

        hf_token = os.environ["HF_TOKEN"]

        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID,
            token=hf_token,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(
            base_model,
            f"{ADAPTER_MOUNT}/adapter",
        )
        model.eval()

        yield

    web_app = FastAPI(lifespan=lifespan)

    @web_app.get("/health")
    def health():
        return {"status": "ok"}

    @web_app.post("/generate")
    def generate(req: ChatRequest):
        messages = [
            {"role": "system", "content": "You are a financial analyst assistant."},
            {"role": "user", "content": req.message},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
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
