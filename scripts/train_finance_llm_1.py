
import pandas as pd

# Tokenizer, Model and Quantization 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import EarlyStoppingCallback

# Parameter Efficient Fine-Tuning
# https://huggingface.co/docs/transformers/main/peft
from peft import (LoraConfig,
                  TaskType,
                  get_peft_model,
                  prepare_model_for_kbit_training)


# Transformer reinforcement learning
from trl import SFTTrainer , SFTConfig

# LLMOps
import wandb

# Paths & Env
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
project_root = str(PROJECT_ROOT)

from alpha_crunch.finance_llm.dataset import build_hf_dataset, tokenize_for_eval

# ===================== CONFIG =====================

# Should configure this
project_name = "finance-llm"
experiment_name = "qlora-mistral-7b-baseline"
trained_weights = "mistralai/Mistral-7B-Instruct-v0.2"

# Paths to save the model
output_dir = os.path.join(project_root, "outputs/")
checkpoints_path = os.path.join(output_dir, project_name + "-checkpoints")
adapter_path = os.path.join(output_dir, project_name + "-adapter")


# PEFT/LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # type of task to train on
    r = 16, # dimension of the smallest matrix/rank adapter capacity
    lora_alpha = 32, # scaling factor
    lora_dropout= 0.05, # dropout of LoRA layers
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # which layers get LoRA
    bias = "none", # freeze the bias terms (only train LoRA matrices)
     
)

# SFT
sft_config = SFTConfig(
        dataset_text_field="chat",
        max_length=2048,
        report_to="wandb",
        output_dir= checkpoints_path,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4, # increase the effective size of the batch (4x4=16)
        gradient_checkpointing= True, 
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        warmup_ratio=0.03, # prevents loss spikes at start
        lr_scheduler_type="cosine", # better for fine-tuning
        learning_rate=2e-4, # standard QLoRA learning rate
        logging_steps=10,
    )


# QLoRA config: 4-bit storage, BF16 compute
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # NF4 is better than INT4 for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,  # RTX 5070 Ti supports BF16 natively
    bnb_4bit_use_double_quant= True # quantize the quantization constants too → saves ~0.4GB extra

)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(trained_weights)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"   # required for causal LM training

# ===================== INIT/CONST =================


wandb.login(key=os.getenv("WANDB_API_KEY")) 

run = wandb.init(
    project=project_name,
    name=experiment_name,
    config={
        # Model
        "model": trained_weights,

        # LoRA — only the fields you actually set
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "lora_target_modules": lora_config.target_modules,

        # Quantization
        "quantization": bnb_config.bnb_4bit_quant_type,
        "double_quant": bnb_config.bnb_4bit_use_double_quant,

        # Training
        "epochs": sft_config.num_train_epochs,
        "learning_rate": sft_config.learning_rate,
        "batch_size": sft_config.per_device_train_batch_size,
        "gradient_accumulation_steps": sft_config.gradient_accumulation_steps,
        "warmup_ratio": sft_config.warmup_ratio,
        "lr_scheduler": sft_config.lr_scheduler_type,
        "max_length": sft_config.max_length,

        # Dataset
        "dataset": "virattt/financial-qa-10K",
        "dataset_mix": "70ctx_30noctx",
    }
)

# Wandb run id to log after training
run_id = run.id

# =================== DATASET =====================

dataset_dir = PROJECT_ROOT / "data" / "fiqa"

train_csv = "train_df.csv"
val_csv = "val_df.csv"
test_csv = "test_df.csv"

train_df = pd.read_csv(str(dataset_dir / train_csv), keep_default_na=False)
val_df   = pd.read_csv(str(dataset_dir / val_csv),   keep_default_na=False)
test_df  = pd.read_csv(str(dataset_dir / test_csv),  keep_default_na=False)

train_dataset = build_hf_dataset(train_df, tokenizer, add_answer=True)
val_dataset   = build_hf_dataset(val_df,   tokenizer, add_answer=True)

# Tokenize for eval
test_dataset = build_hf_dataset(test_df, tokenizer, add_answer=True)  
test_dataset = tokenize_for_eval(test_dataset, tokenizer, max_length=sft_config.max_length)

dataset_splits = {
    "train": train_dataset, 
    "val":   val_dataset,
    "test":  test_dataset,
}

# log dataset sizes to W&B
wandb.log({
    "dataset/train_size": len(train_df),
    "dataset/val_size":   len(val_df),
    "dataset/test_size":  len(test_df),
})

# ==================== MODEL ======================

# quantized base model
base_model = AutoModelForCausalLM.from_pretrained(
    trained_weights,
    quantization_config=bnb_config,
    device_map={"": 0})

# device_map = "auto"

# prepare model for 4-bit quantization (enables gradient checkpointing and casts the right layers to trainable precision)
base_model = prepare_model_for_kbit_training(base_model) 

# Adapt model for the Lora 
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ==================== TRAIN ======================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_splits["train"],
    eval_dataset=dataset_splits["val"],
    processing_class=tokenizer,
    args=sft_config,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

try:
    trainer.train()
finally: # always save even if crash
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

# ==================== EVAL =======================

# --- Perplexity eval on test set (loss-based, fast) ---
trainer.eval_dataset = dataset_splits["test"]
test_results = trainer.evaluate()

print(test_results)
wandb.log({
    "test/final_eval_loss":        test_results["eval_loss"],
    "test/final_perplexity":       torch.exp(torch.tensor(test_results["eval_loss"])).item()
})

# --- Log adapter as W&B artifact ---
artifact = wandb.Artifact(
    name=project_name + "-adapter",
    type="adapter",
    description="QLoRA LoRA adapter — Mistral-7B-Instruct-v0.2 fine-tuned on financial QA",

    metadata={
        "lora_r":        lora_config.r,
        "lora_alpha":    lora_config.lora_alpha,
        "epochs":        sft_config.num_train_epochs,
        "learning_rate": sft_config.learning_rate,
        "eval_loss":     test_results["eval_loss"],
        "dataset":       "virattt/financial-qa-10K",
        "base_model": trained_weights,
        "eval_loss":  test_results["eval_loss"],
        "dataset":    "virattt/financial-qa-10K",
        "run_id":     run.id,
        
    }
)

artifact.add_dir(adapter_path)
run.log_artifact(artifact)

# ================== FINISH =======================
wandb.finish()
