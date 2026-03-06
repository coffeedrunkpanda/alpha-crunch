
import random
from datasets import load_dataset, Dataset

# Tokenizer, Model and Quantization 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

# ===================== CONFIG =====================

# Should configure this
project_name = "finance-llm"
experiment_name = "qlora-mistral-7b-run1"
trained_weights = "mistralai/Mistral-7B-Instruct-v0.2"

# Paths to save the model
output_dir = os.path.join(project_root, "outputs/")
checkpoints_path = os.path.join(output_dir, project_name + "-checkpoints")
adapter_path = os.path.join(output_dir, project_name + "-adapter")

# TODO: add this
# target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# ["q_proj", "v_proj"]

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
    )


# QLoRA config: 4-bit storage, BF16 compute
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # NF4 is better than INT4 for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,  # RTX 5070 Ti supports BF16 natively
    bnb_4bit_use_double_quant= True # quantize the quantization constants too → saves ~0.4GB extra

)

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

# ===================== FUNCTIONS =================

def format_row(row, tokenizer, include_context = True):

    format_1 = """Use the following context to answer the question.

    Context: {context}

    Question: {question}
    """

    try: 
        context = row["context"]
        question = row["question"]
        answer = row["answer"]

        # FORMAT 1: Teaches the model to use a context to answer (using RAG later)
        if include_context:
            content = format_1.format(context = context, question=question)

            chat = [{"role": "user", "content": content},
            {"role": "assistant", "content": answer}]
        
        # FORMAT 2 : When no context is available: use only the question
        else:
            chat = [{"role": "user", "content": question},
            {"role": "assistant", "content": answer}]

        # format chat
    
        chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    
        return chat

    except Exception as e: 
        print(e)
        return None    

def create_chat_dataset(df, tokenizer):

    # context grounded (70%)
    context_grounded = df.apply(
        lambda x: format_row(x, tokenizer, include_context=True),axis=1
        ).to_list()

    # questions only (30%)
    questions_only = df.sample(n=int(len(df) * 0.3 / 0.7), random_state=42)
    questions_only = questions_only.apply(
            lambda x: format_row(x, tokenizer, include_context=False), axis=1
            )

    context_grounded.extend(questions_only)
    combined = [x for x in context_grounded if x is not None] 
    random.shuffle(combined)

    return Dataset.from_dict({"chat": combined}) 


# =================== DATASET =====================

ds = load_dataset("virattt/financial-qa-10K")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(trained_weights)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"   # required for causal LM training


df2 = ds["train"].to_pandas()
dataset = create_chat_dataset(df2, tokenizer)

# split into train, val and test
split_1 = dataset.train_test_split(test_size=0.2, seed = 42)
split_2 = split_1["test"]. train_test_split(test_size=0.5, seed = 42)

# TODO: dont do this for the test, later need regex
dataset_splits = {
    "train": split_1["train"],
    "val": split_2["train"],
    "test": split_2["test"]
}

# log dataset sizes to W&B
wandb.log({
    "dataset/train_size": len(dataset_splits["train"]),
    "dataset/val_size":   len(dataset_splits["val"]),
    "dataset/test_size":  len(dataset_splits["test"]),
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
    args=sft_config
)

try:
    trainer.train()
finally: # always save even if crash
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)


# ==================== EVAL =======================

# Perplexity eval? 
trainer.eval_dataset = dataset_splits["test"]   # ✅ IMPROVEMENT: swap in place — same W&B run
test_results = trainer.evaluate()
print(test_results)
wandb.log({"test/final_loss": test_results["eval_loss"]})

# ================== FINISH =======================

wandb.finish()