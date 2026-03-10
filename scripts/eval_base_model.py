import os
import json
from pathlib import Path
from dotenv import load_dotenv
import wandb

import pandas as pd

import torch
from bert_score import score as bert_score
from openai import OpenAI

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

# env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Wandb
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_LABEL = "Mistral-7B-Instruct-v0.2"

# Project paths
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
dataset_dir = PROJECT_ROOT / "data" / "fiqa"

# Test Dataset
seed= 42
test_df = pd.read_csv(dataset_dir / "test_df.csv")
sample_df = test_df.sample(n=20, random_state=seed).reset_index(drop=True)

# load model locally
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=HF_TOKEN,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0}
)

client = OpenAI(api_key=OPENAI_API_KEY)

def llm_judge(question, context, reference, prediction, client):
    prompt = f"""You are evaluating a financial AI assistant.
Score the answer from 1 to 5 on each dimension.

Important:
- Context may or may not be provided.
- If context is empty, do not penalize the model for not citing or using context.
- Judge the answer based on the question, the reference answer, and financial correctness.
- If context is provided, also consider whether the answer is consistent with that context.
- Ignore minor style differences unless they affect correctness or completeness.

Question: {question}
Context: {context}
Reference Answer: {reference}
Model Answer: {prediction}

Return ONLY valid JSON with this schema:
{{"accuracy": 0, "reasoning": 0, "completeness": 0, "overall": 0}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)

def generate_answer(question, context, prompt_type):

    prompt_template = (
        "Use the following context to answer the question.\n\n"
        "Context: {context}\n\n"
        "Question: {question}"
    )

    if prompt_type == "context_grounded":
        content = prompt_template.format(context=context, question=question)

    elif prompt_type == "question_only":
        content = question
    else:
        raise ValueError(f"Invalid prompt_type: {prompt_type}")

    messages = [
        {"role": "system", "content": "You are a financial analyst assistant."},
        {"role": "user", "content": content},
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
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return answer, content

# ==================== EVALUATE + LOG ====================

run = wandb.init(
    project="finance-llm-evals",
    job_type="evaluation",
    name=f"eval-{MODEL_LABEL}",
    config={
        "model_label": MODEL_LABEL,
        "sample_size": len(sample_df),
        "judge_model": "gpt-4o",
        "dataset": "fiqa_test_sample",
        "seed": seed,
    },
)

# run.use_artifact(MODEL_ARTIFACT, type="adapter")

columns = [
        "idx",
        "model_label",
        "prompt",
        "has_context", # prompt has context? 
        "question",
        "context",
        "expected_answer",
        "predicted_answer",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "judge_accuracy",
        "judge_reasoning",
        "judge_completeness",
        "judge_overall",
    ]

table = wandb.Table(columns=columns)

predictions = []
references = []
rows_cache = []

for idx, row in sample_df.iterrows():
    question = row["question"]
    context = row["context"]
    expected = row["answer"]
    prompt_type = row['prompt_type']
    has_context = (prompt_type == "context_grounded")

    predicted, prompt = generate_answer(question=question,
                                context=context,
                                prompt_type=prompt_type,)

    row_data = {
        "idx": idx,
        "prompt": prompt,
        "has_context": has_context,
        "question": question,
        "context": context,
        "expected_answer": expected,
        "predicted_answer": predicted,
    }
    predictions.append(predicted)
    references.append(expected)
    rows_cache.append(row_data)


# Bert Scores: Precision, Recall, F1
device = "cuda" if torch.cuda.is_available() else "cpu"
P, R, F1 = bert_score(predictions, references, lang="en", device=device)

# LLM as judge
judge_scores = []
for i, row_data in enumerate(rows_cache):
    scores = llm_judge(
        question=row_data["question"],
        context=row_data["context"],
        reference=row_data["expected_answer"],
        prediction=row_data["predicted_answer"],
        client=client,
    )

    judge_scores.append(scores)

    table.add_data(
        row_data["idx"],
        MODEL_LABEL,
        row_data["prompt"],
        row_data["has_context"],
        row_data["question"],
        row_data["context"],
        row_data["expected_answer"],
        row_data["predicted_answer"],
        float(P[i]),
        float(R[i]),
        float(F1[i]),
        float(scores["accuracy"]),
        float(scores["reasoning"]),
        float(scores["completeness"]),
        float(scores["overall"]),
    )

avg_accuracy = sum(s["accuracy"] for s in judge_scores) / len(judge_scores)
avg_reasoning = sum(s["reasoning"] for s in judge_scores) / len(judge_scores)
avg_completeness = sum(s["completeness"] for s in judge_scores) / len(judge_scores)
avg_overall = sum(s["overall"] for s in judge_scores) / len(judge_scores)

f1_with_context = [
    float(F1[i]) for i, row_data in enumerate(rows_cache) if row_data["has_context"]
]
f1_without_context = [
    float(F1[i]) for i, row_data in enumerate(rows_cache) if not row_data["has_context"]
]

run.log({
    "eval_examples": table,
    "bertscore_precision_mean": float(P.mean()),
    "bertscore_recall_mean": float(R.mean()),
    "bertscore_f1_mean": float(F1.mean()),
    "judge_accuracy_mean": avg_accuracy,
    "judge_reasoning_mean": avg_reasoning,
    "judge_completeness_mean": avg_completeness,
    "judge_overall_mean": avg_overall,
    "count_with_context": sum(1 for row_data in rows_cache if row_data["has_context"]),
    "count_without_context": sum(1 for row_data in rows_cache if not row_data["has_context"]),
    "bertscore_f1_with_context_mean": (
        sum(f1_with_context) / len(f1_with_context) if f1_with_context else None
    ),
    "bertscore_f1_without_context_mean": (
        sum(f1_without_context) / len(f1_without_context) if f1_without_context else None
    ),
})

run.finish()