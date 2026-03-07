import pandas as pd
from datasets import Dataset

def format_row(row, tokenizer, add_answer=True):

    format = "Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}"

    try:
    
        if row["prompt_type"] == "context_grounded":
            content = format.format(context=row["context"], question=row["question"])
        elif row["prompt_type"] == "question_only":
            content = row["question"]
        else:
            raise ValueError(f"Invalid prompt_type: {row['prompt_type']}")


        chat = [{"role": "user", "content": content}]

        if add_answer: 
            chat.append({"role": "assistant", "content": row["answer"]})

            return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    
        else:
            return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    except Exception as e:
        print(f"[format_row error] {e}")
        return None

def build_hf_dataset(df, tokenizer, add_answer = True):

    formatted = df.apply(lambda x: format_row(x, tokenizer, add_answer=add_answer), axis=1).tolist()
    formatted = [x for x in formatted if x is not None]  # filter failed rows

    return Dataset.from_dict({"chat": formatted})

def load_datasets(dataset_dir, tokenizer):

    train_csv = "train_df.csv"
    val_csv = "val_df.csv"
    test_csv = "test_df.csv"

    train_df = pd.read_csv(str(dataset_dir / train_csv), keep_default_na=False)
    val_df   = pd.read_csv(str(dataset_dir / val_csv),   keep_default_na=False)
    test_df  = pd.read_csv(str(dataset_dir / test_csv),  keep_default_na=False)

    train_dataset = build_hf_dataset(train_df, tokenizer, add_answer=True)
    val_dataset   = build_hf_dataset(val_df,   tokenizer, add_answer=True)
    test_dataset = build_hf_dataset(test_df, tokenizer, add_answer=True)  # needs answers for loss

    dataset_splits = {
        "train": train_dataset, 
        "val":   val_dataset,
        "test":  test_dataset,
    }

    return dataset_splits