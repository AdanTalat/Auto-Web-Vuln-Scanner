import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def prepare_data(file_path):
    print(f"Loading preprocessed data from: {file_path}")
    df = pd.read_csv(file_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Input CSV must contain 'text' and 'label' columns.")
    dataset = Dataset.from_pandas(df)
    print("Data loaded and converted to Hugging Face Dataset.")
    return dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def train_and_save_model():
    # Step 1: Load and split data
    data_path = os.path.join("Data", "equal_dataset.csv")
    full_dataset = prepare_data(data_path)
    dataset_splits = full_dataset.train_test_split(test_size=0.2, seed=42)
    print(f"Dataset split into {len(dataset_splits['train'])} training examples and "
          f"{len(dataset_splits['test'])} test examples.")

    # Step 2: Model & tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id
    )
    print(f"\nModel '{model_name}' loaded successfully.")

    # Step 3: Tokenization
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_datasets = dataset_splits.map(tokenize_fn, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns("text")
    print("Data tokenization complete.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Step 4: Training arguments
    training_args = TrainingArguments(
        output_dir="./Model/results",               # ✅ save checkpoints here
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        save_steps=500,                             # save every 500 steps
        save_total_limit=5,                         # keep last 5 checkpoints
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Step 5: Resume from latest checkpoint if available
    checkpoint_dir = "./Model/results"
    latest_checkpoint = None
    if os.path.isdir(checkpoint_dir):
        checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))

    print("\nStarting model training...")
    if latest_checkpoint:
        print(f"[+] Resuming from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("[+] No checkpoint found. Starting from scratch...")
        trainer.train()

    print("\nTraining complete!")

    # Step 6: Save final model
    output_dir = os.path.join('Model', 'distilbert_web_scanner')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving model and tokenizer to '{output_dir}'...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully!")


if __name__ == "__main__":
    train_and_save_model()
