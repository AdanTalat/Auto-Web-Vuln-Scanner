import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# -------------------------------
# Load Dataset
# -------------------------------
DATASET_PATH = "./Data/equal_dataset.csv"   # <-- change to your dataset path
df = pd.read_csv(DATASET_PATH)  # normal CSV
# Assuming dataset format: label \t text
df["label"] = df["label"].astype(int)

# Split into train/test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# -------------------------------
# Load Model + Tokenizer
# -------------------------------
MODEL_PATH = "./Model/distilbert_web_scanner"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Tokenize
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

class WebDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx])
        return item

train_dataset = WebDataset(train_encodings, train_labels)
test_dataset = WebDataset(test_encodings, test_labels)

# -------------------------------
# Define Trainer for Evaluation
# -------------------------------
training_args = TrainingArguments(
    output_dir="./results_eval",
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset
)

# -------------------------------
# Evaluate
# -------------------------------
raw_pred = trainer.predict(test_dataset)
y_pred = raw_pred.predictions.argmax(-1)

print("\n[+] Classification Report")
print(classification_report(test_labels, y_pred, target_names=["Benign", "Malicious"]))

print("\n[+] Confusion Matrix")
print(confusion_matrix(test_labels, y_pred))
