import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

# --------------------------
# Parse CLI arguments
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
parser.add_argument("--model", type=str, required=True, help="Path to trained model")
args = parser.parse_args()

# --------------------------
# Load dataset
# --------------------------
print("[+] Loading dataset...")
dataset = load_dataset("csv", data_files=args.data)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
test_dataset = dataset["test"]

# --------------------------
# Load model + tokenizer
# --------------------------
print("[+] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# --------------------------
# Get predictions (probabilities)
# --------------------------
print("[+] Getting predictions...")
loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

all_probs, all_labels = [], []
model.eval()
with torch.no_grad():
    for batch in loader:
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]  # probability of "Malicious"
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch["label"].cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# --------------------------
# Threshold tuning
# --------------------------
print("[+] Calculating precision-recall curve...")
prec, rec, thresholds = precision_recall_curve(all_labels, all_probs)

f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\n[+] Best Threshold = {best_threshold:.3f}")
print(f"Precision = {prec[best_idx]:.3f}, Recall = {rec[best_idx]:.3f}, F1 = {f1_scores[best_idx]:.3f}")

# Apply threshold
preds = (all_probs >= best_threshold).astype(int)

print("\n[+] Classification Report")
print(classification_report(all_labels, preds, target_names=["Benign", "Malicious"]))

print("\n[+] Confusion Matrix")
print(confusion_matrix(all_labels, preds))

# --------------------------
# Plot precision-recall curve
# --------------------------
plt.plot(rec, prec, label="Precision-Recall curve")
plt.scatter(rec[best_idx], prec[best_idx], color="red", label=f"Best threshold {best_threshold:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve with Best Threshold")
plt.show()
