import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: python check_dataset_balance.py dataset.csv")
    sys.exit(1)

dataset_path = sys.argv[1]

# Load dataset
df = pd.read_csv(dataset_path)

# Expect columns: text, label
if "label" not in df.columns:
    print("[ERROR] No 'label' column found. Check your CSV format.")
    sys.exit(1)

# Count per class
counts = df["label"].value_counts().sort_index()

print("\n[+] Dataset Class Distribution")
print("--------------------------------")
for label, count in counts.items():
    label_name = "Benign" if label == 0 else "Malicious"
    print(f"{label} ({label_name}): {count}")

print("\n[+] Total samples:", len(df))
