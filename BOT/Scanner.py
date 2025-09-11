import os
import sys
import csv
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./Model/distilbert_web_scanner"

class WebVulnerabilityScanner:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            print(f"[ERROR] Model path '{model_path}' not found.")
            sys.exit(1)
        print(f"[+] Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")
            self.device = "cuda"
        else:
            self.device = "cpu"
        print("[+] Model loaded successfully!")

    def classify_text(self, text):
        if not text or not isinstance(text, str):
            return "Input empty or invalid"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predicted_class_id = torch.argmax(probs, dim=1).item()
        return "Malicious" if predicted_class_id == 1 else "Benign"

def main():
    scanner = WebVulnerabilityScanner()

    payloads = [
        "' OR '1'='1",
        "<script>alert(1)</script>",
        "../../../../../etc/passwd",
        "'; DROP TABLE users; --"
    ]

    print("\nEnter website URLs to scan (type 'exit' to quit):")

    # Open log CSV and write header if new file
    log_path = "scan_results.csv"
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if new_file:
            writer.writerow(["URL", "Payload", "Status", "Prediction"])

        while True:
            url = input("> ").strip()
            if url.lower() == "exit":
                break
            if not url.startswith(("http://", "https://")):
                print("[!] Please enter a valid URL starting with http:// or https://")
                continue

            print(f"\nStarting scan on: {url}")

            for payload in payloads:
                if "?" in url:
                    test_url = f"{url}&test={requests.utils.quote(payload)}"
                else:
                    test_url = f"{url}?test={requests.utils.quote(payload)}"

                print(f"\n[*] Testing URL: {test_url}")
                try:
                    response = requests.get(test_url, timeout=10)
                    print(f"Status code: {response.status_code}")
                except Exception as e:
                    print(f"[ERROR] Failed to get response: {e}")
                    writer.writerow([url, payload, f"Error: {e}", ""])
                    continue

                prediction = scanner.classify_text(payload)
                print(f"[RESULT] Payload: {payload!r} --> {prediction}")
                writer.writerow([url, payload, response.status_code, prediction])

    print("\nScan complete. Exiting.")

if __name__ == "__main__":
    main()
