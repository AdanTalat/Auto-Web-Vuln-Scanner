# Automated Web Vulnerability Scanner (AI-Powered)

**AI-powered prototype that tests web endpoints with crafted payloads and uses a fine-tuned transformer classifier to flag responses as *Malicious* or *Benign*.**

This README explains what the project contains, how to train and use the model, how to run the scanner, and practical tips (checkpoints, resuming, quantization, Tiny/MobileBERT). Read the **Disclaimer** before testing anything.

---

## Table of contents

* [Project overview](#project-overview)
* [Folder structure](#folder-structure)
* [Requirements](#requirements)
* [Quick start](#quick-start)

  * [Run scanner (inference)](#run-scanner-inference)
  * [Train / resume training](#train--resume-training)
  * [Evaluate & tune threshold](#evaluate--tune-threshold)
* [Important scripts explained](#important-scripts-explained)
* [Tips: checkpoints, batching, token limits](#tips-checkpoints-batching-token-limits)
* [Speed & deployment options](#speed--deployment-options)
* [Safety & legal disclaimer](#safety--legal-disclaimer)
* [License & credits](#license--credits)

---

## Project overview

This project aims to provide a working prototype that:

1. Fine-tunes a transformer (DistilBERT by default) on labeled HTTP request/response data.
2. Saves checkpoints while training (so you can resume).
3. Provides a `Scanner.py` utility that:

   * Injects common attack payloads into a target URL,
   * Sends the requests in real time,
   * Feeds the responses (or the payload/request string) to the classifier,
   * Logs results (`scan_results.csv`) and prints a verdict.

The model can be further optimized for inference by switching to TinyBERT/MobileBERT or by applying post-training quantization.

---

## Folder structure

```
WEB_VULNERABILITY_SCANNER/
├─ BOT/
│  ├─ Data/
│  │  ├─ equal_dataset.csv           # balanced dataset used for training
│  │  └─ ...                         # other data files
│  ├─ Model/                         # trainer output: checkpoints and final model
│  │  ├─ results/                    # HuggingFace Trainer checkpoint folder
│  │  │  ├─ checkpoint-500/
│  │  │  ├─ checkpoint-1000/
│  │  │  └─ ...
│  │  └─ distilbert_web_scanner/     # final exported model for scanner
│  ├─ Scanner.py                      # real-time scanner + inference code
│  ├─ trainer.py                      # training / resume script (Hugging Face Trainer)
│  ├─ tuning.py                       # evaluation + threshold tuning
│  ├─ requirements.txt
│  └─ README.md                       # this file
└─ README.md (root)
```

---

## Requirements

Create a virtual environment (recommended) and install:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r BOT/requirements.txt
```

Typical packages:

* `transformers`
* `datasets`
* `torch`
* `scikit-learn`
* `pandas`
* `requests`
* `matplotlib`

(Exact versions are in `BOT/requirements.txt`.)

---

## Quick start

### Run scanner (inference)

1. Put the trained model directory at `BOT/Model/distilbert_web_scanner` (this folder must contain `config.json`, `pytorch_model.bin`, tokenizer files, etc.).
2. Run:

```bash
cd BOT
python Scanner.py
```

3. Enter website URLs when prompted (type `exit` to quit).
   The scanner injects payloads, sends requests, logs status & result to `scan_results.csv`, and prints `Malicious`/`Benign`.

**Note:** Scanner by default classifies the *payload string* (if the model was trained on payloads) or the *response text* — check which your model was trained on and edit `Scanner.py` accordingly.

---

### Train / resume training

**1. Prepare dataset**

* Place CSV with `text` and `label` columns in `BOT/Data/`.

  * `text` = request text or response text depending on your training goal.
  * `label` = `0` (benign) or `1` (malicious).

**2. Train**

```bash
cd BOT
python trainer.py
```

`trainer.py` (provided) uses Hugging Face `Trainer` and will:

* Tokenize and split the dataset,
* Save checkpoints every `save_steps` (default: 500),
* Detect latest checkpoint in `./Model/results` and resume automatically if present,
* Save final model to `./Model/distilbert_web_scanner`.

**Resume manually** (if required):

* If checkpoint folders are in `BOT/Model/results/checkpoint-XXXX`, `trainer.py` will auto-detect.
* To resume explicitly:

```python
trainer.train(resume_from_checkpoint="BOT/Model/results/checkpoint-2500")
```

---

### Evaluate & tune threshold

By default the model uses `argmax` (threshold 0.5) to decide class. That may not be optimal.

Run the tuning script to compute a better threshold, view precision/recall and confusion matrix:

```bash
cd BOT
python tuning.py --data Data/equal_dataset.csv --model ./Model/distilbert_web_scanner
```

`tuning.py` outputs:

* Best threshold (based on F1 across precision-recall curve)
* Precision, recall, F1, confusion matrix
* Plots PR curve (requires display or saved figure)

**Integration:** After finding best threshold, update `Scanner.py` to use it:

```python
probs = torch.softmax(outputs.logits, dim=1)[:,1].item()
prediction = "Malicious" if probs >= BEST_THRESHOLD else "Benign"
```

---

## Important scripts explained

* **`trainer.py`** – Fine-tune DistilBERT (Hugging Face Trainer). Handles tokenization, checkpointing, resuming, saving final model.
* **`Scanner.py`** – Interactive scanner. Injects payloads, sends HTTP requests (GET), classifies either payload or response text, logs results.
* **`tuning.py`** – Evaluate model on held-out test set, compute precision-recall curve and best decision threshold.
* **`requirements.txt`** – Project dependencies.

---

## Tips: checkpoints, batching, token limits

* **Checkpoints**: `save_steps=500` means a checkpoint is saved after every 500 steps.
  With `per_device_train_batch_size=8`, each step processes 8 requests.
  So `checkpoint-500` = 500 × 8 = **4,000 requests seen**.

* **Steps per epoch** = `ceil(num_train_examples / batch_size)`.
  Example: 40,104 examples / 8 = 5,013 steps per epoch (so checkpoint-3000 means model has processed 24k requests).

* **Token limit**: DistilBERT max sequence length = **512 tokens**.

  * For inputs longer than 512 tokens, **truncate** or **split into chunks**.
  * For shorter inputs, use **padding** to make batch sizes uniform.

* **pin\_memory**: use in DataLoader when training on GPU to speed up CPU→GPU memory transfer. No benefit on CPU-only runs.

---

## Speed & deployment options

* **Use TinyBERT / MobileBERT** (huggingface hosts pretrained variants) for faster training and inference with minimal code changes:

  * Replace `model_name = "distilbert-base-uncased"` with a TinyBERT model id like `huawei-noah/TinyBERT_General_4L_312D`.
* **Quantization (post-training)**: After training, quantize model (INT8) for smaller size and faster inference (especially on CPU).

  * Tools: `torch.quantization`, `optimum`, or `onnxruntime` with INT8.
  * Example (post-training dynamic quantization):

    ```python
    import torch
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("BOT/Model/distilbert_web_scanner")
    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    quantized.save_pretrained("BOT/Model/distilbert_web_scanner_int8")
    ```
* **Deployment ideas**:

  * Wrap `Scanner.py` logic in a Flask/FastAPI service to accept requests and return predictions.
  * Add rate limiting, request queues, and logging for production use.
  * Respect legal constraints — only test sites you own or have explicit permission to test.

---

## Safety & legal disclaimer

**Important:** This tool actively sends crafted payloads to web servers. You must **only** test systems you own or have explicit written permission to scan. Unauthorized testing is illegal and unethical. Use this project for education, research and authorized assessments only.

---

## License & credits

* **License:** Add your preferred license (MIT/Apache-2.0).
* **Credits:** Hugging Face Transformers, Datasets, PyTorch, scikit-learn, OWASP guidelines.

---

## FAQ / Troubleshooting

* **Model says everything is malicious** → Check dataset balance, label mapping (ensure `1 = malicious` in training and inference), and run `tuning.py` to pick a threshold.
* **No checkpoints found on resume** → Ensure `training_args.output_dir` equals the folder where checkpoints are saved (`./Model/results` by default).
* **403 responses during scanning** → The site is blocking suspicious requests (WAF). A `403` means requests were blocked — not necessarily that the site is vulnerable.
* **Want to classify real responses instead of payloads?** Train the model on response text (HTML/JSON) rather than payload strings and update `Scanner.py` accordingly.

---

If you want, I can:

* Create a **short QuickStart.md** with the minimal commands, or
* Auto-generate a **GitHub Actions** workflow to run tests (no training in CI), or
* Produce a ready-to-use **Flask** wrapper for the scanner for demo purposes.

Which of these would you like next?
