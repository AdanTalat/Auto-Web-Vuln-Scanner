# Automated Web Vulnerability Scanner (AI-Powered)

**AI-powered prototype that tests web endpoints with crafted payloads and uses a fine-tuned transformer classifier to flag responses as *Malicious* or *Benign*.**

This README explains what the project contains, how to train and use the model, how to run the scanner, and practical tips (checkpoints, resuming, quantization, Tiny/MobileBERT). Read the **Disclaimer** before testing anything.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Run Scanner (Inference)](#run-scanner-inference)
  - [Train / Resume Training](#train--resume-training)
  - [Evaluate & Tune Threshold](#evaluate--tune-threshold)
- [Important Scripts Explained](#important-scripts-explained)
- [Tips: Checkpoints, Batching, Token Limits](#tips-checkpoints-batching-token-limits)
- [Speed & Deployment Options](#speed--deployment-options)
- [Safety & Legal Disclaimer](#safety--legal-disclaimer)
- [License](#license)
- [Credits](#credits)

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


## Safety & legal disclaimer

**Important:** This tool actively sends crafted payloads to web servers. You must **only** test systems you own or have explicit written permission to scan. Unauthorized testing is illegal and unethical. Use this project for education, research and authorized assessments only.

---

## License

This project is licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Credits

* **Credits:** Hugging Face Transformers, Datasets, PyTorch, scikit-learn, OWASP guidelines.

