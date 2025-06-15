# 🔍🐞 Soft Assertion Fuzzer

> - **A smarter way to find numerical bugs in ML code.**
> - Powered by learned *soft assertions*, this fuzzer doesn’t just catch crashes — it uncovers hidden instabilities like NaNs, Infs, and silent mispredictions that break model reliability.
> - If your ML code is numerically unstable — **we will detect it. Automatically. Precisely. At scale.**

---

![Soft Assertion Fuzzer Banner](https://github.com/AnwarXahid/soft-assertion-fuzzer/blob/main/soft-assertion-fuzzer-banner.png)
*“Detect what others miss. Fix what others ignore.”*

---

## 📋 Table of Contents

- [📚 What is This?](#what-is-this)
- [📌 Abstract](#abstract)
- [🎯 Motivation](#motivation)
- [🚀 Key Features](#key-features)
- [🧠 How It Works](#how-it-works)
- [🧪 Evaluation](#evaluation)
- [🔧 Prerequisites](#prerequisites)
- [🛠️ Installation](#installation)
- [🎮 Usage Modes](#usage-modes)
- [🚀 Quick Start](#quick-start)
- [🧩 Project Structure](#project-structure)
- [🤖 Extend the Tool](#extend-the-tool)
- [🤝 Contributing](#contributing)
- [📜 Citation](#citation)
- [⚖️ License](#license)

---

## 📚 What Is This?

**Soft Assertion Fuzzer** is a precision fuzz-testing framework for detecting *numerical instability* in machine learning (ML) applications. It uniquely combines **pretrained ML-based assertions** with **gradient-guided input mutation**, enabling it to uncover bugs that silently corrupt outputs — such as NaNs, Infs, and **plausible-but-wrong predictions**.

Unlike conventional fuzzers, it doesn't rely on brute-force or coverage alone. Instead, it strategically navigates the input space using soft assertions — lightweight ML models trained to signal when and how numerical errors might occur.

> Designed for deep learning frameworks like **PyTorch** and **TensorFlow**, this tool is built to break brittle code with scientific precision.

**Highlights:**

- ✨ **Targets unstable math operations** like `exp()`, `log()`, `softmax()`, `matmul()`, etc.
- 🧠 **Uses pretrained ML models** to guide fuzzing toward failure-prone regions.
- 📉 **Applies gradient-based mutation**, not just random noise.
- 🔬 **Validates failures using six oracles**, beyond simple NaN checks.
- 📈 **Outperforms five SOTA fuzzers** on benchmarks and real-world applications.

> 📄 **FSE 2025 Paper**:  
> _Automatically Detecting Numerical Instability in Machine Learning Applications via Learned Soft Assertions_  
> 📜 [Read on arXiv](https://arxiv.org/pdf/2504.15507) · 📦 [Replication Package](https://figshare.com/s/6528d21ccd28bea94c32)

---

![Soft Assertion Fuzzer Illustration](https://github.com/AnwarXahid/soft-assertion-fuzzer/blob/main/soft-assertion-fuzzer-tool.png)  
*Illustration: Soft Assertions guide ML fuzzing to expose hidden numerical instabilities.*

---
## 📌 Abstract

**Soft Assertion Fuzzer** is an automated testing framework tailored for detecting numerical instability in Machine Learning (ML) programs. Unlike traditional fuzzers that rely on random or syntactic mutations, this tool leverages *pretrained machine learning models*—called **Soft Assertions**—to identify instability-prone computations and guide input mutations toward failure-inducing conditions. It uncovers issues such as silent prediction errors, NaNs, and Infs in numerical code that are often missed by conventional testing tools.

---

## 🎯 Motivation

Modern ML applications heavily rely on floating-point computations over large or sensitive numerical ranges. Small perturbations in input values or model weights can lead to severe instabilities, affecting both correctness and performance. However, such issues often go undetected during standard validation and deployment workflows.

**Key challenges in existing tools:**

- ❌ Limited to random input fuzzing or shallow heuristics.
- ❌ Lack domain-specific knowledge of numerical behavior in ML.
- ❌ Fail to detect non-crashing but semantically incorrect outputs.

**Soft Assertion Fuzzer is designed to overcome these by:**

- ✅ Learning failure-inducing behavior through supervised training on unit test data.
- ✅ Using ML-based classifiers (Soft Assertions) to guide gradient-informed mutations.
- ✅ Integrating multiple runtime oracles for fine-grained instability detection.

---

## 🚀 Key Features

- **Soft Assertions**: Pretrained models that predict instability regions for ML operators.
- **AST-Based Hooking**: Automatically instruments ML scripts for fuzzing.
- **Gradient-Guided Mutation**: Applies autodiff to refine input generation.
- **Oracle-Based Validation**: Employs semantic oracles (NaN/Inf, incorrect class, etc.).
- **Failure Tracing & Logging**: Captures inputs, timeouts, and full function traces.

---

## 🧠 How It Works

```text
ML Program
   ↓
[AST Scanner] → Identifies unstable numerical functions
   ↓
[Soft Assertion Model] → Predicts which inputs can trigger instability
   ↓
[Auto-Differentiation] → Computes mutation directions via gradient signals
   ↓
[Oracle Evaluation] → Validates outcome (e.g., NaN, Inf, semantic misclassification)
   ↓
[Structured Logging] → Reports root cause, trigger inputs, and function call metadata
```

---

## 🧪 Evaluation

| Dataset              | Programs | Bugs Found | Time/Program |
|----------------------|----------|------------|--------------|
| GRIST Benchmark      | 79       | ✅ 79/79    | ⏱️ 0.646 sec |
| GitHub ML Projects   | 15       | ✅ 12/15    | ⏱️ 1.92 sec  |

> 🏆 Soft Assertion Fuzzer uncovered critical bugs missed by Hypothesis, PyFuzz, GRIST, Atheris, and RANUM.

---

## 🔧 Prerequisites

- Python 3.10+
- pip
- Git
- 4GB RAM minimum (8GB+ recommended)
- Works on Linux, macOS, Windows

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/soft-assertion-fuzzer.git
cd soft-assertion-fuzzer
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Check installation:

```bash
softassertion-cli --help
```

> 📝 Note:
> - Installation typically takes 2-5 minutes to complete. If you encounter "command not found" errors, please wait for the installation to finish completely, then restart your terminal or re-activate your virtual environment.

---

## 🎮 Usage Modes

```bash
# Standard usage
softassertion-cli my_script.py

# Custom config
softassertion-cli my_script.py --config config/custom.yaml

# Verbose and timed
softassertion-cli my_script.py --verbose --timeout 60
```
---

## 🚀 Quick Start

Let’s walk through using Soft Assertion Fuzzer on a real example.

### 🧪 Step 1: Start with a simple ML script

Suppose you have a script named `test_fuzzer.py`:

```python
# test_fuzzer.py
import torch

def test():
    
    x = torch.rand((3, 3))
    y = torch.nn.functional.softmax(x, dim=1)
    print("Softmax output:\n", y)

    z = torch.rand((3, 3)) - 0.5  
    w = torch.log(z)
    print("Log output:\n", w)

if __name__ == '__main__':
    test()
```

This script **runs** — but silently generates `nan` values. No error is thrown.  
This is where **numerical instability** hides.

---

### ⚙️ Step 2: Instrument the script for fuzzing

Now, modify `test_fuzzer.py` by adding Soft Assertion Fuzzer hooks:

```python
# test_fuzzer.py (instrumented)
import torch
from softassertion.analysis.boundary_tracer import start_fuzz, end_fuzz         # import hooks

def test():
    start_fuzz()               # Start fuzzing scope

    x = torch.rand((3, 3))
    y = torch.nn.functional.softmax(x, dim=1)
    print("Softmax output:\n", y)

    z = torch.rand((3, 3)) - 0.5  
    w = torch.log(z)
    print("Log output:\n", w)

    end_fuzz()                  # End fuzzing scope

if __name__ == '__main__':
    test()
```

> 🧠 `start_fuzz()` and `end_fuzz()` mark the **region of interest**. 
> Everything in between is monitored by the fuzzer — especially calls to known **unstable functions** like `log`, `exp`, `softmax`, etc.

---

### ▶️ Step 3: Run the fuzzer

Run Soft Assertion Fuzzer on your script:

```bash
softassertion-cli test_fuzzer.py
```

Behind the scenes:

- It finds `torch.log()` as an unstable function.
- It queries the pretrained soft assertion model for `log`.
- It mutates the inputs using gradients (auto-diff).
- It triggers numerical instability (e.g., `nan`) based on oracle signals.
- It logs failure inputs, timestamps, and outputs.

---

### 📁 Step 4: View Results <!-- 🔽 UPDATED SECTION START -->

All logs and reports are stored under:

```bash
experiments/logs/
```

You’ll find:

| File                          | Description                                 |
|-------------------------------|---------------------------------------------|
| `log_<function>.txt`          | Failure or timeout log for each function    |
| `summary_report.json`         | JSON summary of all results                 |
| `trace.log`                   | Execution timeline (step-by-step trace)     |
| `inputs_failed.npy` (optional)| Serialized failure-triggering inputs        |

> 🧪 Example file: `experiments/logs/log_log.txt`

---

### 📝 Step 5: Sample Failure Output

Here’s what a **real bug discovery** might look like:

#### 📦 Console Output
```bash
[FuzzRunner] ❌ Bug triggered by input:
(tensor([[62.8839, 63.8055, 69.9051],
        [16.3082, 69.6523, -8.1343],
        [52.0661, 17.2648, 30.0087]], requires_grad=True),)
[FuzzRunner] ✅ Done: No failure found for: softmax

📊 Case Study Summary:
  Total Cases: 2
  Passed: 1
  Failed: 1
  Average Time (sec): 5.0
  Failure Rate: 50.0
```

#### 📄 Summary JSON (`summary_report.json`)
```json
[
  {
    "function": "torch.log",
    "status": "FAIL",
    "oracle": "NaN/INF",
    "input": [[62.8839, 63.8055, 69.9051], [16.3082, 69.6523, -8.1343], [52.0661, 17.2648, 30.0087]],
    "model_signal": "no_change",
    "severity": "high",
    "elapsed_time": 5.0
  },
  {
    "function": "torch.softmax",
    "status": "PASS",
    "elapsed_time": 10.0
  }
]
```

---

## 🧩 Project Structure

Below is a high-level overview of the **Soft Assertion Fuzzer** directory layout, highlighting key components of the project. This will help contributors and users understand where to find code, configurations, models, and results.

```text
.
├── softassertion/                  # Core fuzzer logic and components
│   ├── analysis/                   # AST parsing and hook injection
│   ├── assertions/                 # Training pipeline and oracle logic
│   ├── config/                     # Default configuration YAMLs
│   ├── engine/                     # Fuzzing execution engine
│   ├── fuzzing/                    # Autodiff-based mutators and history tracker
│   ├── utils/                      # Shared utilities (logging, enums)
│   ├── cli.py                      # Command-line entry point
│   └── main.py                     # Tool bootstrapper
│
├── oracles/                        # Function-specific numerical checkers
├── generators/                     # Input generators (random, targeted)
├── scripts/                        # Ready-to-fuzz ML scripts (15 real-world cases)
├── experiments/                    # Logs and reports from fuzzing runs
│   ├── bugs/                       # Confirmed bug-triggering cases
│   ├── logs/                       # Fuzzer execution traces
│   └── reports/                    # Summaries and metrics
│
├── resources/                      # Pretrained models, datasets, figures
│   ├── models/                     # Trained SA models for each function
│   ├── dataset/                    # Evaluation results (XLSX)
│   └── images/                     # Visual diagrams used in docs
│
├── docs/                           # Project documentation
│   ├── design.md                   # Architecture and internals
│   ├── extend.md                   # How to add new functions
│   └── usage.md                    # Walkthrough and screenshots
│
├── tests/                          # Unit tests for major components
├── run_models.py                   # Utility to train or test assertion models
├── requirements.txt                # Python dependencies
├── setup.py                        # Installation metadata
├── LICENSE                         # MIT License
├── README.md                       # Main project documentation
└── soft-assertion-fuzzer-tool.png  # Tool illustration for README or docs
```

> 📝 Note:
> - All real-world cases used in the FSE 2025 evaluation are located in `scripts/`.
> - Pretrained models (e.g., `exp_dataset_model_v100.pth`, `relu_model_rf.joblib`) are organized by input shape under `resources/models/`.
> - Additional diagrams and presentation assets live in `resources/images/`.


---

## 🤖 Extend the Tool

Want to support your own function?

1. Write an oracle → `oracles/oracle_myfunc.py`
2. Generate failure/pass dataset
3. Train a soft assertion model (RandomForestClassifier or others)
4. Plug into registry
5. Fuzz it. Crash it. Fix it.

---

## 🤝 Contributing

```bash
pip install -r dev-requirements.txt
pre-commit install
pytest
```

---

## 📜 Citation

```bibtex
@inproceedings{sharmin2025automatically,
  title={Automatically Detecting Numerical Instability in Machine Learning Applications via Soft Assertions},
  author={Sharmin, Shaila and Zahid, Anwar Hossain and Bhattacharjee, Subhankar and Igwilo, Chiamaka and Kim, Miryung and Le, Wei},
  journal={arXiv preprint arXiv:2504.15507},
  year={2025}
}
```

---

## ⚖️ License

MIT License  
© 2025 Anwar Hossain Zahid, Iowa State University

---



