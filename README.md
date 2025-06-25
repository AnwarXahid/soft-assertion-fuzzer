# 🔍🐞 Soft Assertion Fuzzer

> - **A smarter way to find numerical bugs in ML code.**
> - Powered by learned *soft assertions*, this fuzzer doesn’t just catch crashes — it uncovers hidden instabilities like NaNs, INFs, and incorrect outputs that break model reliability.
> - If your ML code is numerically unstable — **we will detect it. Automatically. Precisely. At scale.**

---

![Soft Assertion Fuzzer Banner](https://github.com/AnwarXahid/soft-assertion-fuzzer/blob/main/soft-assertion-fuzzer-banner.png)

---

## 📋 Table of Contents

- [📚 What is This?](#-what-is-this)
- [🎯 Overview & Key Features](#-overview--key-features)
- [🧪 Evaluation](#-evaluation)
- [🔧 Prerequisites](#-prerequisites)
- [🛠️ Installation](#-installation)
- [🎮 Usage Modes](#-usage-modes)
- [🚀 Quick Start](#-quick-start)
- [🧩 Project Structure](#-project-structure)
- [🤖 Extend the Tool](#-extend-the-tool)
- [🤝 Contributing](#-contributing)
- [📜 Citation](#-citation)
- [⚖️ License](#-license)

---

## 📚 What Is This?

**Soft Assertion Fuzzer** is a precision fuzz-testing framework for detecting *numerical instability* in machine learning (ML) applications. It uniquely combines **pretrained ML-based assertions** with **gradient-guided input mutation**, enabling it to uncover bugs that silently corrupt outputs — such as NaNs, Infs, and wrong outputs.

Unlike conventional fuzzers, it doesn't rely on brute-force or coverage alone. Instead, it strategically navigates the input space using soft assertions — lightweight ML models trained to signal when and how numerical errors might occur.

> Designed for deep learning frameworks like **PyTorch** and **TensorFlow**, this tool is built to detect subtle numerical instability in ML code with precision.

> 📄 **FSE 2025 Paper**:  
> _Automatically Detecting Numerical Instability in Machine Learning Applications via Learned Soft Assertions_  
> 📜 [Read on arXiv](https://arxiv.org/pdf/2504.15507) · 📦 [Replication Package](https://figshare.com/s/6528d21ccd28bea94c32)

---

![Soft Assertion Fuzzer Illustration](https://github.com/AnwarXahid/soft-assertion-fuzzer/blob/main/soft-assertion-fuzzer-tool.png)  
> **Illustration:** *Soft Assertions guide ML fuzzing to expose hidden numerical instabilities.*

---

## 🎯 Overview & Key Features

**Soft Assertion Fuzzer** is an automated testing framework tailored for detecting numerical instability in Machine Learning (ML) programs. Modern ML applications heavily rely on floating-point computations over large or sensitive numerical ranges, where small perturbations in input values or model weights can lead to severe instabilities, affecting both correctness and performance. However, such issues often go undetected during standard validation and deployment workflows.

### The Problem with Existing Tools

Traditional testing approaches fall short when dealing with ML numerical instability:

- ❌ **Limited scope**: Random input fuzzing or shallow heuristics miss complex failure patterns
- ❌ **No domain knowledge**: Lack understanding of numerical behavior specific to ML operations  
- ❌ **Silent failures**: Fail to detect non-crashing but semantically incorrect outputs
- ❌ **Inefficient exploration**: Cannot strategically navigate high-dimensional input spaces

### Our Solution: ML-Guided Fuzzing

Unlike traditional fuzzers that rely on random or syntactic mutations, this tool leverages *pretrained machine learning models*—called **Soft Assertions**—to identify instability-prone computations and guide input mutations toward failure-inducing conditions. It uncovers issues such as silent prediction errors, NaNs, and Infs in numerical code that are often missed by conventional testing tools.

### 🚀 Key Features

- **✨ Soft Assertions**: Pretrained ML models that predict instability regions for common ML operators like `exp()`, `log()`, `cosine_similarity()`, `matmul()`, etc.

- **🧠 Intelligent Targeting**: Uses supervised learning on unit test data to understand failure-inducing behavior patterns.

- **📉 Gradient-Guided Mutation**: Applies autodifferentiation to compute optimal mutation directions, not just random noise.

- **🔬 Multi-Oracle Validation**: Employs six different runtime oracles for fine-grained instability detection beyond simple NaN/INF checks.

- **⚙️ AST-Based Hooking**: Automatically instruments ML scripts for fuzzing without manual code modification.

- **📈 Superior Performance**: Outperforms five state-of-the-art fuzzers (Hypothesis, PyFuzz, GRIST, Atheris, RANUM) on benchmarks and real-world applications.

- **🔍 Comprehensive Logging**: Captures failure-triggering inputs, timeouts, and full function execution traces for root cause analysis.

**Why This Matters**: This framework bridges the gap between traditional software testing and the unique challenges of ML numerical computing, enabling developers to catch subtle bugs that could compromise model reliability in production.

---

## 📊 Evaluation Results

### Performance Comparison with State-of-the-Art

| Fuzzer | GRIST (79) | GRIST Avg Time (sec) | Real-World (15) | NaN/INF | Other Failures |
|--------|------------|---------------------|-----------------|---------|----------------|
| **SA Fuzzer** | **79** | **0.646** | **13** | **88** | **4** |
| RANUM [28] | 79 | 2.209 | 3 | 82 | 0 |
| GRIST [47] | 78 | 44.267 | 3 | 81 | 0 |
| Atheris [31] | 25 | 0.283 | 3 | 27 | 1 |
| PyFuzz [1] | 24 | 5.498 | 3 | 26 | 1 |
| Hypothesis [41] | 23 | 5.945 | 3 | 25 | 1 |

**Key Findings:**
- **Superior Detection**: SA Fuzzer achieves the highest detection rates across all categories
- **Fastest Execution**: 0.646 seconds average time, 4x faster than the next best performer
- **Real-World Effectiveness**: Finds 13/15 real-world bugs vs. 3/15 for other tools
- **Comprehensive Coverage**: Detects both NaN/INF failures (88) and other numerical instabilities (4)
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
source venv/bin/activate 
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
├── oracles/                        # Function-specific numerical checkers
├── generators/                     # Input generators (random, targeted)
├── scripts/                        # Ready-to-fuzz ML scripts (15 real-world cases)
├── docs/                           # Project documentation
├── tests/                          # Unit tests for major components
├── run_models.py                   # Utility to run assertion models
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

## ✍️ Author

**Name:**  Anwar Hossain Zahid

**Email:** ahzahid@iastate.edu

**Phone:** +1-515-766-1601

**github:**: https://anwarxahid.github.io/

---



