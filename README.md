# 🧪 Soft Assertion Fuzzer

Soft Assertion Fuzzer is an automated tool that leverages pre-trained ML models (Soft Assertions) to trigger and detect numerical instability in PyTorch-based ML applications. It intelligently mutates inputs to uncover conditions leading to NaNs, Infs, or incorrect outputs — beyond just crashes.

**Soft Assertion Fuzzer** implements the technique from our [FSE 2025 paper](https://arxiv.org/pdf/2504.15507)  
**"Automatically Detecting Numerical Instability in Machine Learning Applications via Soft Assertions"**.

---

Machine learning (ML) applications rely heavily on floating-point arithmetic. They often operate on extremely large or small values, making them vulnerable to **numerical instability** — silent bugs that can cause incorrect outputs, wasted resources, or even model failures.

We introduce a novel technique called **Soft Assertions**, which are **learned numerical safety models** trained during unit testing of unstable functions (e.g., `exp`, `log`, `softmax`). Each soft assertion predicts how to **mutate inputs to trigger instability**.

Given an ML script:
- Our tool scans for known unstable functions
- Inserts monitoring hooks
- Uses pretrained soft assertions to guide gradient-based mutations
- Detects NaN, INF, or incorrect outputs
- Logs failure-inducing inputs and safety violations

This approach outperforms 5 state-of-the-art fuzzers. It also found previously unknown numerical bugs in 15 real-world GitHub ML projects.

📎 Full paper:  
Sharmin, Zahid, Bhattacharjee, Igwilo, Kim, Le  
**“Automatically Detecting Numerical Instability in Machine Learning Applications via Soft Assertions”**,  
*FSE 2025, ACM*  
https://arxiv.org/pdf/2504.15507


### Key Features
- 🚨 Detects hidden numerical bugs using learned boundary models
- 🤖 Supports over 20+ PyTorch operations (exp, relu, log, softmax, matmul, etc.)
- 📊 Logs failure-inducing inputs and timings in `experiments/logs/`
- 🧠 Leverages gradient-based mutation + pretrained soft assertion models
- ✅ Works on arbitrary scripts with `start_fuzz()` and `end_fuzz()` hooks
- ⚙️ Configurable via `config/default.yaml`

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourname/soft-assertion-fuzzer.git
cd soft-assertion-fuzzer
```

### 2. Set up Python (we recommend 3.10)

```bash
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧭 Usage

### Add soft assertion hooks in your script:

```python
from softassertion.analysis.boundary_tracer import start_fuzz, end_fuzz

start_fuzz()

# Your ML code here, using functions like torch.exp, relu, etc.

end_fuzz()
```

### Run fuzzing on the script:

```bash
python -m softassertion.cli scripts/my_test_script.py
```

This will:
- Detect numerical instability
- Report failure-inducing inputs
- Track timing and safety oracle results
- Log results to `experiments/logs/`

---

## 🧪 Running Tests

To run all unit tests (recommended after setup):

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest tests/
```

---

## 📦 Current Features

- ✅ PyTorch support: `exp`, `relu`, `log`, `softmax`, `sqrt`, `matmul`, etc.
- ✅ Fuzzing based on symbolic gradients
- ✅ NaN-guarding oracles for safety assertion
- ✅ AST-based region identification using `start_fuzz()` / `end_fuzz()`
- ✅ Logging of input triggers and fuzzing duration

---

## 📁 Directory Structure

```
softassertion/
├── analysis/           # AST parsing, boundary markers
├── engine/             # fuzz_runner, input/oracle mapping
├── utils/              # config, enums
├── cli.py              # CLI entry point

oracles/                # Safety oracles (one per function)
generators/             # Input generators (torch-based)
scripts/                # Sample test scripts
experiments/logs/       # Auto-saved fuzzing results
```

---

## 📜 License

MIT License © 2025 Anwar Hossain Zahid, Iowa State University

---

## 📝 Citation
If you use this tool in academic work, please cite:

```
@inproceedings{sharmin2025automatically,
  title={Automatically Detecting Numerical Instability in Machine Learning Applications via Soft Assertions},
  author={Sharmin, Shaila and Zahid, Anwar Hossain and Bhattacharjee, Subhankar and Igwilo, Chiamaka and Kim, Miryung and Le, Wei},
  journal={arXiv preprint arXiv:2504.15507},
  year={2025}
}
```

---

## 📦 Installation

You can make this tool installable using:

```bash
pip install -e .
hash -r          

```

This will allow direct CLI execution via:
```bash
softassertion-cli scripts/my_test_script.py
```

>Note: Installation may take a few moments to complete. Please wait a couple of seconds before running commands directly from the CLI.





