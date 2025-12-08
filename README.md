# CINDI: Conditional Imputation and Noisy Data Integrity

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](NotAvailalbeYet.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-orange)](https://pytorch.org/)

**CINDI** is an unsupervised probabilistic framework designed to restore data integrity in complex multivariate time
series. It unifies anomaly detection and imputation into a single end-to-end system
built on **Conditional Normalizing Flows**.

## 🚀 Key Features

* **Unified Framework:** Integrates detection, correction, and training into a single end-to-end framework.
* **Conditional Normalizing Flows:** Uses RealNVP-based flows to model complex temporal dependencies.
* **Iterative Improvement:** Alternates between training on current data and fixing "noisy" data until convergence.
* **Model Selection via CMA-ES:** Automatically searches for optimal hyperparameters using evolutionary strategies.
* **Flexible Encoders:** Supports multiple temporal context encoders (`Base`, `MLP`, `CNN`).

## 🛠️ Installation

1. **Clone the repository:**
   ```
   git clone …/CINDI.git
   cd CINDI
   ```

2. **Create a virtual environment (recommended):**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

   *Key dependencies include: `torch`, `numpy`, `pandas`, `cma` (for CMA-ES), and `matplotlib`.*

## 📂 Project Structure

```
CINDI/
├── dataset/                # Datasets for training and evaluation
├── models/                 # Model architectures and training scripts
├── vus/                    # Scoring code for evaluation
├── summary/                # Summary statistics and results
├── experiment_run.py       # Main script to run different experiments 
├── execute_*.py            # Scripts to execute specific tasks (training, evaluation, etc.)
├── …                       # Utility functions and helpers
├── requirements.txt        # List of required Python packages
└── README.md               # Project documentation
```


