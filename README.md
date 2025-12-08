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
   git clone XXXXXXX/CINDI.git
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

## 📊 Data Preparation

CINDI expects multivariate time series data. The default data loader supports:

1. **Grid Loss Data:** Real-world power consumption and grid loss measurements.
2. **FSB (Fully Synthetic Benchmark):** Synthetic sequences from the mTADS repository.

To use your own data, ensure it is formatted as a CSV with time-indexed rows and feature columns.
> columns = ['timestamp', 'value-0', 'value-1', ..., 'value-n', 'is_anomaly']

## 🏃 Usage

### 1. Basic Training & Imputation Loop

To run the full CINDI pipeline (Train $\to$ Detect $\to$ Impute $\to$ Repeat).

[CINDI Pipeline Illustration](images/FlowImputationOverview.pdf)

### 2. Anomaly Detection (Downstream Task)

After improving the dataset, it runs the anomaly detection on the test set.

```shell
python experiment_run.py --project="selective_loop_v1-fsb" --dataset="fsb" 
    --experiment="fixing_selective_loop_v1_fsb"
    --model_type="tcNF-base" --max_past_range=51 --self_optimization=True
    --shadow_channels=False --sanity_check=True --use_checks_in_optimization=True
    --code_configuration={\"st_linear\":true,\"batch_norm\":\"none\",\"input_embedding\":\"none\"}
```

### Configuration

See `global_utils.py:80` all available configuration options.


## 📚 Citation

If you use CINDI in your research, please cite our paper:

```
@article{XXXXXXX2025cindi,
    title={CINDI: Conditional Imputation and Noisy Data Integrity with Flows in Power Grid Data},
    author={XXXXXXX},
    journal={Engineering Applications of Artificial Intelligence},
    year={2025},
    note={Preprint}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements


This work was carried out XXXXXXX. 
Special thanks to Aneo for providing the grid loss prediction dataset.




