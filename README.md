# Robustness of Tabular Classifiers Under Controlled Missingness Using Self-Supervised Pretraining

MSc Computer Science Thesis — King's College London, 2025–26
Student: Sanda Puce | K25085750
Supervisor: Kathleen Steinhofel

## Research Question

Does self-supervised contrastive pretraining (SCARF) improve the robustness of tabular 
classifiers under missing data, compared to classical imputation baselines?

## Key Insight

Traditional approaches to missing data attempt to reconstruct missing values numerically 
(mean, median, KNN imputation). SCARF takes a fundamentally different approach — it never 
imputes values at all. Instead, it pretrains an encoder to produce representations that are 
robust to feature absence, so that downstream classification is preserved even when data 
arrives incomplete at inference time. This thesis evaluates whether that approach outperforms 
classical imputation under realistic missingness conditions.

## Datasets

- **UCI Credit Card Default** — binary classification, numerical features
- **Telco Customer Churn** — binary classification, mixed numerical/categorical features

Both datasets are publicly available. See `data/download_data.py` for instructions.

## Methods Compared

| Condition | Missing Data Strategy | Classifier |
|---|---|---|
| Clean baseline | None (full data) | MLP |
| Complete case | Drop incomplete rows | MLP |
| Median imputation | Fill with column median | MLP |
| KNN imputation | Fill using k nearest neighbours | MLP |
| SCARF | Contrastive pretraining on incomplete data | MLP (frozen encoder + head) |

## Missingness Regimes

- MCAR (Missing Completely At Random): 10%, 20%, 30%
- Feature-dependent missingness: correlated feature groups removed together

## Evaluation Metrics

- AUC (primary — handles class imbalance)
- Accuracy
- Performance drop relative to clean baseline

## Repository Structure
```
├── data/               # data loading and download scripts (no raw data committed)
├── notebooks/          # exploratory data analysis
├── src/                # core source code
│   ├── data.py         # dataset loading, missingness injection
│   ├── models.py       # MLP classifier, SCARF encoder
│   ├── train.py        # training loops
│   ├── evaluate.py     # metrics
│   └── baselines.py    # imputation pipelines
├── experiments/        # scripts to reproduce each experiment
└── results/            # saved metrics and plots
```

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/scarf-missingness-tabular.git
cd scarf-missingness-tabular
pip install -r requirements.txt
```

## Academic Integrity

This repository contains my own work submitted for academic assessment at King's College 
London. All code and analysis is my own unless explicitly cited otherwise.
