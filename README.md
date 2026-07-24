# SCARF and Missingness in Tabular Data

This project compares complete-case analysis, median and KNN imputation, and a
compact SCARF-style representation learner under tabular missingness and label
scarcity.

## Repository layout

- `src/`: data loading, missingness generation, models, training, and evaluation
- `experiments/`: experiment driver
- `Data/`: input datasets
- `results/`: recorded metrics and generated figures

## Setup and data

Create and activate a Python environment, then install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the UCI credit-card dataset at `Data/Raw/default_credit_card.csv`, the
Telco churn dataset at `Data/Raw/telco_clean.csv`, and the Adult dataset at
`Data/Raw/adult_income_dataset.csv`.

Run the experiment driver with:

```bash
python experiments/run_baselines.py
```

Existing conditions in `results/results.csv` are skipped. Data splits use seed
42, but missingness injection, corruption sampling, and network initialisation
are not individually seeded, so exact deterministic reproduction is not
claimed.

Regenerate the label-scarcity figure with:

```bash
python results/plot_label_scarcity.py
```

Metrics are stored in `results/results.csv` and figures in `results/`. The Telco
0.5% median and KNN entries are blank because an implementation audit
invalidated those historical outputs.
