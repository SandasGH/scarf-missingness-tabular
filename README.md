# SCARF and Missingness in Tabular Data

## Project overview

This archive contains the source code, experiment scripts, retained results and
supporting artefacts for the MSc Artificial Intelligence project:

**Improving Robustness of Tabular Classifiers Under Controlled Missingness Using
Self-Supervised Pretraining**

The project compares four strategies for handling missing tabular data:

1. Complete case analysis
2. Median imputation
3. K-nearest-neighbour imputation
4. SCARF-style self-supervised contrastive pretraining

The methods are evaluated under cell-level MCAR missingness, structured block
missingness and varying levels of label availability.

## Archive contents

### Root files

- `README.md`  
  Describes the archive structure, software requirements, dataset locations,
  execution procedure and reproducibility limitations.

- `AUTHORSHIP_DECLARATION.txt`  
  Contains the required declaration certifying authorship of the submitted
  programs.

- `requirements.txt`  
  Lists the Python dependencies required to run the project.

- `.gitignore`  
  Identifies local data, environments, caches and generated files that should
  not be committed to version control.

### `src/`

Contains the main reusable implementation modules.

- `src/data.py`  
  Loads and preprocesses the UCI Credit Card Default, Telco Customer Churn and
  UCI Adult datasets. It also implements the controlled MCAR and structured
  block missingness mechanisms.

- `src/models.py`  
  Defines the baseline multilayer perceptron and the neural-network components
  used by the SCARF pipeline.

- `src/train.py`  
  Implements supervised MLP training, validation, early stopping and
  evaluation.

- `src/baselines.py`  
  Implements complete case analysis, median imputation and KNN imputation.

- `src/scarf.py`  
  Implements SCARF feature corruption, contrastive pretraining, classification-
  head training and downstream evaluation.

### `experiments/`

- `experiments/run_baselines.py`  
  Main experiment driver. It runs the clean baseline, MCAR experiments,
  structured block missingness experiments and label-scarcity experiments for
  the applicable datasets and methods.

### `Data/`

- `Data/download_data.py`  
  Data-preparation and inspection helper used to prepare the raw dataset files
  and inspect their dimensions, missing values and class distributions.

- `Data/Raw/`  
  Location expected by the data-loading code for the raw or prepared datasets.

The expected filenames are:

- `Data/Raw/default_credit_card.csv`
- `Data/Raw/telco_clean.csv`
- `Data/Raw/adult_income_dataset.csv`

The datasets are publicly available from the UCI Machine Learning Repository
and IBM sample-data resources, as referenced in the accompanying report.

### `notebooks/`

Contains exploratory notebooks used during dataset inspection, preprocessing,
correlation analysis and experimental development. The main reported
experiments are executed through `experiments/run_baselines.py`; the notebooks
are not required to run the main experiment driver.

### `results/`

- `results/results.csv`  
  Retained AUC and accuracy results for the experimental conditions reported in
  the thesis.

- `results/plot_label_scarcity.py`  
  Reads `results/results.csv` and regenerates the label-scarcity AUC figure.

- `results/label_scarcity_auc.png`  
  Generated figure comparing median imputation, KNN imputation and SCARF under
  varying levels of label availability.

- `results/gantt_chart.py`  
  Script used to generate a project-planning Gantt chart. It is not required
  for the experimental pipeline.

Other files in this directory are generated figures or supporting result
artefacts produced during the project.

## Software requirements

The experiments were developed using Python 3.13.2.

Create and activate a Python environment, then install the required
dependencies:

```bash
python -m pip install -r requirements.txt
```

The principal dependencies include PyTorch, scikit-learn, NumPy, pandas and
Matplotlib.

## Running the experiments

From the root of the archive, run:

```bash
python experiments/run_baselines.py
```

The experiment driver runs the configured conditions and writes metrics to:

```text
results/results.csv
```

Existing conditions already present in `results/results.csv` are skipped. To
rerun the complete experimental grid from the beginning, first retain a backup
of the submitted results and then remove or rename the existing
`results/results.csv` file.

## Regenerating the label-scarcity figure

Run:

```bash
python results/plot_label_scarcity.py
```

The generated figure is written to:

```text
results/label_scarcity_auc.png
```

## Reproducibility notes

Train/test splits, validation splits and labelled-subset selection use random
seed 42.

Missingness injection, SCARF corruption sampling and neural-network weight
initialisation are not individually seeded. Exact decimal-for-decimal
reproduction is therefore not claimed; repeated runs reproduce the experimental
procedure and expected behaviour rather than necessarily the identical reported
values.

The submitted `results/results.csv` file contains the retained results used in
the thesis.

## Excluded result

The Telco 0.5% median-imputation and KNN-imputation values are blank in
`results/results.csv`.

An implementation audit found that, when fewer than 50 labelled rows were
available, the no-validation training path returned the baseline MLP's initial
weights rather than its trained weights. Those two historical outputs were
therefore invalid and were excluded from the report.

## Hardware

The experiments were run on a consumer laptop with:

- AMD Ryzen 5 4500U processor
- 16 GB RAM
- Integrated graphics
- CPU-only PyTorch execution

No dedicated GPU is required to execute the project.
