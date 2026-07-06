
import os
import sys
import csv
import numpy as np


# Allow imports from project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _ROOT)

from sklearn.model_selection import train_test_split

from src.data import load_dataset, inject_missingness, _load_uci, _load_telco, ADULT_CORRELATED_GROUPS
from src.train import train_mlp, evaluate_mlp
from src.baselines import complete_case, median_imputation, knn_imputation
from src.scarf import pretrain_scarf, finetune_scarf, evaluate_scarf

RESULTS_DIR = os.path.join(_ROOT, "results")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "results.csv")

DATASETS = ["uci", "telco"]
MCAR_RATES = [0.10, 0.20, 0.30]


def run_clean_baseline(rows):
    for name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}  |  Condition: clean_baseline")
        print("=" * 60)

        X_train, X_test, y_train, y_test = load_dataset(name)
        input_dim = X_train.shape[1]

        print("\nTraining MLP...")
        model = train_mlp(X_train, y_train, input_dim)

        print("\nEvaluating on test set...")
        auc, acc = evaluate_mlp(model, X_test, y_test)

        rows.append(
            {"dataset": name, "condition": "clean_baseline", "AUC": auc, "accuracy": acc}
        )


def run_complete_case(rows):
    for name in DATASETS:
        for rate in MCAR_RATES:
            rate_pct = int(rate * 100)
            condition = f"complete_case_MCAR_{rate_pct}"

            print(f"\n{'='*60}")
            print(f"Dataset: {name}  |  Condition: {condition}")
            print("=" * 60)

            X_train, X_test, y_train, y_test = load_dataset(name)

            print(f"\nInjecting MCAR missingness at {rate_pct}% into X_train...")
            X_train_miss = inject_missingness(X_train, mechanism="MCAR", rate=rate)

            print("\nApplying complete-case analysis to X_train...")
            X_train_cc, y_train_cc = complete_case(X_train_miss, y_train)

            if len(X_train_cc) < 50:
                print(
                    "  WARNING: Too few rows remaining for reliable training"
                    " - skipping this condition"
                )
                rows.append({"dataset": name, "condition": condition, "AUC": None, "accuracy": None})
                continue

            input_dim = X_train_cc.shape[1]

            print("\nTraining MLP on complete-case training set...")
            model = train_mlp(X_train_cc, y_train_cc, input_dim)

            print("\nEvaluating on clean complete X_test...")
            auc, acc = evaluate_mlp(model, X_test, y_test)

            rows.append({"dataset": name, "condition": condition, "AUC": auc, "accuracy": acc})


def run_median_imputation(rows):
    for name in DATASETS:
        for rate in MCAR_RATES:
            rate_pct = int(rate * 100)
            condition = f"median_MCAR_{rate_pct}"

            print(f"\n{'='*60}")
            print(f"Dataset: {name}  |  Condition: {condition}")
            print("=" * 60)

            X_train, X_test, y_train, y_test = load_dataset(name)

            print(f"\nInjecting MCAR missingness at {rate_pct}% into X_train...")
            X_train_miss = inject_missingness(X_train, mechanism="MCAR", rate=rate)

            print("\nApplying median imputation...")
            X_train_imp, X_test_imp = median_imputation(X_train_miss, X_test)

            input_dim = X_train_imp.shape[1]

            print("\nTraining MLP on imputed X_train...")
            model = train_mlp(X_train_imp, y_train, input_dim)

            print("\nEvaluating on X_test...")
            auc, acc = evaluate_mlp(model, X_test_imp, y_test)

            rows.append({"dataset": name, "condition": condition, "AUC": auc, "accuracy": acc})


def run_knn_imputation(rows):
    for name in DATASETS:
        for rate in MCAR_RATES:
            rate_pct = int(rate * 100)
            condition = f"knn_MCAR_{rate_pct}"

            print(f"\n{'='*60}")
            print(f"Dataset: {name}  |  Condition: {condition}")
            print("=" * 60)

            X_train, X_test, y_train, y_test = load_dataset(name)

            print(f"\nInjecting MCAR missingness at {rate_pct}% into X_train...")
            X_train_miss = inject_missingness(X_train, mechanism="MCAR", rate=rate)

            print("\nApplying KNN imputation (k=5)...")
            X_train_imp, X_test_imp = knn_imputation(X_train_miss, X_test, k=5)

            input_dim = X_train_imp.shape[1]

            print("\nTraining MLP on imputed X_train...")
            model = train_mlp(X_train_imp, y_train, input_dim)

            print("\nEvaluating on X_test...")
            auc, acc = evaluate_mlp(model, X_test_imp, y_test)

            rows.append({"dataset": name, "condition": condition, "AUC": auc, "accuracy": acc})


def run_scarf(rows):
    existing_conditions = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_conditions.add((row["dataset"], row["condition"]))

    for name in DATASETS:
        for rate in MCAR_RATES:
            rate_pct = int(rate * 100)
            condition = f"scarf_MCAR_{rate_pct}"

            if (name, condition) in existing_conditions:
                print(f"\nSkipping {name} / {condition} (already in CSV)")
                continue

            print(f"\n{'='*60}")
            print(f"Dataset: {name}  |  Condition: {condition}")
            print("=" * 60)

            X_train, X_test, y_train, y_test = load_dataset(name)

            print(f"\nInjecting MCAR missingness at {rate_pct}% into X_train...")
            X_train_miss = inject_missingness(X_train, mechanism="MCAR", rate=rate)

            print("\nPretraining SCARF encoder on incomplete X_train...")
            encoder = pretrain_scarf(X_train_miss)

            print("\nFinetuning classification head on incomplete X_train...")
            encoder, head = finetune_scarf(encoder, X_train_miss, y_train)

            print("\nEvaluating on clean X_test...")
            train_medians = np.nanmedian(X_train_miss, axis=0)
            auc, acc = evaluate_scarf(encoder, head, X_test, y_test, train_medians)

            rows.append({"dataset": name, "condition": condition, "AUC": auc, "accuracy": acc})


def _feature_indices(feature_names, group_names):
    """Return column indices for each name in group_names."""
    return [feature_names.index(n) for n in group_names]


def run_feature_dependent(rows):
    existing_conditions = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_conditions.add((row["dataset"], row["condition"]))

    # Build correlated-group index lists dynamically from feature names
    _, _, uci_feat_names = _load_uci()
    _, _, telco_feat_names = _load_telco()

    uci_groups = [
        _feature_indices(uci_feat_names,
                         ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]),
        _feature_indices(uci_feat_names,
                         ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                          "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]),
        _feature_indices(uci_feat_names,
                         ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                          "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]),
    ]

    telco_groups = [
        _feature_indices(telco_feat_names,
                         ["StreamingTV", "StreamingMovies", "OnlineBackup",
                          "OnlineSecurity", "DeviceProtection", "TechSupport"]),
    ]

    dataset_groups = {"uci": uci_groups, "telco": telco_groups}

    STRATEGIES = ["complete_case", "median", "knn", "scarf"]
    FEAT_DEP_RATES = [0.10, 0.20, 0.30]

    for name in DATASETS:
        feature_groups = dataset_groups[name]

        for rate in FEAT_DEP_RATES:
            rate_pct = int(rate * 100)

            print(f"\n{'='*60}")
            print(f"Dataset: {name}  |  feature_dependent missingness at {rate_pct}%")
            print("=" * 60)

            # Load and inject missingness fresh for each rate
            X_train_clean, X_test, y_train, y_test = load_dataset(name)

            print(f"\nInjecting feature-dependent missingness (rate={rate})...")
            X_train_miss = inject_missingness(
                X_train_clean, mechanism="feature_dependent",
                rate=rate, feature_groups=feature_groups
            )

            for strategy in STRATEGIES:
                condition = f"{strategy}_featdep_{rate_pct}"

                if (name, condition) in existing_conditions:
                    print(f"\nSkipping {name} / {condition} (already in CSV)")
                    continue

                print(f"\n--- {name} | {condition} ---")

                if strategy == "complete_case":
                    X_tr, y_tr = complete_case(X_train_miss, y_train)
                    if len(X_tr) < 50:
                        print("  WARNING: Too few rows remaining -- skipping")
                        rows.append({"dataset": name, "condition": condition,
                                     "AUC": None, "accuracy": None})
                        continue
                    model = train_mlp(X_tr, y_tr, X_tr.shape[1])
                    auc, acc = evaluate_mlp(model, X_test, y_test)

                elif strategy == "median":
                    X_tr_imp, X_test_imp = median_imputation(X_train_miss, X_test)
                    model = train_mlp(X_tr_imp, y_train, X_tr_imp.shape[1])
                    auc, acc = evaluate_mlp(model, X_test_imp, y_test)

                elif strategy == "knn":
                    X_tr_imp, X_test_imp = knn_imputation(X_train_miss, X_test, k=5)
                    model = train_mlp(X_tr_imp, y_train, X_tr_imp.shape[1])
                    auc, acc = evaluate_mlp(model, X_test_imp, y_test)

                elif strategy == "scarf":
                    encoder = pretrain_scarf(X_train_miss)
                    encoder, head = finetune_scarf(encoder, X_train_miss, y_train)
                    train_medians = np.nanmedian(X_train_miss, axis=0)
                    auc, acc = evaluate_scarf(encoder, head, X_test, y_test, train_medians)

                rows.append({"dataset": name, "condition": condition,
                             "AUC": auc, "accuracy": acc})


def run_adult_clean_baseline(rows):
    name = "adult"

    print(f"\n{'='*60}")
    print(f"Dataset: {name}  |  Condition: clean_baseline")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_dataset(name)
    input_dim = X_train.shape[1]

    print("\nTraining MLP...")
    model = train_mlp(X_train, y_train, input_dim)

    print("\nEvaluating on test set...")
    auc, acc = evaluate_mlp(model, X_test, y_test)

    rows.append(
        {"dataset": name, "condition": "clean_baseline", "AUC": auc, "accuracy": acc}
    )


def run_adult_feature_dependent(rows):
    existing_conditions = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_conditions.add((row["dataset"], row["condition"]))

    name = "adult"
    STRATEGIES = ["complete_case", "median", "knn", "scarf"]
    FEAT_DEP_RATES = [0.10, 0.20, 0.30]

    for rate in FEAT_DEP_RATES:
        rate_pct = int(rate * 100)

        print(f"\n{'='*60}")
        print(f"Dataset: {name}  |  feature_dependent missingness at {rate_pct}%")
        print("=" * 60)

        # Reload for a fresh injection at each rate
        X_train_clean, X_test, y_train, y_test = load_dataset(name)

        print(f"\nInjecting feature-dependent missingness (rate={rate})...")
        # ADULT_CORRELATED_GROUPS already holds integer indices (set by _load_adult)
        X_train_miss = inject_missingness(
            X_train_clean, mechanism="feature_dependent",
            rate=rate, feature_groups=ADULT_CORRELATED_GROUPS
        )

        for strategy in STRATEGIES:
            condition = f"{strategy}_featdep_{rate_pct}"

            if (name, condition) in existing_conditions:
                print(f"\nSkipping {name} / {condition} (already in CSV)")
                continue

            print(f"\n--- {name} | {condition} ---")

            if strategy == "complete_case":
                X_tr, y_tr = complete_case(X_train_miss, y_train)
                if len(X_tr) < 50:
                    print("  WARNING: Too few rows remaining -- skipping")
                    rows.append({"dataset": name, "condition": condition,
                                 "AUC": None, "accuracy": None})
                    continue
                model = train_mlp(X_tr, y_tr, X_tr.shape[1])
                auc, acc = evaluate_mlp(model, X_test, y_test)

            elif strategy == "median":
                X_tr_imp, X_test_imp = median_imputation(X_train_miss, X_test)
                model = train_mlp(X_tr_imp, y_train, X_tr_imp.shape[1])
                auc, acc = evaluate_mlp(model, X_test_imp, y_test)

            elif strategy == "knn":
                X_tr_imp, X_test_imp = knn_imputation(X_train_miss, X_test, k=5)
                model = train_mlp(X_tr_imp, y_train, X_tr_imp.shape[1])
                auc, acc = evaluate_mlp(model, X_test_imp, y_test)

            elif strategy == "scarf":
                encoder = pretrain_scarf(X_train_miss)
                encoder, head = finetune_scarf(encoder, X_train_miss, y_train)
                train_medians = np.nanmedian(X_train_miss, axis=0)
                auc, acc = evaluate_scarf(encoder, head, X_test, y_test, train_medians)

            rows.append({"dataset": name, "condition": condition,
                         "AUC": auc, "accuracy": acc})


def run_adult_label_scarcity(rows):
    existing_conditions = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_conditions.add((row["dataset"], row["condition"]))

    name = "adult"
    LABEL_FRACS = [0.005, 0.01, 0.05, 0.10, 0.20]
    STRATEGIES = ["scarf", "median", "knn"]
    # MCAR 20% was chosen as the missingness condition for label scarcity experiments
    # as a representative moderate missingness rate, avoiding the extreme data loss
    # of 30% MCAR while still presenting a non-trivial imputation challenge.
    MCAR_RATE = 0.20

    print(f"\n{'='*60}")
    print(f"Dataset: {name}  |  Label scarcity (MCAR {int(MCAR_RATE * 100)}%)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_dataset(name)

    print(f"\nInjecting MCAR missingness at {int(MCAR_RATE * 100)}% into X_train...")
    X_train_miss = inject_missingness(X_train, mechanism="MCAR", rate=MCAR_RATE)

    train_medians = np.nanmedian(X_train_miss, axis=0)

    # Imputation is fitted once on the full training set before the label fraction loop.
    # Only the MLP classifier is trained on the labelled subset; the imputation statistics
    # already reflect the full unlabelled pool, matching how SCARF uses all available rows.
    print("\nPre-computing median imputation on full X_train_miss...")
    X_train_median_imp, X_test_median_imp = median_imputation(X_train_miss, X_test)

    print("\nPre-computing KNN imputation on full X_train_miss...")
    X_train_knn_imp, X_test_knn_imp = knn_imputation(X_train_miss, X_test, k=5)

    # SCARF is pretrained once on the full incomplete training set before the label
    # fraction loop.  This reflects the semi-supervised setting where unlabelled data
    # is abundant and labels are the scarce resource; the encoder learns representations
    # from all rows, and only the classification head is trained on the labelled subset.
    print("\nPretraining SCARF encoder on full X_train_miss...")
    encoder = pretrain_scarf(X_train_miss)

    for label_frac in LABEL_FRACS:
        frac_str = f"{label_frac * 100:g}"

        labelled_idx, _ = train_test_split(
            np.arange(len(y_train)), train_size=label_frac,
            random_state=42, stratify=y_train
        )
        y_labelled = y_train[labelled_idx]

        for strategy in STRATEGIES:
            condition = f"{strategy}_label_{frac_str}"

            if (name, condition) in existing_conditions:
                print(f"\nSkipping {name} / {condition} (already in CSV)")
                continue

            print(f"\n--- {name} | {condition} ({len(labelled_idx)} labelled rows) ---")

            if strategy == "median":
                X_tr = X_train_median_imp[labelled_idx]
                model = train_mlp(X_tr, y_labelled, X_tr.shape[1])
                auc, acc = evaluate_mlp(model, X_test_median_imp, y_test)

            elif strategy == "knn":
                X_tr = X_train_knn_imp[labelled_idx]
                model = train_mlp(X_tr, y_labelled, X_tr.shape[1])
                auc, acc = evaluate_mlp(model, X_test_knn_imp, y_test)

            elif strategy == "scarf":
                # Finetune only the head; encoder pretrained on the full unlabelled pool
                _, head = finetune_scarf(encoder, X_train_miss[labelled_idx], y_labelled)
                auc, acc = evaluate_scarf(encoder, head, X_test, y_test, train_medians)

            rows.append({"dataset": name, "condition": condition, "AUC": auc, "accuracy": acc})


def run_label_scarcity(rows):
    existing_conditions = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_conditions.add((row["dataset"], row["condition"]))

    LABEL_FRACS = [0.005, 0.01, 0.05, 0.10, 0.20]
    STRATEGIES = ["scarf", "median", "knn"]
    # MCAR 20% was chosen as the missingness condition for label scarcity experiments
    # as a representative moderate missingness rate, avoiding the extreme data loss
    # of 30% MCAR while still presenting a non-trivial imputation challenge.
    MCAR_RATE = 0.20

    for name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}  |  Label scarcity (MCAR {int(MCAR_RATE * 100)}%)")
        print("=" * 60)

        X_train, X_test, y_train, y_test = load_dataset(name)

        print(f"\nInjecting MCAR missingness at {int(MCAR_RATE * 100)}% into X_train...")
        X_train_miss = inject_missingness(X_train, mechanism="MCAR", rate=MCAR_RATE)

        train_medians = np.nanmedian(X_train_miss, axis=0)

        # Imputation is fitted once on the full training set before the label fraction loop.
        # Only the MLP classifier is trained on the labelled subset; the imputation statistics
        # already reflect the full unlabelled pool, matching how SCARF uses all available rows.
        print("\nPre-computing median imputation on full X_train_miss...")
        X_train_median_imp, X_test_median_imp = median_imputation(X_train_miss, X_test)

        print("\nPre-computing KNN imputation on full X_train_miss...")
        X_train_knn_imp, X_test_knn_imp = knn_imputation(X_train_miss, X_test, k=5)

        # SCARF is pretrained once on the full incomplete training set before the label
        # fraction loop.  This reflects the semi-supervised setting where unlabelled data
        # is abundant and labels are the scarce resource; the encoder learns representations
        # from all rows, and only the classification head is trained on the labelled subset.
        print("\nPretraining SCARF encoder on full X_train_miss...")
        encoder = pretrain_scarf(X_train_miss)

        for label_frac in LABEL_FRACS:
            frac_str = f"{label_frac * 100:g}"

            labelled_idx, _ = train_test_split(
                np.arange(len(y_train)), train_size=label_frac,
                random_state=42, stratify=y_train
            )
            y_labelled = y_train[labelled_idx]

            for strategy in STRATEGIES:
                condition = f"{strategy}_label_{frac_str}"

                if (name, condition) in existing_conditions:
                    print(f"\nSkipping {name} / {condition} (already in CSV)")
                    continue

                print(f"\n--- {name} | {condition} ({len(labelled_idx)} labelled rows) ---")

                if strategy == "median":
                    X_tr = X_train_median_imp[labelled_idx]
                    model = train_mlp(X_tr, y_labelled, X_tr.shape[1])
                    auc, acc = evaluate_mlp(model, X_test_median_imp, y_test)

                elif strategy == "knn":
                    X_tr = X_train_knn_imp[labelled_idx]
                    model = train_mlp(X_tr, y_labelled, X_tr.shape[1])
                    auc, acc = evaluate_mlp(model, X_test_knn_imp, y_test)

                elif strategy == "scarf":
                    # Finetune only the head; encoder pretrained on the full unlabelled pool
                    _, head = finetune_scarf(encoder, X_train_miss[labelled_idx], y_labelled)
                    auc, acc = evaluate_scarf(encoder, head, X_test, y_test, train_medians)

                rows.append({"dataset": name, "condition": condition, "AUC": auc, "accuracy": acc})


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    run_scarf(rows)
    run_feature_dependent(rows)
    run_label_scarcity(rows)
    run_adult_clean_baseline(rows)
    run_adult_feature_dependent(rows)
    run_adult_label_scarcity(rows)

    print(f"\n{'='*60}")
    print("Results summary (new rows + clean baseline reference)")
    print("=" * 60)

    # Print clean baseline from CSV for reference
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if row["condition"] == "clean_baseline":
                    print(f"  {row['dataset']:6s}  {'clean_baseline':30s}  "
                          f"AUC={row['AUC']}  Acc={row['accuracy']}")

    print()
    for r in rows:
        auc_str = f"{r['AUC']:.4f}" if r["AUC"] is not None else "None"
        acc_str = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "None"
        print(f"  {r['dataset']:6s}  {r['condition']:30s}  AUC={auc_str}  Acc={acc_str}")

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "condition", "AUC", "accuracy"])
        for r in rows:
            writer.writerow(
                {
                    "dataset": r["dataset"],
                    "condition": r["condition"],
                    "AUC": f"{r['AUC']:.3f}" if r["AUC"] is not None else "",
                    "accuracy": f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "",
                }
            )

    print(f"\nResults appended to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
