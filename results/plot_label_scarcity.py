import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent
DATASETS = ("uci", "telco", "adult")
METHODS = ("median", "knn", "scarf")
LABEL_FRACTIONS = (0.5, 1, 5, 10, 20)


def read_results(path):
    values = {}
    baselines = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            auc = float(row["AUC"]) if row["AUC"].strip() else np.nan
            key = (row["dataset"], row["condition"])
            values[key] = auc
            if row["condition"] == "clean_baseline":
                baselines[row["dataset"]] = auc
    return values, baselines


def main():
    values, baselines = read_results(RESULTS_DIR / "results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    colours = {"median": "#4c78a8", "knn": "#f58518", "scarf": "#54a24b"}

    for axis, dataset in zip(axes, DATASETS):
        for method in METHODS:
            aucs = [
                values.get((dataset, f"{method}_label_{fraction:g}"), np.nan)
                for fraction in LABEL_FRACTIONS
            ]
            axis.plot(
                LABEL_FRACTIONS,
                aucs,
                marker="o",
                color=colours[method],
                label=method.upper() if method != "knn" else "KNN",
            )
        axis.axhline(
            baselines[dataset],
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            label="Clean baseline",
        )
        axis.set_xscale("log")
        axis.set_xticks(LABEL_FRACTIONS, [f"{value:g}" for value in LABEL_FRACTIONS])
        axis.set_title(dataset.upper())
        axis.set_xlabel("Labelled training data (%)")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Test ROC AUC")
    telco_scarf = values[("telco", "scarf_label_0.5")]
    axes[1].annotate(
        "Not comparable",
        xy=(0.5, telco_scarf),
        xytext=(0.7, telco_scarf - 0.05),
        arrowprops={"arrowstyle": "->", "color": colours["scarf"]},
        fontsize=8,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(RESULTS_DIR / "label_scarcity_auc.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
