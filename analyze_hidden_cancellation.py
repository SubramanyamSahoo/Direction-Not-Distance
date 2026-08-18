#!/usr/bin/env python3
"""Zero-training diagnostic: quantify coordinate-level cancellation.

This asks how often the global scalar g^T u looks harmless because positive and
negative coordinate contributions cancel, even though many coordinates have
positive first-order alignment risk.  It uses only the completed original CSV.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path("pci_h100_outputs/results/all_metrics_enriched.csv")
OUT = Path("norm_matched_outputs/results")
OUT.mkdir(parents=True, exist_ok=True)

x = pd.read_csv(SRC)
x = x[(x["phase"] > 0) & (x["condition"] == "unconstrained")].copy()
pos = x["positive_coordinate_risk_sum"].astype(float)
net = x["raw_first_order_alignment_change"].astype(float)
neg_abs = (pos - net).clip(lower=0)
total_abs = pos + neg_abs
x["negative_coordinate_risk_abs_sum_derived"] = neg_abs
x["coordinate_contribution_total_abs"] = total_abs
x["cancellation_fraction"] = 1.0 - net.abs() / total_abs.replace(0, np.nan)
x["globally_nonharmful_but_positive_coordinate_risk"] = ((net <= 0) & (pos > 0)).astype(float)

cols = [
    "model", "seed", "regime", "phase", "raw_first_order_alignment_change",
    "positive_coordinate_risk_fraction", "positive_coordinate_risk_sum",
    "negative_coordinate_risk_abs_sum_derived", "cancellation_fraction",
    "globally_nonharmful_but_positive_coordinate_risk",
]
x[cols].to_csv(OUT / "hidden_coordinate_cancellation_rows.csv", index=False)
summary = x.groupby(["model", "regime"]).agg(
    rows=("phase", "count"),
    mean_positive_coordinate_fraction=("positive_coordinate_risk_fraction", "mean"),
    mean_cancellation_fraction=("cancellation_fraction", "mean"),
    median_cancellation_fraction=("cancellation_fraction", "median"),
    fraction_globally_nonharmful_with_positive_risk=("globally_nonharmful_but_positive_coordinate_risk", "mean"),
    mean_positive_risk_sum=("positive_coordinate_risk_sum", "mean"),
).reset_index()
summary.to_csv(OUT / "hidden_coordinate_cancellation_summary.csv", index=False)
print(summary.to_string(index=False))
print("HIDDEN CANCELLATION ANALYSIS COMPLETE")
