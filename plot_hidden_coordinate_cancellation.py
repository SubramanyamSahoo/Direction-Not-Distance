#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent

ROWS_PATH = (
    ROOT
    / "results"
    / "norm_matched"
    / "hidden_coordinate_cancellation_rows.csv"
)

SUMMARY_PATH = (
    ROOT
    / "results"
    / "norm_matched"
    / "hidden_coordinate_cancellation_summary.csv"
)

OUT_DIR = ROOT / "results" / "main" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_STEM = OUT_DIR / "hidden_coordinate_cancellation"


# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------

rows = pd.read_csv(ROWS_PATH)
summary = pd.read_csv(SUMMARY_PATH)

required_summary = {
    "model",
    "regime",
    "mean_positive_coordinate_fraction",
    "mean_cancellation_fraction",
    "median_cancellation_fraction",
    "fraction_globally_nonharmful_with_positive_risk",
    "mean_positive_risk_sum",
}

missing = required_summary - set(summary.columns)
if missing:
    raise RuntimeError(
        f"Missing required columns in {SUMMARY_PATH}: {sorted(missing)}"
    )


# The analysis code should have emitted these names.
# Keep a small fallback list so the plotting script is robust to
# harmless naming differences.
net_candidates = [
    "raw_first_order_alignment_change",
    "net_first_order_alignment_change",
    "global_first_order_alignment_change",
]

positive_risk_candidates = [
    "positive_coordinate_risk_sum",
    "positive_risk_sum",
]


def find_column(df, candidates, meaning):
    for col in candidates:
        if col in df.columns:
            return col
    raise RuntimeError(
        f"Could not locate {meaning} column.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


net_col = find_column(
    rows,
    net_candidates,
    "net first-order alignment change",
)

positive_risk_col = find_column(
    rows,
    positive_risk_candidates,
    "positive coordinate risk",
)


# ---------------------------------------------------------------------
# 2. Fixed display order
# ---------------------------------------------------------------------

ORDER = [
    ("Qwen3-8B", "benign"),
    ("Qwen3-8B", "conflict"),
    ("Qwen3-14B", "benign"),
    ("Qwen3-14B", "conflict"),
]

MARKERS = {
    ("Qwen3-8B", "benign"): "o",
    ("Qwen3-8B", "conflict"): "s",
    ("Qwen3-14B", "benign"): "^",
    ("Qwen3-14B", "conflict"): "D",
}

DISPLAY = {
    ("Qwen3-8B", "benign"): "8B · benign",
    ("Qwen3-8B", "conflict"): "8B · conflict",
    ("Qwen3-14B", "benign"): "14B · benign",
    ("Qwen3-14B", "conflict"): "14B · conflict",
}


# ---------------------------------------------------------------------
# 3. Publication settings
# ---------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------

# ~AAAI double-column width.
fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.0, 3.05),
)

ax = axes[0]


# ---------------------------------------------------------------------
# Panel A:
# Global first-order signal versus hidden positive coordinate risk
# ---------------------------------------------------------------------

for model, regime in ORDER:
    sub = rows[
        (rows["model"] == model)
        & (rows["regime"] == regime)
    ]

    if len(sub) == 0:
        continue

    ax.scatter(
        sub[net_col],
        sub[positive_risk_col],
        s=24,
        marker=MARKERS[(model, regime)],
        alpha=0.72,
        linewidths=0.5,
        label=DISPLAY[(model, regime)],
    )


# g^T u = 0 boundary.
ax.axvline(
    0.0,
    linestyle="--",
    linewidth=1.0,
)

# Mark the region in which the *global* first-order criterion
# judges the update as non-harmful.
xmin, xmax = ax.get_xlim()

if xmin < 0:
    ax.axvspan(
        xmin,
        0.0,
        alpha=0.06,
        zorder=-10,
    )

ax.set_xlim(xmin, xmax)

ax.set_xlabel(
    r"Net first-order alignment change  $g^\top u$"
)

ax.set_ylabel(
    r"Positive coordinate risk  "
    r"$C^{+}=\sum_i \max(g_i u_i,0)$"
)

ax.set_title(
    "(a) Global safety can conceal local conflict",
    loc="left",
)

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper left",
)

ax.text(
    0.02,
    0.04,
    r"$g^\top u \leq 0$:"
    "\nglobally non-harmful",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=7.5,
)


# ---------------------------------------------------------------------
# Panel B:
# How much opposing coordinate contribution cancels globally?
# ---------------------------------------------------------------------

ax = axes[1]

ordered_rows = []

for key in ORDER:
    model, regime = key

    sub = summary[
        (summary["model"] == model)
        & (summary["regime"] == regime)
    ]

    if len(sub) != 1:
        raise RuntimeError(
            f"Expected one summary row for {model}/{regime}; "
            f"found {len(sub)}."
        )

    ordered_rows.append(sub.iloc[0])

ordered = pd.DataFrame(ordered_rows).reset_index(drop=True)

x = np.arange(len(ordered))

mean_cancel = (
    ordered["mean_cancellation_fraction"].to_numpy()
)

median_cancel = (
    ordered["median_cancellation_fraction"].to_numpy()
)

global_hidden = (
    ordered[
        "fraction_globally_nonharmful_with_positive_risk"
    ].to_numpy()
)

positive_fraction = (
    ordered["mean_positive_coordinate_fraction"].to_numpy()
)


bars = ax.bar(
    x,
    mean_cancel,
    width=0.62,
    label="Mean cancellation",
)

ax.scatter(
    x,
    median_cancel,
    marker="D",
    s=25,
    zorder=5,
    label="Median cancellation",
)


# Values are approximately 94--95%, so zooming the y-axis makes
# the between-condition variation visible without hiding the absolute
# values, which are printed on each bar.
lower = max(
    0.0,
    float(np.nanmin(mean_cancel)) - 0.04,
)

ax.set_ylim(lower, 1.01)

ax.yaxis.set_major_formatter(
    PercentFormatter(xmax=1.0)
)

ax.set_ylabel(
    "Coordinate contribution cancelled"
)

ax.set_xticks(x)

ax.set_xticklabels(
    [
        "8B\nbenign",
        "8B\nconflict",
        "14B\nbenign",
        "14B\nconflict",
    ]
)

ax.set_title(
    "(b) Cancellation is extensive across regimes",
    loc="left",
)


# Add exact mean cancellation percentage above each bar.
for i, value in enumerate(mean_cancel):
    ax.text(
        i,
        value + 0.006,
        f"{100.0 * value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )


# Add two mechanistically useful annotations inside each bar:
#   positive-risk coordinate fraction
#   fraction of updates globally non-harmful despite positive risk
annotation_y = lower + 0.007

for i in range(len(ordered)):
    ax.text(
        i,
        annotation_y,
        (
            f"risk coords: "
            f"{100 * positive_fraction[i]:.0f}%\n"
            f"net-safe: "
            f"{100 * global_hidden[i]:.0f}%"
        ),
        ha="center",
        va="bottom",
        fontsize=6.7,
    )


ax.legend(
    frameon=False,
    loc="upper right",
)


# ---------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------

fig.tight_layout(
    pad=0.7,
    w_pad=1.6,
)

for suffix in (".pdf", ".png", ".svg"):
    path = Path(str(OUT_STEM) + suffix)

    if suffix == ".png":
        fig.savefig(
            path,
            bbox_inches="tight",
            dpi=600,
        )
    else:
        fig.savefig(
            path,
            bbox_inches="tight",
        )

plt.close(fig)


# ---------------------------------------------------------------------
# 6. Print exact values used in the figure
# ---------------------------------------------------------------------

print("\nHidden-coordinate cancellation figure generated.")
print(f"Rows source:    {ROWS_PATH}")
print(f"Summary source: {SUMMARY_PATH}")

print("\nSummary values used:")

display = ordered[
    [
        "model",
        "regime",
        "mean_positive_coordinate_fraction",
        "mean_cancellation_fraction",
        "median_cancellation_fraction",
        "fraction_globally_nonharmful_with_positive_risk",
        "mean_positive_risk_sum",
    ]
].copy()

print(display.to_string(index=False))

print("\nOutputs:")
for suffix in (".pdf", ".png", ".svg"):
    print(Path(str(OUT_STEM) + suffix))
