#!/usr/bin/env python3
"""Plot artifact-backed C1 GEPA Val125 ask_perception trends from trend.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trend_csv", type=Path)
    parser.add_argument("--output", type=Path, help="PNG output path (default: sibling trend.png)")
    parser.add_argument("--title", default="GEPA", help="Experiment label used in the first-panel title")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.trend_csv.open(encoding="utf-8", newline="")))
    if not rows:
        parser.error("trend CSV has no rows")
    out = args.output or args.trend_csv.with_name("trend.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(rows))
    scores = np.array([float(row["full_val_accuracy"]) for row in rows]) * 100
    coverage = np.array([float(row["saved_trajectory_coverage"]) for row in rows]) * 100
    saved = np.array([int(row["saved_trajectories"]) for row in rows])
    calls = np.array([int(row["observed_ask_perception_calls"]) for row in rows])
    call_rate = calls / saved * 100

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.35]})
    ax = axes[0]
    ax.plot(x, scores, "o-", color="#304f93", lw=2)
    ax.axhline(scores[0], color="#777777", ls="--", lw=1, label="Candidate 0 baseline")
    for i, row in enumerate(rows):
        ax.annotate(f'{int(row["full_val_correct"])}/125', (i, scores[i]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.set(ylabel="Accuracy (%)", title=f"{args.title}: full Val125 score at each accepted candidate")
    ax.set_ylim(min(scores) - 5, max(scores) + 6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left")

    ax = axes[1]
    ax.bar(x, coverage, color="#8eafcf", label="Saved trajectories / 125")
    ax.plot(x, call_rate, "o-", color="#dd7444", label="ask_perception calls / saved trajectories")
    for i, n in enumerate(saved):
        ax.annotate(str(n), (i, coverage[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set(ylabel="Percent (%)", title="Archive coverage and observed perception-call rate", ylim=(0, 105))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", ncol=2, fontsize=9)

    ax = axes[2]
    colors = {1: "#c7564f", 2: "#4d83b4", 3: "#5d9e72", 4: "#999999"}
    labels = {1: "1  direct option request", 2: "2  narrowed original QA", 3: "3  intermediate subquestion", 4: "4  other"}
    bottom = np.zeros(len(rows))
    for category in range(1, 5):
        values = np.array([float(row[f"category_{category}_share_of_observed_calls"] or 0) for row in rows]) * 100
        ax.bar(x, values, bottom=bottom, color=colors[category], label=labels[category], width=0.73)
        bottom += values
    for i, n in enumerate(calls):
        ax.annotate(f"n={n}", (i, 100), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.set(ylabel="Share of observed calls (%)", xlabel="Validation candidate index (search order)", ylim=(0, 113), title="Question category among saved ask_perception calls")
    ax.set_xticks(x, [f'{int(row["candidate_index"])}\n{int(row["discovery_eval_count"])} calls' for row in rows])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=False, fontsize=9)

    fig.text(0.5, 0.015,
             "Archive limitation: each candidate retains all correct Val predictions plus 11 incorrect ones; missing incorrect trajectories cannot be classified.",
             ha="center", fontsize=9, color="#8b2f2f")
    fig.tight_layout(rect=(0, 0.065, 1, 1), h_pad=2.2)
    fig.savefig(out, dpi=180)
    fig.savefig(out.with_suffix(".svg"))
    print(out)


if __name__ == "__main__":
    main()
