#!/usr/bin/env python3
"""Render category 1/2/3 percentages by saved GEPA validation step.

Percentages use only the `observed_ask_perception_calls` denominator written by
the historical-trajectory analyzer. The archive does not retain every failed
validation trajectory, so this is not a full-Val125 question distribution.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CATEGORIES = {
    1: ("Category 1: direct option request", "#c7564f"),
    2: ("Category 2: narrowed original QA", "#4d83b4"),
    3: ("Category 3: intermediate subquestion", "#5d9e72"),
}


def load(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def render(label: str, path: Path, output: Path) -> None:
    rows = load(path)
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.suptitle(f"{label}: ask_perception category share by Val125 candidate", fontsize=15)
    for category, (name, color) in CATEGORIES.items():
        values = [
            100 * int(row[f"category_{category}_count"]) / int(row["observed_ask_perception_calls"])
            if int(row["observed_ask_perception_calls"]) else 0
            for row in rows
        ]
        ax.plot(x, values, "o-", lw=2.2, ms=5.5, color=color, label=name)
    for index, row in enumerate(rows):
        ax.annotate(
            f'n={row["observed_ask_perception_calls"]}', (index, 101.5),
            ha="center", va="bottom", fontsize=7.5, color="#444444",
        )
    ax.set(
        xlabel="Candidate index (search order; discovery-call count is in trend.csv)",
        ylabel="Share of observed ask_perception calls (%)",
        ylim=(0, 110),
    )
    ax.set_xticks(
        x,
        [row["candidate_index"] for row in rows],
        fontsize=8,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.035), ncol=3, fontsize=9)
    fig.text(
        0.5, 0.012,
        "Historical archive only: percentages use saved ask_perception calls; missing trajectories are not counted as no-call trajectories.",
        ha="center", fontsize=8.5, color="#8b2f2f",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trend", action="append", nargs=2, metavar=("LABEL", "CSV"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for label, raw_path in args.trend:
        safe = "".join(char.lower() if char.isalnum() else "_" for char in label).strip("_")
        render(label, Path(raw_path), args.output_dir / f"{safe}_category_percentages.png")


if __name__ == "__main__":
    main()
