#!/usr/bin/env python3
"""Plot the modality-merge accuracy/efficiency trade-offs.

The accepted Stage B v2 rerun substitutions are applied automatically.  B2's
rejected rerun is deliberately excluded.  Each point is the mean of three
logical repeats. The cache-inclusive mean-only figure includes B0--B11.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox


V2_ROOT = Path(
    "/mnt/ceph_rbd/data/avqa_project/daily_omni/"
    "modality_merge_ablation_v2_selectedVal125_repeats"
)
V2_RERUN_ROOT = Path(
    "/mnt/ceph_rbd/data/avqa_project/daily_omni/"
    "modality_merge_ablation_v2_selectedVal125_outlier_reruns"
)
V3_ROOT = Path(
    "/mnt/ceph_rbd/data/avqa_project/daily_omni/"
    "modality_merge_ablation_v3_selectedVal125_repeats"
)

MAIN_IDS = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"]
CACHE_INCLUSIVE_IDS = MAIN_IDS + ["B11"]
ALL_IDS = MAIN_IDS[:6] + ["B5-flex"] + MAIN_IDS[6:]

ACCURACY_LIM = (69.5, 83.5)
COMMON_X_LIM = (0.0, 1.0)
MAIN_COLOR = "#3566A8"
FLEX_COLOR = "#D55E00"
ERROR_COLOR = "#56616F"

# A 12-color qualitative palette for the mean-only paper variant.  Colors are
# deliberately stable across all panels and separate figures.
EXPERIMENT_COLORS = {
    "B0": "#332288",
    "B1": "#88CCEE",
    "B2": "#44AA99",
    "B3": "#117733",
    "B4": "#999933",
    "B5": "#DDCC77",
    "B5-flex": "#CC6677",
    "B6": "#882255",
    "B7": "#AA4499",
    "B8": "#661100",
    "B9": "#6699CC",
    "B10": "#DD4444",
    "B11": "#A14E00",
}


@dataclass(frozen=True)
class Summary:
    mean: float
    std: float


@dataclass(frozen=True)
class Experiment:
    accuracy: Summary
    total_tokens: Summary
    cost: Summary
    latency: Summary


@dataclass(frozen=True)
class XMetric:
    field: str
    xlabel: str
    bounds: tuple[float, float]
    ticks: tuple[float, ...]
    tick_label: Callable[[float], str]
    stem: str

    def normalize(self, value: float) -> float:
        low, high = self.bounds
        return (value - low) / (high - low)

    def normalize_error(self, value: float) -> float:
        low, high = self.bounds
        return value / (high - low)


METRICS = (
    XMetric(
        field="total_tokens",
        xlabel="Total live tokens per question (input + all output)",
        bounds=(5_000.0, 28_000.0),
        ticks=(5_000.0, 10_000.0, 15_000.0, 20_000.0, 25_000.0),
        tick_label=lambda value: f"{value / 1000:.0f}k",
        stem="total_tokens",
    ),
    XMetric(
        field="cost",
        xlabel="Live cost per question (USD)",
        bounds=(0.01, 0.09),
        ticks=(0.01, 0.03, 0.05, 0.07, 0.09),
        tick_label=lambda value: f"${value:.2f}",
        stem="cost",
    ),
    XMetric(
        field="latency",
        xlabel="Retry-adjusted latency per question (seconds)",
        bounds=(10.0, 76.0),
        ticks=(10.0, 25.0, 40.0, 55.0, 70.0),
        tick_label=lambda value: f"{value:.0f}",
        stem="latency",
    ),
)

# Point-label offsets in display points.  Per-panel overrides keep labels clear
# without adding a dependency on an automatic label-placement package.
DEFAULT_OFFSETS = {
    "B0": (5, 6),
    "B1": (5, -13),
    "B2": (5, -13),
    "B3": (5, 6),
    "B4": (-20, 7),
    "B5": (5, -13),
    "B5-flex": (6, 7),
    "B6": (6, 7),
    "B7": (5, 7),
    "B8": (-21, -13),
    "B9": (5, 7),
    "B10": (5, -13),
}

OFFSET_OVERRIDES = {
    "total_tokens": {
        "B0": (5, 7),
        "B1": (-22, -13),
        "B4": (5, 7),
        "B5-flex": (-34, 7),
        "B6": (6, 8),
        "B8": (-21, -13),
        "B10": (5, -13),
    },
    "cost": {
        "B0": (5, 7),
        "B1": (-22, -13),
        "B4": (5, 7),
        "B5-flex": (-34, 7),
        "B6": (6, 8),
        "B8": (-21, -13),
    },
    "latency": {
        "B0": (5, 7),
        "B1": (-22, -13),
        "B4": (5, 7),
        "B5-flex": (-34, 7),
        "B6": (6, 8),
        "B8": (-21, -13),
    },
}


def result_path(experiment_id: str, repeat: int) -> Path:
    key = experiment_id.lower()
    accepted_replacements = {
        ("b0", 3): V2_RERUN_ROOT / "b0-repeat3-rerun" / "metrics_summary.json",
        ("b1", 2): V2_RERUN_ROOT / "b1-repeat2-rerun" / "metrics_summary.json",
        ("b3", 1): V2_RERUN_ROOT / "b3-repeat1-rerun" / "metrics_summary.json",
        ("b5-flex", 2): V2_RERUN_ROOT / "b5-flex-repeat2-rerun" / "metrics_summary.json",
    }
    if (key, repeat) in accepted_replacements:
        return accepted_replacements[(key, repeat)]
    if key in {"b7", "b8", "b9", "b10"}:
        return V3_ROOT / f"repeat{repeat}" / key / "metrics_summary.json"
    return V2_ROOT / f"repeat{repeat}" / key / "metrics_summary.json"


def summarize(values: list[float]) -> Summary:
    return Summary(statistics.mean(values), statistics.stdev(values))


def load_data() -> dict[str, Experiment]:
    experiments: dict[str, Experiment] = {}
    missing: list[Path] = []
    for experiment_id in ALL_IDS:
        runs = []
        for repeat in (1, 2, 3):
            path = result_path(experiment_id, repeat)
            if not path.is_file():
                missing.append(path)
                continue
            with path.open(encoding="utf-8") as handle:
                runs.append(json.load(handle))
        if len(runs) != 3:
            continue
        experiments[experiment_id] = Experiment(
            accuracy=summarize([run["accuracy"] * 100 for run in runs]),
            total_tokens=summarize(
                [
                    run["average_live_input_tokens"] + run["average_live_output_tokens"]
                    for run in runs
                ]
            ),
            cost=summarize([run["average_live_cost_usd"] for run in runs]),
            latency=summarize(
                [run["average_retry_adjusted_latency_seconds"] for run in runs]
            ),
        )
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing result summaries:\n{paths}")
    return experiments


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_panel(ax: plt.Axes, data: dict[str, Experiment], metric: XMetric) -> None:
    main_x = [metric.normalize(getattr(data[key], metric.field).mean) for key in MAIN_IDS]
    main_y = [data[key].accuracy.mean for key in MAIN_IDS]
    ax.plot(main_x, main_y, color=MAIN_COLOR, lw=1.35, zorder=2)

    for key, x, y in zip(MAIN_IDS, main_x, main_y):
        x_summary = getattr(data[key], metric.field)
        ax.errorbar(
            x,
            y,
            xerr=metric.normalize_error(x_summary.std),
            yerr=data[key].accuracy.std,
            fmt="o",
            ms=5.0,
            mfc="white",
            mec=MAIN_COLOR,
            mew=1.2,
            ecolor=ERROR_COLOR,
            elinewidth=0.75,
            capsize=2.2,
            capthick=0.75,
            zorder=3,
        )

    b5 = data["B5"]
    flex = data["B5-flex"]
    b5_x = metric.normalize(getattr(b5, metric.field).mean)
    flex_x_summary = getattr(flex, metric.field)
    flex_x = metric.normalize(flex_x_summary.mean)
    ax.plot(
        [b5_x, flex_x],
        [b5.accuracy.mean, flex.accuracy.mean],
        color=FLEX_COLOR,
        lw=1.25,
        ls=(0, (2.2, 2.2)),
        zorder=2,
    )
    ax.errorbar(
        flex_x,
        flex.accuracy.mean,
        xerr=metric.normalize_error(flex_x_summary.std),
        yerr=flex.accuracy.std,
        fmt="D",
        ms=5.1,
        mfc="white",
        mec=FLEX_COLOR,
        mew=1.2,
        ecolor=FLEX_COLOR,
        elinewidth=0.75,
        capsize=2.2,
        capthick=0.75,
        zorder=4,
    )

    offsets = DEFAULT_OFFSETS | OFFSET_OVERRIDES.get(metric.field, {})
    for key in ALL_IDS:
        experiment = data[key]
        x = metric.normalize(getattr(experiment, metric.field).mean)
        dx, dy = offsets[key]
        ax.annotate(
            key,
            (x, experiment.accuracy.mean),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="bottom" if dy >= 0 else "top",
            fontsize=7.4,
            color=FLEX_COLOR if key == "B5-flex" else "#222222",
            zorder=5,
        )

    tick_positions = [metric.normalize(value) for value in metric.ticks]
    ax.set_xticks(tick_positions, [metric.tick_label(value) for value in metric.ticks])
    ax.set_xlim(*COMMON_X_LIM)
    ax.set_ylim(*ACCURACY_LIM)
    ax.set_xlabel(metric.xlabel)
    ax.set_ylabel("Accuracy on selected Val125 (%)")
    ax.grid(axis="both", color="#CBD2D9", alpha=0.62, lw=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.0, width=0.75)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=MAIN_COLOR,
            lw=1.35,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            label="Main ablation path (B0–B10)",
        ),
        Line2D(
            [0],
            [0],
            color=FLEX_COLOR,
            lw=1.25,
            ls=(0, (2.2, 2.2)),
            marker="D",
            markersize=5,
            markerfacecolor="white",
            label="B5-flex order/placement control",
        ),
    ]


def plot_mean_only_panel(
    ax: plt.Axes, data: dict[str, Experiment], metric: XMetric
) -> None:
    """Plot unconnected solid means with one consistently placed point label."""
    for key in ALL_IDS:
        experiment = data[key]
        x = metric.normalize(getattr(experiment, metric.field).mean)
        y = experiment.accuracy.mean
        color = EXPERIMENT_COLORS[key]
        ax.plot(
            x,
            y,
            marker="o",
            ls="none",
            ms=9.0,
            mfc=color,
            mec="white",
            mew=0.65,
            zorder=3,
        )
        ax.annotate(
            key,
            (x, y),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.4,
            color=color,
            fontweight="bold",
            zorder=4,
        )

    tick_positions = [metric.normalize(value) for value in metric.ticks]
    ax.set_xticks(tick_positions, [metric.tick_label(value) for value in metric.ticks])
    ax.set_xlim(*COMMON_X_LIM)
    ax.set_ylim(*ACCURACY_LIM)
    ax.set_xlabel(metric.xlabel)
    ax.set_ylabel("Accuracy on selected Val125 (%)")
    ax.grid(axis="both", color="#CBD2D9", alpha=0.62, lw=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.0, width=0.75)


def mean_only_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            ls="none",
            marker="o",
            markersize=7.2,
            markerfacecolor=EXPERIMENT_COLORS[key],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=key,
        )
        for key in ALL_IDS
    ]


def row_major_mean_only_legend_handles(ncol: int) -> list[Line2D]:
    """Counter Matplotlib's column-major legend layout for sequential IDs."""
    handles = mean_only_legend_handles()
    nrows = (len(handles) + ncol - 1) // ncol
    return [
        handles[row * ncol + column]
        for column in range(ncol)
        for row in range(nrows)
        if row * ncol + column < len(handles)
    ]


def main_only_legend_handles(
    ncol: int, ids: list[str] = MAIN_IDS
) -> list[Line2D]:
    handles = [
        Line2D(
            [0], [0], ls="none", marker="o", markersize=7.2,
            markerfacecolor=EXPERIMENT_COLORS[key], markeredgecolor="white",
            markeredgewidth=0.6, label=key,
        )
        for key in ids
    ]
    nrows = (len(handles) + ncol - 1) // ncol
    return [
        handles[row * ncol + column]
        for column in range(ncol)
        for row in range(nrows)
        if row * ncol + column < len(handles)
    ]


def intersection_area(first: Bbox, second: Bbox) -> float:
    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def plot_mean_only_no_flex_panel(
    ax: plt.Axes, data: dict[str, Experiment], metric: XMetric,
    ids: list[str] = MAIN_IDS,
) -> list[plt.Annotation]:
    """Draw the main path as distinct means, leaving label placement for later."""
    labels = []
    for key in ids:
        experiment = data[key]
        x = metric.normalize(getattr(experiment, metric.field).mean)
        y = experiment.accuracy.mean
        color = EXPERIMENT_COLORS[key]
        ax.plot(
            x, y, marker="o", ls="none", ms=9.0, mfc=color,
            mec="white", mew=0.65, zorder=3,
        )
        labels.append(
            ax.annotate(
                key, (x, y), xytext=(0, 0), textcoords="offset points",
                ha="left", va="bottom", fontsize=7.4, color=color,
                fontweight="bold", zorder=4,
            )
        )

    tick_positions = [metric.normalize(value) for value in metric.ticks]
    ax.set_xticks(tick_positions, [metric.tick_label(value) for value in metric.ticks])
    ax.set_xlim(*COMMON_X_LIM)
    ax.set_ylim(*ACCURACY_LIM)
    ax.set_xlabel(metric.xlabel)
    ax.set_ylabel("Accuracy on selected Val125 (%)")
    ax.grid(axis="both", color="#CBD2D9", alpha=0.62, lw=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.0, width=0.75)
    return labels


def place_labels_around_points(ax: plt.Axes, labels: list[plt.Annotation]) -> None:
    """Choose nearby directions that avoid other labels, markers and plot edges."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    centers = [ax.transData.transform(label.xy) for label in labels]
    marker_radius_px = 5.2 * fig.dpi / 72
    marker_boxes = [
        Bbox.from_bounds(
            center[0] - marker_radius_px,
            center[1] - marker_radius_px,
            2 * marker_radius_px,
            2 * marker_radius_px,
        )
        for center in centers
    ]
    axes_box = ax.bbox.padded(-2)
    directions = (
        (1, 1, "left", "bottom"), (-1, 1, "right", "bottom"),
        (1, -1, "left", "top"), (-1, -1, "right", "top"),
        (0, 1, "center", "bottom"), (0, -1, "center", "top"),
        (1, 0, "left", "center"), (-1, 0, "right", "center"),
    )
    candidates = [
        (radius * vx, radius * vy, ha, va)
        for radius in (6, 11, 17, 24, 32)
        for vx, vy, ha, va in directions
    ]
    distances = []
    for index, center in enumerate(centers):
        distances.append(
            min(
                (sum((center - other) ** 2)) ** 0.5
                for other_index, other in enumerate(centers)
                if other_index != index
            )
        )
    order = sorted(range(len(labels)), key=lambda index: (distances[index], index))
    placements: dict[int, tuple[tuple[int, int, str, str], Bbox]] = {}

    for _ in range(3):
        for index in order:
            label = labels[index]
            best = None
            for candidate in candidates:
                dx, dy, ha, va = candidate
                label.set_position((dx, dy))
                label.set_horizontalalignment(ha)
                label.set_verticalalignment(va)
                box = label.get_window_extent(renderer).padded(1.2)
                outside = box.width * box.height - intersection_area(box, axes_box)
                label_overlap = sum(
                    intersection_area(box, other_box)
                    for other_index, (_, other_box) in placements.items()
                    if other_index != index
                )
                marker_overlap = sum(intersection_area(box, marker_box) for marker_box in marker_boxes)
                distance_penalty = (dx * dx + dy * dy) ** 0.5
                score = 1000 * outside + 120 * label_overlap + 40 * marker_overlap + distance_penalty
                if best is None or score < best[0]:
                    best = (score, candidate, box)
            assert best is not None
            _, candidate, box = best
            label.set_position(candidate[:2])
            label.set_horizontalalignment(candidate[2])
            label.set_verticalalignment(candidate[3])
            placements[index] = (candidate, box)


def write_main_only_data_csv(data: dict[str, Experiment], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["id", "accuracy_mean_pct", "total_live_tokens_mean",
             "live_cost_usd_mean", "retry_adjusted_latency_seconds_mean"]
        )
        for key in MAIN_IDS:
            experiment = data[key]
            writer.writerow(
                [key, experiment.accuracy.mean, experiment.total_tokens.mean,
                 experiment.cost.mean, experiment.latency.mean]
            )


def load_cache_inclusive_csv(path: Path) -> dict[str, Experiment]:
    data: dict[str, Experiment] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            data[row["id"]] = Experiment(
                accuracy=Summary(float(row["accuracy_mean_pct"]), 0.0),
                total_tokens=Summary(float(row["total_tokens_mean"]), 0.0),
                cost=Summary(float(row["cost_usd_mean"]), 0.0),
                latency=Summary(float(row["measured_latency_seconds_mean"]), 0.0),
            )
    missing = set(CACHE_INCLUSIVE_IDS) - set(data)
    if missing:
        raise ValueError(f"Cache-inclusive CSV lacks configurations: {sorted(missing)}")
    return data


def plot_cache_inclusive_three_panel(data: dict[str, Experiment], output_dir: Path) -> None:
    metrics = (
        replace(
            METRICS[0],
            xlabel="Total tokens per question (input + all output)",
            bounds=(10_000.0, 30_000.0),
            ticks=(10_000.0, 15_000.0, 20_000.0, 25_000.0, 30_000.0),
        ),
        replace(METRICS[1], xlabel="Estimated cost per question (USD)"),
        replace(METRICS[2], xlabel="Measured latency per question (seconds)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.55), sharey=False)
    panel_labels = [
        plot_mean_only_no_flex_panel(ax, data, metric, CACHE_INCLUSIVE_IDS)
        for ax, metric in zip(axes, metrics)
    ]
    fig.legend(
        handles=main_only_legend_handles(6, CACHE_INCLUSIVE_IDS), loc="upper center",
        bbox_to_anchor=(0.5, 1.012), ncol=6, columnspacing=1.3,
        handletextpad=0.35, frameon=False,
    )
    fig.subplots_adjust(left=0.052, right=0.995, bottom=0.155, top=0.82, wspace=0.29)
    for ax, labels in zip(axes, panel_labels):
        place_labels_around_points(ax, labels)
    save_figure(fig, output_dir / "ablation_accuracy_efficiency_three_panel_cache_inclusive")
    plt.close(fig)


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=300, bbox_inches="tight")


def write_data_csv(data: dict[str, Experiment], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "accuracy_mean_pct",
                "accuracy_sample_std_pct",
                "total_live_tokens_mean",
                "total_live_tokens_sample_std",
                "live_cost_usd_mean",
                "live_cost_usd_sample_std",
                "retry_adjusted_latency_seconds_mean",
                "retry_adjusted_latency_seconds_sample_std",
            ]
        )
        for key in ALL_IDS:
            experiment = data[key]
            writer.writerow(
                [
                    key,
                    experiment.accuracy.mean,
                    experiment.accuracy.std,
                    experiment.total_tokens.mean,
                    experiment.total_tokens.std,
                    experiment.cost.mean,
                    experiment.cost.std,
                    experiment.latency.mean,
                    experiment.latency.std,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/figures/modality_merge_ablation"),
    )
    parser.add_argument(
        "--cache-inclusive-csv",
        type=Path,
        help="Plot the 11 main configurations from a cache-inclusive summary CSV",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    if args.cache_inclusive_csv is not None:
        data = load_cache_inclusive_csv(args.cache_inclusive_csv)
        plot_cache_inclusive_three_panel(data, args.output_dir)
        print(args.output_dir.resolve())
        return
    data = load_data()
    write_data_csv(data, args.output_dir / "ablation_tradeoff_data.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.35), sharey=False)
    for ax, metric in zip(axes, METRICS):
        plot_panel(ax, data, metric)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.052, right=0.995, bottom=0.16, top=0.88, wspace=0.29)
    save_figure(fig, args.output_dir / "ablation_accuracy_efficiency_three_panel")
    plt.close(fig)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(5.45, 4.35))
        plot_panel(ax, data, metric)
        ax.legend(handles=legend_handles(), loc="upper right", frameon=False)
        fig.subplots_adjust(left=0.145, right=0.98, bottom=0.15, top=0.98)
        save_figure(fig, args.output_dir / f"ablation_accuracy_vs_{metric.stem}")
        plt.close(fig)

    # Cleaner paper alternative: means only, no connecting lines or error bars.
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.55), sharey=False)
    for ax, metric in zip(axes, METRICS):
        plot_mean_only_panel(ax, data, metric)
    fig.legend(
        handles=row_major_mean_only_legend_handles(6),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.012),
        ncol=6,
        columnspacing=1.3,
        handletextpad=0.35,
        frameon=False,
    )
    fig.subplots_adjust(left=0.052, right=0.995, bottom=0.155, top=0.82, wspace=0.29)
    save_figure(fig, args.output_dir / "ablation_accuracy_efficiency_three_panel_mean_only")
    plt.close(fig)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(5.45, 5.0))
        plot_mean_only_panel(ax, data, metric)
        fig.legend(
            handles=row_major_mean_only_legend_handles(4),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=4,
            columnspacing=1.05,
            handletextpad=0.3,
            frameon=False,
        )
        fig.subplots_adjust(left=0.145, right=0.98, bottom=0.13, top=0.74)
        save_figure(fig, args.output_dir / f"ablation_accuracy_vs_{metric.stem}_mean_only")
        plt.close(fig)

    # Main-path-only mean plots with labels placed near the points where space
    # is available. Keep the previous variants as separate, reproducible files.
    write_main_only_data_csv(data, args.output_dir / "ablation_tradeoff_data_mean_only_no_flex.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.55), sharey=False)
    panel_labels = [
        plot_mean_only_no_flex_panel(ax, data, metric)
        for ax, metric in zip(axes, METRICS)
    ]
    fig.legend(
        handles=main_only_legend_handles(6), loc="upper center",
        bbox_to_anchor=(0.5, 1.012), ncol=6, columnspacing=1.3,
        handletextpad=0.35, frameon=False,
    )
    fig.subplots_adjust(left=0.052, right=0.995, bottom=0.155, top=0.82, wspace=0.29)
    for ax, labels in zip(axes, panel_labels):
        place_labels_around_points(ax, labels)
    save_figure(fig, args.output_dir / "ablation_accuracy_efficiency_three_panel_mean_only_no_flex")
    plt.close(fig)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(5.45, 5.0))
        labels = plot_mean_only_no_flex_panel(ax, data, metric)
        fig.legend(
            handles=main_only_legend_handles(4), loc="upper center",
            bbox_to_anchor=(0.5, 0.99), ncol=4, columnspacing=1.05,
            handletextpad=0.3, frameon=False,
        )
        fig.subplots_adjust(left=0.145, right=0.98, bottom=0.13, top=0.74)
        place_labels_around_points(ax, labels)
        save_figure(fig, args.output_dir / f"ablation_accuracy_vs_{metric.stem}_mean_only_no_flex")
        plt.close(fig)

    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
