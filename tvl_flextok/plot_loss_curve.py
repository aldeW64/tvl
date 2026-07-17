#!/usr/bin/env python3
"""Plot TVL-FlexTok flow-loss curves from archived JSONL metric logs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smooth", type=int, default=21)
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.logs):
        parser.error("--labels must contain one label per metric log")

    labels = args.labels or [path.parent.name for path in args.logs]
    phases = [load_metrics(path) for path in args.logs]
    offsets = np.cumsum([0] + [len(phase) for phase in phases[:-1]])

    fig, (history_ax, modality_ax) = plt.subplots(
        2, 1, figsize=(11, 8), constrained_layout=True
    )

    colors = ["#2878B5", "#E07A1F", "#2A9D6F", "#8E5EA2"]
    for index, (phase, offset, label) in enumerate(zip(phases, offsets, labels)):
        epochs = np.arange(len(phase)) + offset
        values = np.array([row["val"]["flow"] for row in phase])
        color = colors[index % len(colors)]
        history_ax.plot(epochs, values, color=color, alpha=0.18, linewidth=0.8)
        history_ax.plot(
            epochs,
            moving_average(values, args.smooth),
            color=color,
            linewidth=2,
            label=label,
        )
        if index:
            history_ax.axvline(offset, color="#666666", linestyle="--", linewidth=1)

    history_ax.set_title("Validation flow loss across training phases")
    history_ax.set_xlabel("Cumulative epoch / optimizer update")
    history_ax.set_ylabel("Flow-matching loss")
    history_ax.grid(alpha=0.2)
    history_ax.legend(frameon=False)

    latest = phases[-1]
    latest_epochs = np.arange(len(latest))
    for key, label, color in (
        ("flow", "Combined", "#222222"),
        ("flow_vision", "Vision", "#D1495B"),
        ("flow_tactile", "Tactile", "#2878B5"),
    ):
        values = np.array([row["val"][key] for row in latest])
        modality_ax.plot(
            latest_epochs,
            moving_average(values, args.smooth),
            label=label,
            color=color,
            linewidth=2,
        )

    modality_ax.set_title(f"Latest phase by modality ({labels[-1]})")
    modality_ax.set_xlabel("Epoch / optimizer update")
    modality_ax.set_ylabel("Validation flow-matching loss")
    modality_ax.grid(alpha=0.2)
    modality_ax.legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
