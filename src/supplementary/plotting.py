"""High-resolution plots for the controlled DMP experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "full_esn": "Full ESN",
    "fresh_small_esn": "Fresh smaller ESN",
    "random_rho": "Random + $\\rho$",
    "magnitude_rho": "Magnitude + $\\rho$",
    "activation_variance_rho": "Activation variance + $\\rho$",
    "jacobian_column_energy_rho": "Jacobian col. energy + $\\rho$",
    "betweenness_rho": "Betweenness + $\\rho$",
    "dmp_rho": "DMP + $\\rho$",
    "random_no_rescale": "Random, no rescale",
    "dmp_no_rescale": "DMP, no rescale",
}


def configure_plots(font_size: int = 17, dpi: int = 320) -> None:
    """Apply one readable paper-style configuration to every generated plot."""
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size + 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 2,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_prediction(
    target: np.ndarray,
    prediction: np.ndarray,
    save_path: Path,
    title: str,
    max_points: int = 500,
) -> None:
    ensure_parent(save_path)
    n = min(len(target), len(prediction), max_points)
    if n <= 1:
        return

    x = np.arange(n)
    target = np.asarray(target)[:n]
    prediction = np.asarray(prediction)[:n]
    mse = float(np.mean((target - prediction) ** 2))

    fig, ax = plt.subplots(figsize=(14.5, 5.2))
    ax.plot(x, target, label="Target", linewidth=2.4)
    ax.plot(x, prediction, label="Prediction", linewidth=2.1, alpha=0.90)
    ax.fill_between(x, target, prediction, alpha=0.10)
    ax.set_title(f"{title}\nMSE={mse:.4g}")
    ax.set_xlabel("Test time index")
    ax.set_ylabel("Normalized value")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_error_heatmap(
    target: np.ndarray,
    prediction: np.ndarray,
    save_path: Path,
    title: str,
    max_points: int = 1200,
) -> None:
    ensure_parent(save_path)
    n = min(len(target), len(prediction), max_points)
    if n <= 1:
        return

    error = np.abs(np.asarray(target)[:n] - np.asarray(prediction)[:n])
    vmax = max(float(np.percentile(error, 99)), 1e-12)

    fig, ax = plt.subplots(figsize=(14.5, 3.2))
    image = ax.imshow(
        error.reshape(1, -1),
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Absolute error")
    ax.set_title(f"{title} | maximum error={np.max(error):.4g}")
    ax.set_xlabel("Test time index")
    ax.set_yticks([0])
    ax.set_yticklabels(["|e|"])
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_dmp_diagnostics(
    scores: np.ndarray,
    eigenvalues: np.ndarray,
    cumulative_energy: np.ndarray,
    selected_rank: int,
    energy_tau: float,
    save_dir: Path,
    title_prefix: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    order = np.argsort(scores)[::-1]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.plot(np.arange(len(scores)), scores[order], linewidth=2.2)
    ax.set_xlabel("Neuron rank")
    ax.set_ylabel("DMP score")
    ax.set_title(f"{title_prefix}: sorted one-shot DMP scores")
    ax.annotate(
        f"highest={scores[order[0]]:.3g}\nlowest={scores[order[-1]]:.3g}",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(save_dir / "dmp_scores_sorted.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, linewidth=2.2)
    ax.axvline(selected_rank, linestyle="--", linewidth=2.0, label=f"retained rank r={selected_rank}")
    ax.set_yscale("log")
    ax.set_xlabel("Mode rank")
    ax.set_ylabel("Gramian eigenvalue")
    ax.set_title(f"{title_prefix}: Jacobian-Gramian spectrum")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / "gramian_eigenspectrum.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    modes = np.arange(1, len(cumulative_energy) + 1)
    ax.plot(modes, cumulative_energy, linewidth=2.2)
    ax.axhline(energy_tau, linestyle="--", linewidth=2.0, label=f"energy threshold={energy_tau:.2f}")
    ax.axvline(selected_rank, linestyle="--", linewidth=2.0, label=f"selected rank={selected_rank}")
    ax.scatter([selected_rank], [cumulative_energy[selected_rank - 1]], s=90, zorder=4)
    ax.annotate(
        f"r={selected_rank}",
        xy=(selected_rank, cumulative_energy[selected_rank - 1]),
        xytext=(12, -28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_xlabel("Mode rank")
    ax.set_ylabel("Cumulative normalized energy")
    ax.set_ylim(0.0, 1.03)
    ax.set_title(f"{title_prefix}: retained dynamical energy")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_dir / "gramian_cumulative_energy.png", bbox_inches="tight")
    plt.close(fig)


def plot_table1_dataset_summary(
    aggregate: pd.DataFrame,
    dataset: str,
    metric: str,
    save_path: Path,
) -> None:
    ensure_parent(save_path)
    subset = aggregate[aggregate["dataset"] == dataset].copy()
    if subset.empty:
        return

    subset = subset.sort_values("method_order")
    means = subset[f"{metric}_mean"].to_numpy(dtype=float)
    stds = subset[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
    labels = [METHOD_LABELS.get(m, m) for m in subset["method"]]

    fig, ax = plt.subplots(figsize=(15.5, 7.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"Matched pruning comparison: {dataset}")

    for bar, value in zip(bars, means):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=max(11, plt.rcParams["font.size"] - 4),
        )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_latency(
    aggregate: pd.DataFrame,
    dataset: str,
    save_path: Path,
) -> None:
    ensure_parent(save_path)
    subset = aggregate[aggregate["dataset"] == dataset].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    x = subset["latency_us_per_step_mean"].to_numpy(dtype=float)
    y = subset["nrmse_at_h_mean"].to_numpy(dtype=float)
    ax.scatter(x, y, s=120)

    for _, row in subset.iterrows():
        ax.annotate(
            METHOD_LABELS.get(row["method"], row["method"]),
            xy=(row["latency_us_per_step_mean"], row["nrmse_at_h_mean"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=max(11, plt.rcParams["font.size"] - 4),
        )

    ax.set_xlabel("Median inference latency (µs / step)")
    ax.set_ylabel("NRMSE at $h$")
    ax.set_title(f"Accuracy–latency trade-off: {dataset}")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_rollout(
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    save_path: Path,
    title: str,
    max_points: int = 500,
) -> None:
    ensure_parent(save_path)
    n = min(max_points, len(target), *(len(v) for v in predictions.values()))
    if n <= 1:
        return

    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(15.0, 6.0))
    ax.plot(x, np.asarray(target)[:n], label="Target", linewidth=2.8)
    for method, values in predictions.items():
        ax.plot(
            x,
            np.asarray(values)[:n],
            label=METHOD_LABELS.get(method, method),
            linewidth=1.9,
            alpha=0.85,
        )
    ax.set_xlabel("Autonomous rollout step")
    ax.set_ylabel("Normalized value")
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_power_spectra(
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    save_path: Path,
    title: str,
) -> None:
    ensure_parent(save_path)

    def spectrum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(values, dtype=float)
        x = x - np.mean(x)
        power = np.abs(np.fft.rfft(x)) ** 2
        frequencies = np.fft.rfftfreq(len(x))
        power = power / max(float(np.sum(power)), 1e-12)
        return frequencies, power

    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    f, p = spectrum(target)
    ax.plot(f, p, label="Target", linewidth=2.8)
    for method, values in predictions.items():
        f, p = spectrum(values)
        ax.plot(f, p, label=METHOD_LABELS.get(method, method), linewidth=1.9, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Normalized spectral power")
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
