#!/usr/bin/env python3
"""Run the controlled one-shot DMP experiments and supplementary analyses.

Usage
-----
Full experiment:
    python run_supplementary.py --config configs/supplementary_experiments.yml

Small end-to-end smoke test:
    python run_supplementary.py --config configs/supplementary_experiments.yml --quick

Rebuild CSV/LaTeX tables from existing run metrics without training:
    python run_supplementary.py --config configs/supplementary_experiments.yml --tables-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from src.supplementary.dmp_core import (
    ESNHyperParams,
    ESNModel,
    GramianDiagnostics,
    PreparedSeries,
    SplitConfig,
    activation_variance_scores,
    approximate_betweenness_scores,
    clone_pruned_model,
    compute_gramian_diagnostics,
    create_reservoir,
    load_prepared_series,
    magnitude_scores,
    model_size_metrics,
    random_keep_indices,
    select_top_nodes,
    spectral_radius_estimate,
)
from src.supplementary.evaluation import (
    autonomous_rollout,
    benchmark_inference_latency,
    cuda_peak_memory_mb,
    reset_peak_memory,
)
from src.supplementary.plotting import (
    METHOD_LABELS,
    configure_plots,
    plot_accuracy_latency,
    plot_dmp_diagnostics,
    plot_error_heatmap,
    plot_power_spectra,
    plot_prediction,
    plot_rollout,
    plot_table1_dataset_summary,
)
from src.supplementary.reporting import (
    TABLE1_METHODS,
    TABLE2_METHODS,
    TABLE3_METHODS,
    read_run_metrics,
    save_all_tables_and_statistics,
    write_metrics_row,
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def resolve_device(config: Mapping[str, object]) -> torch.device:
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_dtype(device: torch.device, config: Mapping[str, object]) -> torch.dtype:
    requested = str(config.get("dtype", "auto")).lower()
    if requested == "float64":
        return torch.float64
    if requested == "float32":
        return torch.float32
    return torch.float32 if device.type == "cuda" else torch.float64


def discover_datasets(config: Mapping[str, object]) -> List[Tuple[str, Path]]:
    """Discover all 1-D CSV files, with optional explicit name/path entries."""
    data_cfg = config.get("data", {}) or {}
    root = Path(str(data_cfg.get("root", "data")))
    explicit = data_cfg.get("datasets", []) or []

    datasets: List[Tuple[str, Path]] = []
    if explicit:
        for item in explicit:
            if isinstance(item, str):
                path = root / item
                datasets.append((path.stem, path))
            else:
                path = root / str(item["file"])
                datasets.append((str(item.get("name", path.stem)), path))
    else:
        pattern = str(data_cfg.get("glob", "*.csv"))
        datasets = [(path.stem, path) for path in sorted(root.glob(pattern))]

    if not datasets:
        raise FileNotFoundError(f"No CSV datasets were found below {root.resolve()}")

    missing = [str(path) for _, path in datasets if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing configured datasets: " + ", ".join(missing))

    return datasets


def save_resolved_config(config: dict, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "resolved_config.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


# ---------------------------------------------------------------------------
# Per-model fitting, evaluation, and artifacts
# ---------------------------------------------------------------------------


def predictions_frame(targets: torch.Tensor, predictions: torch.Tensor) -> pd.DataFrame:
    columns: Dict[str, np.ndarray] = {}
    target_np = targets.detach().cpu().numpy()
    pred_np = predictions.detach().cpu().numpy()
    for step in range(target_np.shape[0]):
        columns[f"target_h{step + 1}"] = target_np[step]
        columns[f"prediction_h{step + 1}"] = pred_np[step]
    return pd.DataFrame(columns)


def train_direct_model(
    model: ESNModel,
    series: PreparedSeries,
    horizon: int,
) -> Tuple[dict, object, float]:
    """Fit one readout and return direct metrics, evaluation object, train time."""
    start = time.perf_counter()
    bundle = model.collect_training_bundle(
        series.normalized,
        washout_end=series.washout_end,
        train_end=series.train_end,
        horizon=horizon,
    )
    model.fit_readout(bundle)
    readout_time = time.perf_counter() - start

    evaluation = model.evaluate_direct(
        series.normalized,
        test_start=series.train_end,
        horizon=horizon,
    )
    metrics = {
        "mse_at_h": evaluation.mse_at_h,
        "avg_mse_1_to_h": evaluation.avg_mse_1_to_h,
        "nrmse_at_h": evaluation.nrmse_at_h,
        "avg_nrmse_1_to_h": evaluation.avg_nrmse_1_to_h,
    }
    return metrics, evaluation, readout_time


def save_direct_artifacts(
    run_dir: Path,
    method: str,
    dataset: str,
    seed: int,
    horizon: int,
    evaluation: object,
    save_plots: bool,
    max_plot_points: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_frame(evaluation.targets, evaluation.predictions).to_csv(
        run_dir / "predictions_all_horizons.csv",
        index=False,
        float_format="%.10g",
    )

    if not save_plots:
        return

    target_h = evaluation.targets[-1].detach().cpu().numpy()
    prediction_h = evaluation.predictions[-1].detach().cpu().numpy()
    title = f"{METHOD_LABELS.get(method, method)} | {dataset} | seed={seed} | h={horizon}"
    plot_prediction(
        target_h,
        prediction_h,
        run_dir / "prediction_at_final_horizon.png",
        title=title,
        max_points=max_plot_points,
    )
    plot_error_heatmap(
        target_h,
        prediction_h,
        run_dir / "error_at_final_horizon.png",
        title=title,
        max_points=max_plot_points,
    )


def direct_run_directory(
    output_root: Path,
    dataset: str,
    horizon: int,
    seed: int,
    method: str,
) -> Path:
    return output_root / "runs" / dataset / f"direct_h{horizon}" / f"seed_{seed}" / method


def dynamical_run_directory(
    output_root: Path,
    dataset: str,
    seed: int,
    method: str,
) -> Path:
    return output_root / "runs" / dataset / "dynamical_h1" / f"seed_{seed}" / method


def all_direct_methods(centrality_enabled: bool) -> List[str]:
    methods = [
        "full_esn",
        "fresh_small_esn",
        "random_no_rescale",
        "random_rho",
        "magnitude_rho",
        "activation_variance_rho",
        "jacobian_column_energy_rho",
        "dmp_no_rescale",
        "dmp_rho",
    ]
    if centrality_enabled:
        methods.insert(-2, "betweenness_rho")
    return methods


def suite_is_complete(
    output_root: Path,
    dataset: str,
    horizon: int,
    seed: int,
    methods: Iterable[str],
    experiment: str,
) -> bool:
    if experiment == "direct":
        paths = [direct_run_directory(output_root, dataset, horizon, seed, m) / "metrics.csv" for m in methods]
    else:
        paths = [dynamical_run_directory(output_root, dataset, seed, m) / "metrics.csv" for m in methods]
    return all(path.exists() for path in paths)


# ---------------------------------------------------------------------------
# Table 1 + Table 2 experiment suite
# ---------------------------------------------------------------------------


def run_direct_suite(
    series: PreparedSeries,
    seed: int,
    reservoir_size: int,
    prune_ratio: float,
    horizon: int,
    hyper: ESNHyperParams,
    dmp_energy_tau: float,
    output_root: Path,
    runtime_cfg: Mapping[str, object],
    centrality_cfg: Mapping[str, object],
    save_plots: bool,
    max_plot_points: int,
) -> None:
    """Run all matched baselines and DMP ablations from one shared reservoir."""

    target_size = max(1, int(round((1.0 - prune_ratio) * reservoir_size)))
    centrality_enabled = bool(centrality_cfg.get("enabled", True))
    methods = all_direct_methods(centrality_enabled)

    if bool(runtime_cfg.get("skip_existing", True)) and suite_is_complete(
        output_root,
        series.name,
        horizon,
        seed,
        methods,
        experiment="direct",
    ):
        print(f"  [skip] direct suite already exists: {series.name}, seed={seed}")
        return

    print(f"  [run] direct suite: dataset={series.name}, seed={seed}, h={horizon}")
    reset_peak_memory(series.normalized.device)

    # ------------------------------------------------------------------
    # Shared full ESN.  Every pruning baseline below starts from these
    # exact W and Win tensors for a fair paired comparison.
    # ------------------------------------------------------------------
    full_start = time.perf_counter()
    full_model = create_reservoir(
        reservoir_size=reservoir_size,
        seed=seed,
        hyper=hyper,
        device=series.normalized.device,
        dtype=series.normalized.dtype,
    )
    full_bundle = full_model.collect_training_bundle(
        series.normalized,
        washout_end=series.washout_end,
        train_end=series.train_end,
        horizon=horizon,
    )
    readout_start = time.perf_counter()
    full_model.fit_readout(full_bundle)
    full_readout_time = time.perf_counter() - readout_start
    full_build_time = time.perf_counter() - full_start
    full_eval = full_model.evaluate_direct(series.normalized, series.train_end, horizon)

    # DMP and Jacobian column-energy use the same trajectory and one exact Gramian.
    gramian_start = time.perf_counter()
    diagnostics = compute_gramian_diagnostics(
        W=full_model.W,
        derivatives=full_bundle.derivatives,
        leak_rate=hyper.leak_rate,
        energy_tau=dmp_energy_tau,
    )
    gramian_time = time.perf_counter() - gramian_start

    # Other matched scores are computed from the same W and trajectory.
    magnitude_start = time.perf_counter()
    magnitude = magnitude_scores(full_model.W)
    magnitude_time = time.perf_counter() - magnitude_start

    variance_start = time.perf_counter()
    activation_variance = activation_variance_scores(full_bundle.states)
    variance_time = time.perf_counter() - variance_start

    random_start = time.perf_counter()
    random_keep = random_keep_indices(
        reservoir_size=reservoir_size,
        target_size=target_size,
        seed=seed + 7919,
        device=full_model.device,
    )
    random_score_time = time.perf_counter() - random_start

    dmp_keep = select_top_nodes(diagnostics.dmp_scores, target_size)
    jacobian_keep = select_top_nodes(diagnostics.jacobian_column_energy_scores, target_size)
    magnitude_keep = select_top_nodes(magnitude, target_size)
    variance_keep = select_top_nodes(activation_variance, target_size)

    betweenness_keep = None
    betweenness_time = 0.0
    betweenness_scores = None
    if centrality_enabled:
        centrality_start = time.perf_counter()
        betweenness_scores = approximate_betweenness_scores(
            full_model.W,
            sample_sources=int(centrality_cfg.get("sample_sources", 128)),
            strongest_edges_per_node=int(centrality_cfg.get("strongest_edges_per_node", 20)),
            seed=seed,
        )
        betweenness_keep = select_top_nodes(betweenness_scores, target_size)
        betweenness_time = time.perf_counter() - centrality_start

    # Save DMP diagnostics and all score vectors once per seed.
    diagnostics_dir = direct_run_directory(
        output_root,
        series.name,
        horizon,
        seed,
        "dmp_diagnostics",
    )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    score_frame = pd.DataFrame(
        {
            "neuron": np.arange(reservoir_size),
            "dmp_score": diagnostics.dmp_scores.detach().cpu().numpy(),
            "jacobian_column_energy_score": diagnostics.jacobian_column_energy_scores.detach().cpu().numpy(),
            "magnitude_score": magnitude.detach().cpu().numpy(),
            "activation_variance_score": activation_variance.detach().cpu().numpy(),
            "dmp_kept": np.isin(np.arange(reservoir_size), dmp_keep.detach().cpu().numpy()).astype(int),
        }
    )
    if betweenness_scores is not None:
        score_frame["betweenness_score"] = betweenness_scores.detach().cpu().numpy()
    score_frame.to_csv(diagnostics_dir / "node_scores.csv", index=False, float_format="%.10g")
    pd.DataFrame(
        {
            "mode": np.arange(1, reservoir_size + 1),
            "eigenvalue": diagnostics.eigenvalues.detach().cpu().numpy(),
            "normalized_eigenvalue": diagnostics.normalized_eigenvalues.detach().cpu().numpy(),
            "cumulative_energy": diagnostics.cumulative_energy.detach().cpu().numpy(),
        }
    ).to_csv(diagnostics_dir / "gramian_eigenspectrum.csv", index=False, float_format="%.10g")

    if save_plots and diagnostics.selected_rank > 0:
        plot_dmp_diagnostics(
            scores=diagnostics.dmp_scores.detach().cpu().numpy(),
            eigenvalues=diagnostics.eigenvalues.detach().cpu().numpy(),
            cumulative_energy=diagnostics.cumulative_energy.detach().cpu().numpy(),
            selected_rank=diagnostics.selected_rank,
            energy_tau=dmp_energy_tau,
            save_dir=diagnostics_dir,
            title_prefix=f"{series.name}, seed={seed}, h={horizon}",
        )

    # Each tuple is: method, model factory, method-specific offline score time.
    factories: List[Tuple[str, object, float]] = [
        ("full_esn", lambda: full_model, 0.0),
        (
            "fresh_small_esn",
            lambda: create_reservoir(
                reservoir_size=target_size,
                seed=seed,
                hyper=hyper,
                device=full_model.device,
                dtype=full_model.dtype,
            ),
            0.0,
        ),
        ("random_no_rescale", lambda: clone_pruned_model(full_model, random_keep, False), random_score_time),
        ("random_rho", lambda: clone_pruned_model(full_model, random_keep, True), random_score_time),
        ("magnitude_rho", lambda: clone_pruned_model(full_model, magnitude_keep, True), magnitude_time),
        ("activation_variance_rho", lambda: clone_pruned_model(full_model, variance_keep, True), variance_time),
        ("jacobian_column_energy_rho", lambda: clone_pruned_model(full_model, jacobian_keep, True), gramian_time),
    ]
    if centrality_enabled and betweenness_keep is not None:
        factories.append(
            ("betweenness_rho", lambda: clone_pruned_model(full_model, betweenness_keep, True), betweenness_time)
        )
    factories.extend(
        [
            ("dmp_no_rescale", lambda: clone_pruned_model(full_model, dmp_keep, False), gramian_time),
            ("dmp_rho", lambda: clone_pruned_model(full_model, dmp_keep, True), gramian_time),
        ]
    )

    benchmark_steps = int(runtime_cfg.get("benchmark_steps", 300))
    warmup_repeats = int(runtime_cfg.get("warmup_repeats", 3))
    measured_repeats = int(runtime_cfg.get("measured_repeats", 20))

    rows: List[dict] = []
    evaluations: Dict[str, object] = {}

    for method, factory, score_time in factories:
        method_start = time.perf_counter()

        if method == "full_esn":
            model = full_model
            readout_time = full_readout_time
            evaluation = full_eval
            model_build_time = full_build_time
            clone_and_rescale_time = 0.0
        else:
            clone_start = time.perf_counter()
            model = factory()
            clone_and_rescale_time = time.perf_counter() - clone_start

            metrics, evaluation, readout_time = train_direct_model(model, series, horizon)
            model_build_time = time.perf_counter() - method_start

        if method == "full_esn":
            direct_metrics = {
                "mse_at_h": full_eval.mse_at_h,
                "avg_mse_1_to_h": full_eval.avg_mse_1_to_h,
                "nrmse_at_h": full_eval.nrmse_at_h,
                "avg_nrmse_1_to_h": full_eval.avg_nrmse_1_to_h,
            }
        else:
            direct_metrics = metrics

        latency = benchmark_inference_latency(
            model=model,
            data=series.normalized,
            start_index=series.train_end,
            benchmark_steps=benchmark_steps,
            warmup_repeats=warmup_repeats,
            measured_repeats=measured_repeats,
        )
        size_metrics = model_size_metrics(model)

        offline_pruning = float(score_time + clone_and_rescale_time) if "pruning" not in method else 0.0
        # For clarity, every pruned method records score/selection + cloning/rescaling.
        if method in {"full_esn", "fresh_small_esn"}:
            offline_pruning = 0.0

        row = {
            "experiment": "direct",
            "dataset": series.name,
            "dataset_file": str(series.path),
            "seed": int(seed),
            "method": method,
            "horizon": int(horizon),
            "original_reservoir_size": int(reservoir_size),
            "prune_ratio": float(prune_ratio if method != "full_esn" else 0.0),
            "energy_tau": float(dmp_energy_tau),
            "selected_dmp_rank": int(diagnostics.selected_rank),
            "leak_rate": float(hyper.leak_rate),
            "spectral_radius_target": float(hyper.spectral_radius),
            "spectral_radius_final_estimate": float(
                spectral_radius_estimate(model.W, hyper.spectral_radius_iterations)
            ),
            "ridge_alpha": float(hyper.ridge_alpha),
            "input_scaling": float(hyper.input_scaling),
            "bias_scaling": float(hyper.bias_scaling),
            "zero_fraction": float(hyper.zero_fraction),
            "washout_end": int(series.washout_end),
            "train_end": int(series.train_end),
            "test_end": int(series.test_end),
            **direct_metrics,
            **size_metrics,
            "latency_us_per_step": float(latency.median_us_per_step),
            "latency_iqr_us_per_step": float(latency.iqr_us_per_step),
            "speedup_vs_full": np.nan,  # filled after the full latency is known
            "offline_pruning_s": float(offline_pruning),
            "readout_training_s": float(readout_time),
            "total_model_build_s": float(model_build_time),
            "cuda_peak_memory_mb": float(cuda_peak_memory_mb(model.device)),
            "centrality_sample_sources": int(centrality_cfg.get("sample_sources", 128)),
            "centrality_strongest_edges_per_node": int(
                centrality_cfg.get("strongest_edges_per_node", 20)
            ),
            "one_shot_pruning": True,
            "progressive": False,
        }
        rows.append(row)
        evaluations[method] = evaluation

    full_latency = next(row["latency_us_per_step"] for row in rows if row["method"] == "full_esn")
    for row in rows:
        row["speedup_vs_full"] = float(full_latency / max(row["latency_us_per_step"], 1e-12))
        run_dir = direct_run_directory(output_root, series.name, horizon, seed, row["method"])
        write_metrics_row(run_dir / "metrics.csv", row)
        save_direct_artifacts(
            run_dir=run_dir,
            method=row["method"],
            dataset=series.name,
            seed=seed,
            horizon=horizon,
            evaluation=evaluations[row["method"]],
            save_plots=save_plots,
            max_plot_points=max_plot_points,
        )


# ---------------------------------------------------------------------------
# Table 3: autonomous dynamical fidelity
# ---------------------------------------------------------------------------


def run_dynamical_suite(
    series: PreparedSeries,
    seed: int,
    reservoir_size: int,
    prune_ratio: float,
    hyper: ESNHyperParams,
    dmp_energy_tau: float,
    output_root: Path,
    dynamics_cfg: Mapping[str, object],
    save_plots: bool,
) -> None:
    methods = TABLE3_METHODS
    if bool(dynamics_cfg.get("skip_existing", True)) and suite_is_complete(
        output_root,
        series.name,
        1,
        seed,
        methods,
        experiment="dynamical",
    ):
        print(f"  [skip] dynamical suite already exists: {series.name}, seed={seed}")
        return

    print(f"  [run] dynamical suite: dataset={series.name}, seed={seed}")
    target_size = max(1, int(round((1.0 - prune_ratio) * reservoir_size)))

    full_model = create_reservoir(
        reservoir_size=reservoir_size,
        seed=seed,
        hyper=hyper,
        device=series.normalized.device,
        dtype=series.normalized.dtype,
    )
    full_bundle = full_model.collect_training_bundle(
        series.normalized,
        washout_end=series.washout_end,
        train_end=series.train_end,
        horizon=1,
    )
    full_model.fit_readout(full_bundle)

    score_start = time.perf_counter()
    diagnostics = compute_gramian_diagnostics(
        full_model.W,
        full_bundle.derivatives,
        leak_rate=hyper.leak_rate,
        energy_tau=dmp_energy_tau,
    )
    dmp_score_time = time.perf_counter() - score_start
    dmp_keep = select_top_nodes(diagnostics.dmp_scores, target_size)
    random_keep = random_keep_indices(
        reservoir_size,
        target_size,
        seed=seed + 7919,
        device=full_model.device,
    )

    model_factories = {
        "full_esn": lambda: full_model,
        "fresh_small_esn": lambda: create_reservoir(
            target_size,
            seed,
            hyper,
            full_model.device,
            full_model.dtype,
        ),
        "random_rho": lambda: clone_pruned_model(full_model, random_keep, True),
        "dmp_rho": lambda: clone_pruned_model(full_model, dmp_keep, True),
    }

    rollout_steps = int(dynamics_cfg.get("rollout_steps", 500))
    threshold = float(dynamics_cfg.get("valid_threshold_std", 0.40))
    max_lag = int(dynamics_cfg.get("autocorrelation_max_lag", 100))
    prediction_map: Dict[str, np.ndarray] = {}
    target_for_plot: Optional[np.ndarray] = None

    for method in methods:
        start = time.perf_counter()
        model = model_factories[method]()
        if method != "full_esn":
            bundle = model.collect_training_bundle(
                series.normalized,
                washout_end=series.washout_end,
                train_end=series.train_end,
                horizon=1,
            )
            model.fit_readout(bundle)
        build_time = time.perf_counter() - start

        rollout = autonomous_rollout(
            model,
            series.normalized,
            test_start=series.train_end,
            rollout_steps=rollout_steps,
            valid_threshold_std=threshold,
            autocorrelation_max_lag=max_lag,
        )

        row = {
            "experiment": "dynamical",
            "dataset": series.name,
            "dataset_file": str(series.path),
            "seed": int(seed),
            "method": method,
            "horizon": 1,
            "original_reservoir_size": int(reservoir_size),
            "prune_ratio": float(prune_ratio if method != "full_esn" else 0.0),
            "energy_tau": float(dmp_energy_tau),
            "selected_dmp_rank": int(diagnostics.selected_rank),
            "rollout_steps": int(len(rollout.target)),
            "valid_threshold_std": threshold,
            "one_step_nrmse": rollout.one_step_nrmse,
            "rollout_nrmse": rollout.rollout_nrmse,
            "valid_prediction_steps": rollout.valid_prediction_steps,
            "valid_prediction_fraction": rollout.valid_prediction_fraction,
            "psd_distance": rollout.psd_distance,
            "autocorrelation_distance": rollout.autocorrelation_distance,
            "offline_pruning_s": float(dmp_score_time if method == "dmp_rho" else 0.0),
            "total_model_build_s": float(build_time),
            **model_size_metrics(model),
            "one_shot_pruning": True,
            "progressive": False,
        }

        run_dir = dynamical_run_directory(output_root, series.name, seed, method)
        write_metrics_row(run_dir / "metrics.csv", row)
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "target": rollout.target,
                "prediction": rollout.prediction,
                "absolute_error": np.abs(rollout.target - rollout.prediction),
            }
        ).to_csv(run_dir / "autonomous_rollout.csv", index=False, float_format="%.10g")

        prediction_map[method] = rollout.prediction
        target_for_plot = rollout.target

    if save_plots and target_for_plot is not None:
        plot_dir = output_root / "plots" / "table3_dynamical" / series.name / f"seed_{seed}"
        plot_rollout(
            target_for_plot,
            prediction_map,
            plot_dir / "autonomous_rollout_comparison.png",
            title=f"Autonomous rollout: {series.name}, seed={seed}",
            max_points=rollout_steps,
        )
        plot_power_spectra(
            target_for_plot,
            prediction_map,
            plot_dir / "power_spectrum_comparison.png",
            title=f"Autonomous power spectra: {series.name}, seed={seed}",
        )


# ---------------------------------------------------------------------------
# Final table and aggregate plots
# ---------------------------------------------------------------------------


def build_outputs(config: dict, output_root: Path) -> None:
    all_metrics = read_run_metrics(output_root)
    if all_metrics.empty:
        print("No run metrics were found; tables were not generated.")
        return

    experiment_cfg = config["experiment"]
    horizon = int(experiment_cfg.get("direct_horizon", 20))
    primary_metric = str(experiment_cfg.get("primary_metric", "nrmse_at_h"))
    tables = save_all_tables_and_statistics(
        all_metrics=all_metrics,
        output_root=output_root,
        primary_metric=primary_metric,
        horizon=horizon,
    )

    # Aggregate figures are produced from the same CSV values used in LaTeX.
    aggregate_plot_dir = output_root / "plots" / "aggregate"
    for dataset in sorted(tables["table1_aggregate"]["dataset"].astype(str).unique()):
        plot_table1_dataset_summary(
            tables["table1_aggregate"],
            dataset=dataset,
            metric=primary_metric,
            save_path=aggregate_plot_dir / f"table1_{dataset}_{primary_metric}.png",
        )

    if not tables["table2_aggregate"].empty:
        for dataset in sorted(tables["table2_aggregate"]["dataset"].astype(str).unique()):
            plot_accuracy_latency(
                tables["table2_aggregate"],
                dataset=dataset,
                save_path=aggregate_plot_dir / f"table2_accuracy_latency_{dataset}.png",
            )

    print(f"\nSaved CSV tables to:   {output_root / 'csv'}")
    print(f"Saved LaTeX tables to: {output_root / 'latex'}")
    print(f"Saved plots to:        {output_root / 'plots'}")


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/supplementary_experiments.yml", help="Path to the YAML configuration")
    parser.add_argument("--quick", action="store_true", help="Run one small seed/dataset smoke test")
    parser.add_argument("--tables-only", action="store_true", help="Only rebuild CSV and LaTeX outputs")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    output_root = Path(str(config.get("output", {}).get("root", "results/supplementary_experiments")))
    configure_plots(
        font_size=int(config.get("plots", {}).get("font_size", 17)),
        dpi=int(config.get("plots", {}).get("dpi", 320)),
    )
    save_resolved_config(config, output_root)

    if args.tables_only:
        build_outputs(config, output_root)
        return

    device = resolve_device(config.get("runtime", {}))
    dtype = resolve_dtype(device, config.get("runtime", {}))
    datasets = discover_datasets(config)

    split_cfg = config.get("split", {})
    split = SplitConfig(
        washout_ratio=float(split_cfg.get("washout_ratio", 0.10)),
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        test_ratio=float(split_cfg.get("test_ratio", 0.20)),
    )

    hyper_cfg = config.get("esn", {})
    hyper = ESNHyperParams(
        leak_rate=float(hyper_cfg.get("leak_rate", 0.4)),
        spectral_radius=float(hyper_cfg.get("spectral_radius", 0.95)),
        input_scaling=float(hyper_cfg.get("input_scaling", 0.5)),
        bias_scaling=float(hyper_cfg.get("bias_scaling", 0.2)),
        zero_fraction=float(hyper_cfg.get("zero_fraction", 0.10)),
        ridge_alpha=float(hyper_cfg.get("ridge_alpha", 1e-6)),
        spectral_radius_iterations=int(hyper_cfg.get("spectral_radius_iterations", 80)),
    )

    exp_cfg = config.get("experiment", {})
    reservoir_size = int(exp_cfg.get("reservoir_size", 1000))
    prune_ratio = float(exp_cfg.get("prune_ratio", 0.20))
    horizon = int(exp_cfg.get("direct_horizon", 20))
    seeds = [int(x) for x in exp_cfg.get("seeds", list(range(10)))]
    dmp_energy_tau = float(config.get("dmp", {}).get("energy_tau", 0.90))

    plot_cfg = config.get("plots", {})
    save_plots = bool(plot_cfg.get("save_per_run_plots", True))
    max_plot_points = int(plot_cfg.get("max_points", 500))

    dynamics_cfg = config.get("dynamical_evaluation", {})
    dynamical_names = {str(x).lower() for x in dynamics_cfg.get("datasets", ["mackey"])}

    if args.quick:
        datasets = datasets[:1]
        seeds = seeds[:1]
        reservoir_size = min(reservoir_size, 120)
        horizon = min(horizon, 5)
        config["experiment"]["direct_horizon"] = horizon
        config["experiment"]["reservoir_size"] = reservoir_size
        config.setdefault("centrality", {})["enabled"] = False
        config.setdefault("runtime", {})["benchmark_steps"] = 30
        config.setdefault("runtime", {})["warmup_repeats"] = 1
        config.setdefault("runtime", {})["measured_repeats"] = 3
        config.setdefault("dynamical_evaluation", {})["rollout_steps"] = 60
        save_resolved_config(config, output_root)

    print("=" * 78)
    print("Controlled one-shot DMP experiment suite")
    print(f"Device: {device} | dtype: {dtype}")
    print(f"Datasets: {[name for name, _ in datasets]}")
    print(f"Seeds: {seeds}")
    print(f"Reservoir: {reservoir_size} -> {int(round((1-prune_ratio)*reservoir_size))}")
    print(f"Direct horizon: {horizon} | DMP energy threshold: {dmp_energy_tau}")
    print("Progressive pruning: False (one-shot only)")
    print("=" * 78)

    prepared: Dict[str, PreparedSeries] = {}
    for name, path in datasets:
        prepared[name] = load_prepared_series(
            path,
            name=name,
            split=split,
            device=device,
            dtype=dtype,
        )
        print(
            f"Loaded {name}: n={len(prepared[name].normalized)}, "
            f"washout_end={prepared[name].washout_end}, "
            f"train_end={prepared[name].train_end}"
        )

    for name, series in prepared.items():
        print(f"\n### Dataset: {name}")
        for seed in seeds:
            run_direct_suite(
                series=series,
                seed=seed,
                reservoir_size=reservoir_size,
                prune_ratio=prune_ratio,
                horizon=horizon,
                hyper=hyper,
                dmp_energy_tau=dmp_energy_tau,
                output_root=output_root,
                runtime_cfg=config.get("runtime", {}),
                centrality_cfg=config.get("centrality", {}),
                save_plots=save_plots,
                max_plot_points=max_plot_points,
            )

            normalized_name = name.lower()
            if bool(dynamics_cfg.get("enabled", True)) and (
                normalized_name in dynamical_names
                or any(token in normalized_name for token in dynamical_names)
            ):
                run_dynamical_suite(
                    series=series,
                    seed=seed,
                    reservoir_size=reservoir_size,
                    prune_ratio=prune_ratio,
                    hyper=hyper,
                    dmp_energy_tau=dmp_energy_tau,
                    output_root=output_root,
                    dynamics_cfg=dynamics_cfg,
                    save_plots=save_plots,
                )

    build_outputs(config, output_root)


if __name__ == "__main__":
    main()
