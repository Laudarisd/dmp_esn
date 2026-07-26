"""CSV aggregation and paper-ready LaTeX table generation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .plotting import METHOD_LABELS


TABLE1_METHODS = [
    "full_esn",
    "fresh_small_esn",
    "random_rho",
    "magnitude_rho",
    "activation_variance_rho",
    "jacobian_column_energy_rho",
    "betweenness_rho",
    "dmp_rho",
]

TABLE2_METHODS = [
    "full_esn",
    "fresh_small_esn",
    "random_no_rescale",
    "random_rho",
    "dmp_no_rescale",
    "dmp_rho",
]

TABLE3_METHODS = [
    "full_esn",
    "fresh_small_esn",
    "random_rho",
    "dmp_rho",
]


LATEX_METHOD_LABELS = {
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


DATASET_LABELS = {
    "mackey": "Mackey--Glass",
    "mackey_glass": "Mackey--Glass",
    "mackey_glass_1d": "Mackey--Glass",
    "electricity": "Electricity",
    "electricity_consumption_1d": "Electricity",
    "temperature": "Temperature",
    "wind": "Wind",
    "solar": "Solar",
    "solar_consumption_1d": "Solar",
    "traffic": "Traffic",
}


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def dataset_label(name: str) -> str:
    return DATASET_LABELS.get(str(name), latex_escape(str(name).replace("_", " ").title()))


def method_label(name: str) -> str:
    return LATEX_METHOD_LABELS.get(name, latex_escape(name))


def write_metrics_row(path: Path, row: Mapping[str, object]) -> None:
    """Write one self-contained metrics.csv file for a model run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row)]).to_csv(path, index=False, float_format="%.10g")


def read_run_metrics(results_root: Path) -> pd.DataFrame:
    """Load all per-run metrics files below the experiment output root."""
    rows: List[dict] = []
    for path in sorted(results_root.glob("runs/**/metrics.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        # Older runs used the mathematically ambiguous label
        # ``jacobian_diagonal_rho``.  The score is diag(G), which equals the
        # trajectory-averaged squared Jacobian column norm, so existing result
        # files are normalised to the clearer public name on load.
        if row.get("method") == "jacobian_diagonal_rho":
            row["method"] = "jacobian_column_energy_rho"
        row["metrics_path"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metrics(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate seed-wise values into mean/std/CI/count columns."""
    if frame.empty:
        return pd.DataFrame()

    records: List[dict] = []
    grouped = frame.groupby(list(group_columns), dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record["seed_count"] = int(group["seed"].nunique()) if "seed" in group else len(group)

        for metric in metric_columns:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                record[f"{metric}_mean"] = np.nan
                record[f"{metric}_std"] = np.nan
                record[f"{metric}_ci95_low"] = np.nan
                record[f"{metric}_ci95_high"] = np.nan
                continue

            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            half = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_ci95_low"] = mean - half
            record[f"{metric}_ci95_high"] = mean + half

        records.append(record)

    return pd.DataFrame(records)


def paired_statistics(
    table1_seedwise: pd.DataFrame,
    reference_method: str = "full_esn",
    metric: str = "nrmse_at_h",
    bootstrap_samples: int = 5000,
    seed: int = 12345,
) -> pd.DataFrame:
    """Compute paired differences, bootstrap CIs, win rates, and Wilcoxon p-values."""
    if table1_seedwise.empty:
        return pd.DataFrame()

    try:
        from scipy.stats import wilcoxon
    except Exception:  # pragma: no cover - scipy is listed as a requirement
        wilcoxon = None

    rng = np.random.default_rng(seed)
    records: List[dict] = []

    for dataset, dataset_group in table1_seedwise.groupby("dataset"):
        reference = dataset_group[dataset_group["method"] == reference_method][["seed", metric]]
        reference = reference.rename(columns={metric: "reference_value"})

        for method in TABLE1_METHODS:
            if method == reference_method:
                continue
            candidate = dataset_group[dataset_group["method"] == method][["seed", metric]]
            candidate = candidate.rename(columns={metric: "candidate_value"})
            paired = reference.merge(candidate, on="seed", how="inner").dropna()
            if paired.empty:
                continue

            # Negative difference means the candidate improves on the full ESN.
            differences = (
                paired["candidate_value"].to_numpy(dtype=float)
                - paired["reference_value"].to_numpy(dtype=float)
            )
            n = len(differences)

            if n == 1:
                ci_low = ci_high = float(differences[0])
            else:
                samples = rng.choice(differences, size=(bootstrap_samples, n), replace=True)
                sample_means = np.mean(samples, axis=1)
                ci_low, ci_high = np.percentile(sample_means, [2.5, 97.5])

            p_value = np.nan
            if wilcoxon is not None and n >= 2 and not np.allclose(differences, 0.0):
                try:
                    p_value = float(wilcoxon(differences).pvalue)
                except Exception:
                    p_value = np.nan

            records.append(
                {
                    "dataset": dataset,
                    "reference_method": reference_method,
                    "method": method,
                    "metric": metric,
                    "paired_seed_count": n,
                    "mean_difference_candidate_minus_reference": float(np.mean(differences)),
                    "median_difference_candidate_minus_reference": float(np.median(differences)),
                    "bootstrap_ci95_low": float(ci_low),
                    "bootstrap_ci95_high": float(ci_high),
                    "candidate_win_rate": float(np.mean(differences < 0.0)),
                    "candidate_tie_rate": float(np.mean(np.isclose(differences, 0.0))),
                    "wilcoxon_p_value": p_value,
                }
            )

    return pd.DataFrame(records)


def _format_number(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def _format_mean_std(mean: object, std: object, digits: int = 3, bold: bool = False) -> str:
    mean_text = _format_number(mean, digits)
    std_text = _format_number(std, digits)
    if mean_text == "--":
        return "--"
    core = f"{mean_text} $\\pm$ {std_text}"
    return f"\\textbf{{{core}}}" if bold else core


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_table1_latex(
    seedwise: pd.DataFrame,
    aggregate: pd.DataFrame,
    output_dir: Path,
    metric: str,
    horizon: int,
) -> None:
    """Write matched-comparison seed-wise and mean±std LaTeX tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed-wise file: one table per seed, useful for appendix/debugging.
    lines: List[str] = ["% Auto-generated matched pruning comparison: seed-wise results", ""]
    methods = [m for m in TABLE1_METHODS if m in set(seedwise["method"].astype(str))]
    for seed in sorted(seedwise["seed"].dropna().astype(int).unique()):
        subset = seedwise[seedwise["seed"] == seed]
        column_spec = "l" + "c" * len(methods)
        lines.extend(
            [
                "\\begin{table*}[t]",
                "\\centering",
                f"\\caption{{Seed {seed}: matched one-shot pruning comparison at horizon $h={horizon}$. Lower is better.}}",
                "\\small",
                "\\resizebox{\\linewidth}{!}{%",
                f"\\begin{{tabular}}{{{column_spec}}}",
                "\\toprule",
                "Dataset & " + " & ".join(method_label(m) for m in methods) + r" \\",
                "\\midrule",
            ]
        )

        for dataset in sorted(subset["dataset"].astype(str).unique()):
            row = [dataset_label(dataset)]
            dataset_subset = subset[subset["dataset"] == dataset]
            for method in methods:
                match = dataset_subset[dataset_subset["method"] == method]
                row.append(_format_number(match.iloc[0][metric], 4) if not match.empty else "--")
            lines.append(" & ".join(row) + r" \\")

        lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])

    _write(output_dir / "table1_matched_comparison_seedwise.tex", "\n".join(lines))

    # Aggregate file: datasets as rows and methods as columns.
    lines = ["% Auto-generated matched pruning comparison: mean and standard deviation", ""]
    column_spec = "l" + "c" * len(methods)
    lines.extend(
        [
            "\\begin{table*}[t]",
            "\\centering",
            f"\\caption{{Matched one-shot reservoir pruning at horizon $h={horizon}$. Values are mean $\\pm$ standard deviation over paired seeds; lower is better.}}",
            "\\label{tab:matched_pruning}",
            "\\small",
            "\\resizebox{\\linewidth}{!}{%",
            f"\\begin{{tabular}}{{{column_spec}}}",
            "\\toprule",
            "Dataset & " + " & ".join(method_label(m) for m in methods) + r" \\",
            "\\midrule",
        ]
    )

    for dataset in sorted(aggregate["dataset"].astype(str).unique()):
        dataset_subset = aggregate[aggregate["dataset"] == dataset]
        means = {
            method: float(dataset_subset[dataset_subset["method"] == method][f"{metric}_mean"].iloc[0])
            for method in methods
            if not dataset_subset[dataset_subset["method"] == method].empty
        }
        best_method = min(means, key=means.get) if means else None
        row = [dataset_label(dataset)]
        for method in methods:
            match = dataset_subset[dataset_subset["method"] == method]
            if match.empty:
                row.append("--")
            else:
                row.append(
                    _format_mean_std(
                        match.iloc[0][f"{metric}_mean"],
                        match.iloc[0][f"{metric}_std"],
                        digits=4,
                        bold=(method == best_method),
                    )
                )
        lines.append(" & ".join(row) + r" \\")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    _write(output_dir / "table1_matched_comparison_mean_std.tex", "\n".join(lines))


def build_table2_latex(
    seedwise: pd.DataFrame,
    aggregate: pd.DataFrame,
    output_dir: Path,
    horizon: int,
) -> None:
    """Write DMP ablation and inference-efficiency LaTeX tables."""
    methods = [m for m in TABLE2_METHODS if m in set(seedwise["method"].astype(str))]

    lines: List[str] = ["% Auto-generated DMP ablation and efficiency: seed-wise", ""]
    for seed in sorted(seedwise["seed"].dropna().astype(int).unique()):
        subset = seedwise[seedwise["seed"] == seed]
        lines.extend(
            [
                "\\begin{table*}[t]",
                "\\centering",
                f"\\caption{{Seed {seed}: DMP ablation and final-model efficiency at horizon $h={horizon}$.}}",
                "\\small",
                "\\resizebox{\\linewidth}{!}{%",
                "\\begin{tabular}{llrrrrrrr}",
                "\\toprule",
                r"Dataset & Variant & NRMSE@h & Final $N$ & Recurrent params & Model MB & Latency ($\mu$s/step) & Speedup & Offline pruning (s) \\",
                "\\midrule",
            ]
        )
        for dataset in sorted(subset["dataset"].astype(str).unique()):
            ds = subset[subset["dataset"] == dataset]
            first = True
            for method in methods:
                match = ds[ds["method"] == method]
                if match.empty:
                    continue
                row = match.iloc[0]
                lines.append(
                    " & ".join(
                        [
                            dataset_label(dataset) if first else "",
                            method_label(method),
                            _format_number(row["nrmse_at_h"], 4),
                            _format_number(row["final_reservoir_size"], 0),
                            _format_number(row["recurrent_parameters"], 0),
                            _format_number(row["model_memory_mb"], 3),
                            _format_number(row["latency_us_per_step"], 2),
                            _format_number(row["speedup_vs_full"], 2) + "$\\times$",
                            _format_number(row["offline_pruning_s"], 3),
                        ]
                    )
                    + r" \\"
                )
                first = False
            lines.append("\\midrule")
        if lines[-1] == "\\midrule":
            lines[-1] = "\\bottomrule"
        else:
            lines.append("\\bottomrule")
        lines.extend(["\\end{tabular}%", "}", "\\end{table*}", ""])

    _write(output_dir / "table2_ablation_efficiency_seedwise.tex", "\n".join(lines))

    lines = [
        "% Auto-generated DMP ablation and efficiency: mean and standard deviation",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{DMP ablation and final-model efficiency at horizon $h={horizon}$. Values are mean $\\pm$ standard deviation over seeds.}}",
        "\\label{tab:dmp_ablation_efficiency}",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        r"Dataset & Variant & NRMSE@h & Final $N$ & Recurrent params & Model MB & Latency ($\mu$s/step) & Speedup & Offline pruning (s) \\",
        "\\midrule",
    ]

    for dataset in sorted(aggregate["dataset"].astype(str).unique()):
        ds = aggregate[aggregate["dataset"] == dataset]
        first = True
        for method in methods:
            match = ds[ds["method"] == method]
            if match.empty:
                continue
            row = match.iloc[0]
            lines.append(
                " & ".join(
                    [
                        dataset_label(dataset) if first else "",
                        method_label(method),
                        _format_mean_std(row["nrmse_at_h_mean"], row["nrmse_at_h_std"], 4),
                        _format_mean_std(row["final_reservoir_size_mean"], row["final_reservoir_size_std"], 0),
                        _format_mean_std(row["recurrent_parameters_mean"], row["recurrent_parameters_std"], 0),
                        _format_mean_std(row["model_memory_mb_mean"], row["model_memory_mb_std"], 3),
                        _format_mean_std(row["latency_us_per_step_mean"], row["latency_us_per_step_std"], 2),
                        _format_mean_std(row["speedup_vs_full_mean"], row["speedup_vs_full_std"], 2) + "$\\times$",
                        _format_mean_std(row["offline_pruning_s_mean"], row["offline_pruning_s_std"], 3),
                    ]
                )
                + r" \\"
            )
            first = False
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines.extend(["\\end{tabular}%", "}", "\\end{table*}", ""])
    _write(output_dir / "table2_ablation_efficiency_mean_std.tex", "\n".join(lines))


def build_table3_latex(
    seedwise: pd.DataFrame,
    aggregate: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write autonomous dynamical-fidelity seed-wise and mean±std tables."""
    methods = [m for m in TABLE3_METHODS if m in set(seedwise["method"].astype(str))]

    lines: List[str] = ["% Auto-generated autonomous dynamical evaluation: seed-wise", ""]
    for seed in sorted(seedwise["seed"].dropna().astype(int).unique()):
        subset = seedwise[seedwise["seed"] == seed]
        lines.extend(
            [
                "\\begin{table*}[t]",
                "\\centering",
                f"\\caption{{Seed {seed}: autonomous dynamical-fidelity evaluation. Lower is better except valid prediction steps.}}",
                "\\small",
                "\\begin{tabular}{llrrrrr}",
                "\\toprule",
                r"Dataset & Method & Rollout NRMSE & Valid steps $\uparrow$ & Valid fraction $\uparrow$ & PSD distance & ACF distance \\",
                "\\midrule",
            ]
        )
        for dataset in sorted(subset["dataset"].astype(str).unique()):
            ds = subset[subset["dataset"] == dataset]
            first = True
            for method in methods:
                match = ds[ds["method"] == method]
                if match.empty:
                    continue
                row = match.iloc[0]
                lines.append(
                    " & ".join(
                        [
                            dataset_label(dataset) if first else "",
                            method_label(method),
                            _format_number(row["rollout_nrmse"], 4),
                            _format_number(row["valid_prediction_steps"], 0),
                            _format_number(row["valid_prediction_fraction"], 3),
                            _format_number(row["psd_distance"], 4),
                            _format_number(row["autocorrelation_distance"], 4),
                        ]
                    )
                    + r" \\"
                )
                first = False
            lines.append("\\midrule")
        if lines[-1] == "\\midrule":
            lines[-1] = "\\bottomrule"
        lines.extend(["\\end{tabular}", "\\end{table*}", ""])

    _write(output_dir / "table3_dynamical_seedwise.tex", "\n".join(lines))

    lines = [
        "% Auto-generated autonomous dynamical evaluation: mean and standard deviation",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Autonomous dynamical-fidelity evaluation. Values are mean $\\pm$ standard deviation over seeds. Lower is better except valid prediction steps.}",
        "\\label{tab:dynamical_fidelity}",
        "\\small",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        r"Dataset & Method & Rollout NRMSE & Valid steps $\uparrow$ & Valid fraction $\uparrow$ & PSD distance & ACF distance \\",
        "\\midrule",
    ]

    for dataset in sorted(aggregate["dataset"].astype(str).unique()):
        ds = aggregate[aggregate["dataset"] == dataset]
        first = True
        for method in methods:
            match = ds[ds["method"] == method]
            if match.empty:
                continue
            row = match.iloc[0]
            lines.append(
                " & ".join(
                    [
                        dataset_label(dataset) if first else "",
                        method_label(method),
                        _format_mean_std(row["rollout_nrmse_mean"], row["rollout_nrmse_std"], 4),
                        _format_mean_std(row["valid_prediction_steps_mean"], row["valid_prediction_steps_std"], 1),
                        _format_mean_std(row["valid_prediction_fraction_mean"], row["valid_prediction_fraction_std"], 3),
                        _format_mean_std(row["psd_distance_mean"], row["psd_distance_std"], 4),
                        _format_mean_std(
                            row["autocorrelation_distance_mean"],
                            row["autocorrelation_distance_std"],
                            4,
                        ),
                    ]
                )
                + r" \\"
            )
            first = False
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines.extend(["\\end{tabular}", "\\end{table*}", ""])
    _write(output_dir / "table3_dynamical_mean_std.tex", "\n".join(lines))


def save_all_tables_and_statistics(
    all_metrics: pd.DataFrame,
    output_root: Path,
    primary_metric: str,
    horizon: int,
) -> Dict[str, pd.DataFrame]:
    """Create the three requested CSV and LaTeX table families."""

    csv_dir = output_root / "csv"
    latex_dir = output_root / "latex"
    csv_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    all_metrics.to_csv(csv_dir / "all_seedwise_metrics.csv", index=False, float_format="%.10g")

    table1 = all_metrics[
        (all_metrics["experiment"] == "direct")
        & (all_metrics["method"].isin(TABLE1_METHODS))
    ].copy()
    table1["method_order"] = table1["method"].map({m: i for i, m in enumerate(TABLE1_METHODS)})
    table1 = table1.sort_values(["dataset", "seed", "method_order"])

    table1_metrics = [
        "mse_at_h",
        "avg_mse_1_to_h",
        "nrmse_at_h",
        "avg_nrmse_1_to_h",
        "latency_us_per_step",
        "model_memory_mb",
    ]
    table1_agg = aggregate_metrics(table1, ["dataset", "method", "method_order"], table1_metrics)
    table1.to_csv(csv_dir / "table1_matched_comparison_seedwise.csv", index=False, float_format="%.10g")
    table1_agg.to_csv(csv_dir / "table1_matched_comparison_mean_std.csv", index=False, float_format="%.10g")
    build_table1_latex(table1, table1_agg, latex_dir, metric=primary_metric, horizon=horizon)

    stats = paired_statistics(table1, reference_method="full_esn", metric=primary_metric)
    stats.to_csv(csv_dir / "table1_paired_statistics.csv", index=False, float_format="%.10g")

    table2 = all_metrics[
        (all_metrics["experiment"] == "direct")
        & (all_metrics["method"].isin(TABLE2_METHODS))
    ].copy()
    table2["method_order"] = table2["method"].map({m: i for i, m in enumerate(TABLE2_METHODS)})
    table2 = table2.sort_values(["dataset", "seed", "method_order"])
    table2_metrics = [
        "nrmse_at_h",
        "final_reservoir_size",
        "recurrent_parameters",
        "recurrent_nonzeros",
        "total_parameters",
        "model_memory_mb",
        "latency_us_per_step",
        "latency_iqr_us_per_step",
        "speedup_vs_full",
        "offline_pruning_s",
        "readout_training_s",
        "total_model_build_s",
    ]
    table2_agg = aggregate_metrics(table2, ["dataset", "method", "method_order"], table2_metrics)
    table2.to_csv(csv_dir / "table2_ablation_efficiency_seedwise.csv", index=False, float_format="%.10g")
    table2_agg.to_csv(csv_dir / "table2_ablation_efficiency_mean_std.csv", index=False, float_format="%.10g")
    build_table2_latex(table2, table2_agg, latex_dir, horizon=horizon)

    table3 = all_metrics[
        (all_metrics["experiment"] == "dynamical")
        & (all_metrics["method"].isin(TABLE3_METHODS))
    ].copy()
    table3["method_order"] = table3["method"].map({m: i for i, m in enumerate(TABLE3_METHODS)})
    table3 = table3.sort_values(["dataset", "seed", "method_order"])
    table3_metrics = [
        "one_step_nrmse",
        "rollout_nrmse",
        "valid_prediction_steps",
        "valid_prediction_fraction",
        "psd_distance",
        "autocorrelation_distance",
    ]
    table3_agg = aggregate_metrics(table3, ["dataset", "method", "method_order"], table3_metrics)
    table3.to_csv(csv_dir / "table3_dynamical_seedwise.csv", index=False, float_format="%.10g")
    table3_agg.to_csv(csv_dir / "table3_dynamical_mean_std.csv", index=False, float_format="%.10g")
    if not table3.empty:
        build_table3_latex(table3, table3_agg, latex_dir)

    return {
        "table1_seedwise": table1,
        "table1_aggregate": table1_agg,
        "table1_statistics": stats,
        "table2_seedwise": table2,
        "table2_aggregate": table2_agg,
        "table3_seedwise": table3,
        "table3_aggregate": table3_agg,
    }
