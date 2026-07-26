"""Evaluation, timing, and long autonomous rollout utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from .dmp_core import ESNModel


@dataclass
class LatencyResult:
    median_us_per_step: float
    iqr_us_per_step: float


@dataclass
class RolloutResult:
    target: np.ndarray
    prediction: np.ndarray
    one_step_nrmse: float
    rollout_nrmse: float
    valid_prediction_steps: int
    valid_prediction_fraction: float
    psd_distance: float
    autocorrelation_distance: float


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_inference_latency(
    model: ESNModel,
    data: torch.Tensor,
    start_index: int,
    benchmark_steps: int = 300,
    warmup_repeats: int = 3,
    measured_repeats: int = 20,
) -> LatencyResult:
    """Measure median recurrent update + readout latency per time step.

    The state is first driven to ``start_index`` using observed history.  Every
    repeat then processes the same fixed input segment from the same starting
    state.  CUDA synchronization is included around the timed section.
    """

    if model.Wout is None:
        raise ValueError("Readout must be trained before benchmarking")

    usable = max(1, min(int(benchmark_steps), len(data) - start_index))
    inputs = data[start_index : start_index + usable]

    x_start = model.zero_state()
    for t in range(start_index):
        x_start, _, _ = model.step(data[t], x_start)

    def run_once() -> None:
        x = x_start.clone()
        for u in inputs:
            x, _, _ = model.step(u, x)
            _ = model.predict_from_state(u, x)

    for _ in range(max(1, int(warmup_repeats))):
        run_once()
    _synchronize(model.device)

    measurements = []
    for _ in range(max(3, int(measured_repeats))):
        _synchronize(model.device)
        start = time.perf_counter()
        run_once()
        _synchronize(model.device)
        elapsed = time.perf_counter() - start
        measurements.append(elapsed * 1e6 / usable)

    values = np.asarray(measurements, dtype=np.float64)
    q25, q75 = np.percentile(values, [25, 75])
    return LatencyResult(
        median_us_per_step=float(np.median(values)),
        iqr_us_per_step=float(q75 - q25),
    )


def _normalized_power_spectrum(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    total = float(np.sum(spectrum))
    if total <= 1e-14:
        return np.zeros_like(spectrum)
    return spectrum / total


def power_spectrum_distance(target: np.ndarray, prediction: np.ndarray) -> float:
    """Total-variation distance between normalized power spectra."""
    p = _normalized_power_spectrum(target)
    q = _normalized_power_spectrum(prediction)
    size = min(len(p), len(q))
    return float(0.5 * np.sum(np.abs(p[:size] - q[:size])))


def _autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x - np.mean(x)
    denominator = float(np.dot(x, x))
    if denominator <= 1e-14:
        return np.zeros(max_lag + 1, dtype=np.float64)

    result = np.empty(max_lag + 1, dtype=np.float64)
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        result[lag] = float(np.dot(x[:-lag], x[lag:]) / denominator)
    return result


def autocorrelation_distance(
    target: np.ndarray,
    prediction: np.ndarray,
    max_lag: int = 100,
) -> float:
    """RMSE between target and predicted autocorrelation functions."""
    size = min(len(target), len(prediction))
    lag = min(int(max_lag), max(1, size // 4))
    target_acf = _autocorrelation(np.asarray(target)[:size], lag)
    pred_acf = _autocorrelation(np.asarray(prediction)[:size], lag)
    return float(np.sqrt(np.mean((target_acf - pred_acf) ** 2)))


@torch.no_grad()
def autonomous_rollout(
    model: ESNModel,
    data: torch.Tensor,
    test_start: int,
    rollout_steps: int,
    valid_threshold_std: float = 0.40,
    autocorrelation_max_lag: int = 100,
) -> RolloutResult:
    """Run a one-step readout in closed loop from the test boundary.

    During rollout, each prediction becomes the next input.  This is a strict
    autonomous test and is therefore most meaningful for Mackey--Glass or other
    self-driven dynamical datasets.
    """

    if model.Wout is None:
        raise ValueError("Readout must be trained before rollout")
    if model.Wout.shape[0] != 1:
        raise ValueError("Autonomous rollout requires a one-step readout")

    available = len(data) - test_start
    steps = min(max(2, int(rollout_steps)), available)
    target = data[test_start : test_start + steps].detach().cpu().numpy().astype(np.float64)

    # Drive through the final observed training input.  The readout attached to
    # that state predicts the first unseen test value.
    x = model.zero_state()
    for t in range(test_start):
        x, _, _ = model.step(data[t], x)

    current_input = data[test_start - 1].reshape(1)
    prediction = np.zeros(steps, dtype=np.float64)

    for step in range(steps):
        predicted = model.predict_from_state(current_input, x)[0, 0]
        prediction[step] = float(predicted.item())
        current_input = predicted.reshape(1)
        x, _, _ = model.step(current_input, x)

    target_std = float(np.std(target))
    target_std = max(target_std, 1e-12)
    errors = prediction - target
    rollout_nrmse = float(np.sqrt(np.mean(errors**2)) / target_std)

    # One-step NRMSE is reported separately because the first autonomous output
    # is still based entirely on observed history.
    one_step_nrmse = float(abs(errors[0]) / target_std)

    normalized_abs_error = np.abs(errors) / target_std
    exceeded = np.where(normalized_abs_error > float(valid_threshold_std))[0]
    valid_steps = int(exceeded[0]) if len(exceeded) else int(steps)

    return RolloutResult(
        target=target,
        prediction=prediction,
        one_step_nrmse=one_step_nrmse,
        rollout_nrmse=rollout_nrmse,
        valid_prediction_steps=valid_steps,
        valid_prediction_fraction=float(valid_steps / steps),
        psd_distance=power_spectrum_distance(target, prediction),
        autocorrelation_distance=autocorrelation_distance(
            target,
            prediction,
            max_lag=autocorrelation_max_lag,
        ),
    )


def cuda_peak_memory_mb(device: torch.device) -> float:
    """Return peak allocated CUDA memory, or zero on CPU."""
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
