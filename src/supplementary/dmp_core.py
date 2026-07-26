"""Core Echo State Network and one-shot DMP utilities.

The implementation is intentionally compact and paper-oriented:

* every dataset is treated as a single 1-D time series;
* normalization uses the washout + training region only;
* training targets never cross into the test region;
* DMP is strictly one-shot pruning;
* every pruning baseline starts from the same full reservoir for a seed;
* only the linear readout is retrained after pruning.

The Jacobian Gramian is computed with an exact algebraic simplification.  For

    J_t = (1-a)I + a diag(d_t) W,

we use

    E[J_t^T J_t]
      = c^2 I + ac(E[D]W + W^T E[D]) + a^2 W^T E[D^2] W,

where c = 1-a.  This avoids explicitly building a dense Jacobian for every
training time step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """Fractions used to divide a series into washout, train, and test blocks."""

    washout_ratio: float = 0.10
    train_ratio: float = 0.70
    test_ratio: float = 0.20

    def validate(self) -> None:
        total = self.washout_ratio + self.train_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-8:
            raise ValueError("washout_ratio + train_ratio + test_ratio must equal 1.0")
        if min(self.washout_ratio, self.train_ratio, self.test_ratio) <= 0.0:
            raise ValueError("All split ratios must be positive")


@dataclass(frozen=True)
class ESNHyperParams:
    """Shared reservoir and readout hyperparameters."""

    leak_rate: float = 0.4
    spectral_radius: float = 0.95
    input_scaling: float = 0.5
    bias_scaling: float = 0.2
    zero_fraction: float = 0.10
    ridge_alpha: float = 1e-6
    spectral_radius_iterations: int = 80

    def validate(self) -> None:
        if not 0.0 < self.leak_rate <= 1.0:
            raise ValueError("leak_rate must be in (0, 1]")
        if self.spectral_radius <= 0.0:
            raise ValueError("spectral_radius must be positive")
        if not 0.0 <= self.zero_fraction < 1.0:
            raise ValueError("zero_fraction must be in [0, 1)")
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative")


@dataclass
class PreparedSeries:
    """A normalized 1-D series and the fixed experiment boundaries."""

    name: str
    path: Path
    raw: torch.Tensor
    normalized: torch.Tensor
    data_min: float
    data_max: float
    washout_end: int
    train_end: int
    test_end: int

    @property
    def train_scale(self) -> float:
        return max(self.data_max - self.data_min, 1e-12)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Map values from [-1, 1] back to the original data scale."""
        if self.data_max <= self.data_min:
            return values.clone()
        return (values + 1.0) * 0.5 * (self.data_max - self.data_min) + self.data_min


@dataclass
class TrainingBundle:
    """Training signals reused by DMP and matched pruning baselines."""

    states: torch.Tensor
    derivatives: torch.Tensor
    preactivations: torch.Tensor
    design: torch.Tensor
    targets: torch.Tensor


@dataclass
class DirectEvaluation:
    """Multi-horizon direct forecasting outputs and metrics."""

    targets: torch.Tensor
    predictions: torch.Tensor
    mse_at_h: float
    avg_mse_1_to_h: float
    nrmse_at_h: float
    avg_nrmse_1_to_h: float


@dataclass
class GramianDiagnostics:
    """DMP scores and eigenspectrum information for one full reservoir."""

    gramian: torch.Tensor
    eigenvalues: torch.Tensor
    normalized_eigenvalues: torch.Tensor
    cumulative_energy: torch.Tensor
    selected_rank: int
    dmp_scores: torch.Tensor
    jacobian_column_energy_scores: torch.Tensor


# ---------------------------------------------------------------------------
# Dataset loading and leakage-safe normalization
# ---------------------------------------------------------------------------


def _first_numeric_column(frame: pd.DataFrame) -> pd.Series:
    """Return the first column that can be interpreted as numeric."""
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() >= max(10, int(0.8 * len(frame))):
            return numeric
    raise ValueError("No usable numeric column was found in the CSV file")


def load_prepared_series(
    csv_path: str | Path,
    name: Optional[str],
    split: SplitConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> PreparedSeries:
    """Load one 1-D CSV and normalize using training-region statistics only.

    The normalization range is estimated from indices ``[0, train_end)``.
    Test values therefore never influence preprocessing.
    """

    split.validate()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    series = _first_numeric_column(frame)
    values = series.to_numpy(dtype=np.float64)

    # Linear interpolation is preferable to replacing isolated missing values
    # with zero, while the final fill handles missing values at either edge.
    values = (
        pd.Series(values)
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    if values.ndim != 1 or len(values) < 100:
        raise ValueError(f"{path} must contain a 1-D series with at least 100 samples")

    n = len(values)
    washout_end = max(1, int(round(split.washout_ratio * n)))
    train_count = max(2, int(round(split.train_ratio * n)))
    train_end = min(n - 2, washout_end + train_count)
    test_end = n

    train_reference = values[:train_end]
    data_min = float(np.min(train_reference))
    data_max = float(np.max(train_reference))

    if data_max > data_min:
        normalized = (values - data_min) / (data_max - data_min) * 2.0 - 1.0
    else:
        normalized = values.copy()

    raw_tensor = torch.as_tensor(values, dtype=dtype, device=device)
    norm_tensor = torch.as_tensor(normalized, dtype=dtype, device=device)

    return PreparedSeries(
        name=name or path.stem,
        path=path,
        raw=raw_tensor,
        normalized=norm_tensor,
        data_min=data_min,
        data_max=data_max,
        washout_end=washout_end,
        train_end=train_end,
        test_end=test_end,
    )


# ---------------------------------------------------------------------------
# Reservoir model
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set NumPy and Torch seeds for one deterministic experiment run."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@torch.no_grad()
def spectral_radius_estimate(W: torch.Tensor, iterations: int = 80) -> float:
    """Estimate the dominant eigenvalue magnitude with power iteration.

    The same deterministic estimator is used for every model, so matched
    comparisons are not biased by different stabilization procedures.
    """

    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.numel() == 0:
        return 0.0

    # A fixed initial vector avoids consuming the experiment RNG state.
    v = torch.linspace(0.5, 1.5, W.shape[0], device=W.device, dtype=W.dtype)
    v = v / (torch.linalg.vector_norm(v) + 1e-12)

    previous = torch.tensor(0.0, device=W.device, dtype=W.dtype)
    estimate = torch.tensor(0.0, device=W.device, dtype=W.dtype)

    for _ in range(max(5, int(iterations))):
        wv = W @ v
        norm = torch.linalg.vector_norm(wv)
        if norm <= 1e-12:
            return 0.0
        v = wv / norm
        estimate = torch.linalg.vector_norm(W @ v)
        if torch.abs(estimate - previous) <= 1e-7 * torch.clamp(estimate, min=1.0):
            break
        previous = estimate

    return float(estimate.item())


@torch.no_grad()
def rescale_to_target_radius(
    W: torch.Tensor,
    target_radius: float,
    iterations: int = 80,
) -> torch.Tensor:
    """Return a copy of ``W`` rescaled to the requested spectral radius."""

    current = spectral_radius_estimate(W, iterations=iterations)
    if current <= 1e-12:
        return W.clone()
    return W * (float(target_radius) / current)


class ESNModel:
    """Leaky tanh ESN with a direct multi-horizon linear readout."""

    def __init__(
        self,
        W: torch.Tensor,
        Win: torch.Tensor,
        hyper: ESNHyperParams,
        Wout: Optional[torch.Tensor] = None,
    ) -> None:
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be square")
        if Win.shape != (W.shape[0], 2):
            raise ValueError("Win must have shape (reservoir_size, 2)")

        self.W = W
        self.Win = Win
        self.hyper = hyper
        self.Wout = Wout

    @property
    def reservoir_size(self) -> int:
        return int(self.W.shape[0])

    @property
    def device(self) -> torch.device:
        return self.W.device

    @property
    def dtype(self) -> torch.dtype:
        return self.W.dtype

    @torch.no_grad()
    def zero_state(self) -> torch.Tensor:
        return torch.zeros((self.reservoir_size, 1), device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def step(self, u: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one reservoir step and return state, preactivation, derivative."""
        u_scalar = u.reshape(1, 1).to(device=self.device, dtype=self.dtype)
        feature = torch.vstack(
            (
                torch.ones((1, 1), device=self.device, dtype=self.dtype),
                u_scalar,
            )
        )
        preactivation = self.Win @ feature + self.W @ x
        tanh_value = torch.tanh(preactivation)
        x_new = (1.0 - self.hyper.leak_rate) * x + self.hyper.leak_rate * tanh_value
        derivative = 1.0 - tanh_value.square()
        return x_new, preactivation, derivative

    @torch.no_grad()
    def readout_features(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.vstack(
            (
                torch.ones((1, 1), device=self.device, dtype=self.dtype),
                u.reshape(1, 1).to(device=self.device, dtype=self.dtype),
                x,
            )
        )

    @torch.no_grad()
    def predict_from_state(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.Wout is None:
            raise ValueError("Readout is not trained")
        return self.Wout @ self.readout_features(u, x)

    @torch.no_grad()
    def collect_training_bundle(
        self,
        data: torch.Tensor,
        washout_end: int,
        train_end: int,
        horizon: int,
    ) -> TrainingBundle:
        """Collect training features while keeping all targets before train_end."""

        if horizon <= 0:
            raise ValueError("horizon must be positive")

        final_input_index = train_end - horizon - 1
        if final_input_index < washout_end:
            raise ValueError("Training block is too short for the selected horizon")

        sample_count = final_input_index - washout_end + 1
        design = torch.zeros(
            (2 + self.reservoir_size, sample_count),
            device=self.device,
            dtype=self.dtype,
        )
        targets = torch.zeros(
            (horizon, sample_count),
            device=self.device,
            dtype=self.dtype,
        )
        states = torch.zeros(
            (sample_count, self.reservoir_size),
            device=self.device,
            dtype=self.dtype,
        )
        derivatives = torch.zeros_like(states)
        preactivations = torch.zeros_like(states)

        x = self.zero_state()

        # Washout uses only historical observed inputs.
        for t in range(washout_end):
            x, _, _ = self.step(data[t], x)

        for column, t in enumerate(range(washout_end, final_input_index + 1)):
            x, pre, derivative = self.step(data[t], x)
            design[:, column] = self.readout_features(data[t], x)[:, 0]
            states[column] = x[:, 0]
            derivatives[column] = derivative[:, 0]
            preactivations[column] = pre[:, 0]

            for step in range(1, horizon + 1):
                targets[step - 1, column] = data[t + step]

        return TrainingBundle(
            states=states,
            derivatives=derivatives,
            preactivations=preactivations,
            design=design,
            targets=targets,
        )

    def fit_readout(self, bundle: TrainingBundle) -> None:
        """Fit the multi-horizon readout by numerically guarded ridge regression."""

        X = bundle.design
        Y = bundle.targets
        gram = X @ X.T
        rhs = (Y @ X.T).T
        eye = torch.eye(gram.shape[0], device=self.device, dtype=self.dtype)

        # Scale the fallback jitter to the feature energy.  The requested ridge
        # coefficient is still tried first and is what is normally used.
        trace_scale = float(torch.trace(gram).abs().item()) / max(1, gram.shape[0])
        trace_scale = max(trace_scale, 1.0)

        for jitter_multiplier in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
            regularization = self.hyper.ridge_alpha + jitter_multiplier * trace_scale
            try:
                self.Wout = torch.linalg.solve(
                    (gram + regularization * eye).T,
                    rhs,
                ).T
                return
            except RuntimeError:
                self.Wout = None

        # Last-resort CPU solve for rare singular float32 cases.
        gram_cpu = gram.detach().cpu().double()
        rhs_cpu = rhs.detach().cpu().double()
        eye_cpu = torch.eye(gram_cpu.shape[0], dtype=torch.float64)
        regularization = self.hyper.ridge_alpha + 1e-3 * trace_scale
        solution = torch.linalg.lstsq(
            (gram_cpu + regularization * eye_cpu).T,
            rhs_cpu,
        ).solution.T
        self.Wout = solution.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def evaluate_direct(
        self,
        data: torch.Tensor,
        test_start: int,
        horizon: int,
    ) -> DirectEvaluation:
        """Evaluate direct forecasts using observed current-time test inputs."""

        if self.Wout is None:
            raise ValueError("Readout is not trained")

        final_input_index = len(data) - horizon - 1
        if final_input_index < test_start:
            raise ValueError("Test block is too short for the selected horizon")

        sample_count = final_input_index - test_start + 1
        predictions = torch.zeros(
            (horizon, sample_count),
            device=self.device,
            dtype=self.dtype,
        )
        targets = torch.zeros_like(predictions)

        x = self.zero_state()
        for t in range(test_start):
            x, _, _ = self.step(data[t], x)

        for column, t in enumerate(range(test_start, final_input_index + 1)):
            x, _, _ = self.step(data[t], x)
            predictions[:, column] = self.predict_from_state(data[t], x)[:, 0]
            for step in range(1, horizon + 1):
                targets[step - 1, column] = data[t + step]

        errors = predictions - targets
        mse_per_horizon = torch.mean(errors.square(), dim=1)
        mse_at_h = float(mse_per_horizon[-1].item())
        avg_mse = float(torch.mean(errors.square()).item())

        target_std_at_h = float(torch.std(targets[-1], unbiased=False).item())
        target_std_all = float(torch.std(targets, unbiased=False).item())
        nrmse_at_h = float(np.sqrt(mse_at_h) / max(target_std_at_h, 1e-12))
        avg_nrmse = float(np.sqrt(avg_mse) / max(target_std_all, 1e-12))

        return DirectEvaluation(
            targets=targets,
            predictions=predictions,
            mse_at_h=mse_at_h,
            avg_mse_1_to_h=avg_mse,
            nrmse_at_h=nrmse_at_h,
            avg_nrmse_1_to_h=avg_nrmse,
        )


def create_reservoir(
    reservoir_size: int,
    seed: int,
    hyper: ESNHyperParams,
    device: torch.device,
    dtype: torch.dtype,
) -> ESNModel:
    """Create a reproducible random reservoir and scale it once."""

    hyper.validate()
    if reservoir_size <= 0:
        raise ValueError("reservoir_size must be positive")

    set_seed(seed)
    W = torch.rand((reservoir_size, reservoir_size), device=device, dtype=dtype) - 0.5
    if hyper.zero_fraction > 0.0:
        zero_mask = torch.rand((reservoir_size, reservoir_size), device=device) < hyper.zero_fraction
        W[zero_mask] = 0.0

    W = rescale_to_target_radius(
        W,
        target_radius=hyper.spectral_radius,
        iterations=hyper.spectral_radius_iterations,
    )

    Win = torch.rand((reservoir_size, 2), device=device, dtype=dtype) - 0.5
    Win[:, 0] *= hyper.bias_scaling
    Win[:, 1] *= hyper.input_scaling

    return ESNModel(W=W, Win=Win, hyper=hyper)


def clone_pruned_model(
    full_model: ESNModel,
    keep_indices: torch.Tensor,
    rescale: bool,
) -> ESNModel:
    """Restrict a full reservoir to selected neurons and optionally rescale W."""

    indices = keep_indices.to(device=full_model.device, dtype=torch.long)
    W = full_model.W.index_select(0, indices).index_select(1, indices).clone()
    Win = full_model.Win.index_select(0, indices).clone()

    if rescale:
        W = rescale_to_target_radius(
            W,
            target_radius=full_model.hyper.spectral_radius,
            iterations=full_model.hyper.spectral_radius_iterations,
        )

    return ESNModel(W=W, Win=Win, hyper=full_model.hyper)


# ---------------------------------------------------------------------------
# DMP and matched pruning scores
# ---------------------------------------------------------------------------


@torch.no_grad()
def expected_jacobian_gramian(
    W: torch.Tensor,
    derivatives: torch.Tensor,
    leak_rate: float,
) -> torch.Tensor:
    """Compute the exact trajectory-average of J_t^T J_t efficiently."""

    if derivatives.ndim != 2 or derivatives.shape[1] != W.shape[0]:
        raise ValueError("derivatives must have shape (time, reservoir_size)")

    compute_dtype = torch.float64
    W64 = W.to(dtype=compute_dtype)
    D64 = derivatives.to(dtype=compute_dtype)

    mean_d = torch.mean(D64, dim=0)
    mean_d2 = torch.mean(D64.square(), dim=0)
    a = float(leak_rate)
    c = 1.0 - a
    n = W.shape[0]

    # diag(mean_d) @ W is row-wise scaling; W^T @ diag(mean_d) is
    # column-wise scaling after transposition.
    mean_D_W = mean_d[:, None] * W64
    Wt_mean_D = W64.T * mean_d[None, :]
    quadratic = W64.T @ (mean_d2[:, None] * W64)

    G = (
        (c * c) * torch.eye(n, device=W.device, dtype=compute_dtype)
        + (a * c) * (mean_D_W + Wt_mean_D)
        + (a * a) * quadratic
    )

    # Remove tiny numerical asymmetry before eigh.
    return 0.5 * (G + G.T)


@torch.no_grad()
def compute_gramian_diagnostics(
    W: torch.Tensor,
    derivatives: torch.Tensor,
    leak_rate: float,
    energy_tau: float,
) -> GramianDiagnostics:
    """Compute DMP scores and the full-spectrum Jacobian column-energy baseline."""

    if not 0.0 < energy_tau <= 1.0:
        raise ValueError("energy_tau must be in (0, 1]")

    G = expected_jacobian_gramian(W, derivatives, leak_rate)
    eigenvalues, eigenvectors = torch.linalg.eigh(G)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvectors = torch.flip(eigenvectors, dims=[1])

    total = torch.sum(eigenvalues)
    if total <= 1e-14:
        normalized = torch.zeros_like(eigenvalues)
        cumulative = torch.zeros_like(eigenvalues)
        selected_rank = 0
        dmp_scores = torch.zeros(W.shape[0], device=W.device, dtype=torch.float64)
    else:
        normalized = eigenvalues / total
        cumulative = torch.cumsum(normalized, dim=0)
        selected = torch.where(cumulative >= energy_tau)[0]
        selected_rank = int(selected[0].item() + 1) if len(selected) else int(W.shape[0])

        V_r = eigenvectors[:, :selected_rank]
        L_r = normalized[:selected_rank]
        # Squared modal participation equals the diagonal of the retained,
        # normalized low-rank Gramian approximation.
        dmp_scores = torch.sum(V_r.square() * L_r[None, :], dim=1)

    jacobian_column_energy = torch.diagonal(G).clone()

    return GramianDiagnostics(
        gramian=G,
        eigenvalues=eigenvalues,
        normalized_eigenvalues=normalized,
        cumulative_energy=cumulative,
        selected_rank=selected_rank,
        dmp_scores=dmp_scores,
        jacobian_column_energy_scores=jacobian_column_energy,
    )


@torch.no_grad()
def magnitude_scores(W: torch.Tensor) -> torch.Tensor:
    """Node magnitude score using incoming and outgoing recurrent strength."""
    return torch.linalg.vector_norm(W, dim=0) + torch.linalg.vector_norm(W, dim=1)


@torch.no_grad()
def activation_variance_scores(states: torch.Tensor) -> torch.Tensor:
    """Node activity score from training-trajectory state variance."""
    return torch.var(states, dim=0, unbiased=False)


@torch.no_grad()
def select_top_nodes(scores: torch.Tensor, target_size: int) -> torch.Tensor:
    """Return sorted indices of the highest-scoring nodes."""
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 1 <= target_size <= len(scores):
        raise ValueError("target_size is outside the valid range")
    indices = torch.topk(scores, k=target_size, largest=True, sorted=False).indices
    return torch.sort(indices).values


@torch.no_grad()
def random_keep_indices(
    reservoir_size: int,
    target_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate one deterministic random node subset for paired ablations."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(reservoir_size, generator=generator)[:target_size]
    return torch.sort(permutation).values.to(device=device)


def approximate_betweenness_scores(
    W: torch.Tensor,
    sample_sources: int,
    strongest_edges_per_node: int,
    seed: int,
) -> torch.Tensor:
    """Compute a practical weighted betweenness-centrality baseline.

    A nearly dense 1000-node reservoir can make exact NetworkX betweenness
    prohibitively slow.  The graph therefore retains the strongest outgoing
    edges per source node and optionally samples source nodes.  Both settings
    are written to every metrics file and should be disclosed in the paper.
    """

    import networkx as nx

    W_cpu = W.detach().cpu().double()
    n = W_cpu.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))

    edge_limit = min(max(1, strongest_edges_per_node), n)
    for source in range(n):
        # W[target, source] is the influence source -> target.
        strengths = torch.abs(W_cpu[:, source])
        top = torch.topk(strengths, k=edge_limit, largest=True).indices.tolist()
        for target in top:
            strength = float(strengths[target].item())
            if strength <= 1e-12 or target == source:
                continue
            graph.add_edge(
                source,
                int(target),
                weight=strength,
                distance=1.0 / (strength + 1e-12),
            )

    k = None if sample_sources <= 0 or sample_sources >= n else int(sample_sources)
    scores = nx.betweenness_centrality(
        graph,
        k=k,
        normalized=True,
        weight="distance",
        seed=int(seed),
    )
    values = np.asarray([scores.get(i, 0.0) for i in range(n)], dtype=np.float64)
    return torch.as_tensor(values, device=W.device, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Model-size accounting
# ---------------------------------------------------------------------------


def tensor_bytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def model_size_metrics(model: ESNModel) -> Dict[str, float]:
    """Return parameter counts and exact in-memory tensor size."""
    recurrent_parameters = int(model.W.numel())
    recurrent_nonzeros = int(torch.count_nonzero(model.W).item())
    total_parameters = recurrent_parameters + int(model.Win.numel())
    if model.Wout is not None:
        total_parameters += int(model.Wout.numel())

    total_bytes = tensor_bytes(model.W) + tensor_bytes(model.Win) + tensor_bytes(model.Wout)
    return {
        "final_reservoir_size": int(model.reservoir_size),
        "recurrent_parameters": recurrent_parameters,
        "recurrent_nonzeros": recurrent_nonzeros,
        "total_parameters": total_parameters,
        "model_memory_mb": float(total_bytes / (1024.0 * 1024.0)),
    }
