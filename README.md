# Dynamical Mode Pruning for Echo State Networks

Research code for **Dynamical Mode Pruning (DMP)**, a structured reservoir-pruning method for Echo State Networks (ESNs). DMP ranks reservoir neurons according to their participation in dominant input-driven transition modes obtained from a trajectory-averaged Jacobian Gramian.

This repository contains the experimental implementation used for the paper:

> **Rethinking Reservoir Pruning: A Dynamical Perspective for Echo State Networks**

The codebase includes:

- broad benchmark experiments across reservoir sizes, forecasting horizons, and pruning ratios;
- Base, Leaky, and Deep ESN reference models;
- betweenness- and closeness-based reservoir pruning;
- controlled one-shot comparisons with fresh smaller reservoirs, random pruning, magnitude pruning, activation-variance pruning, Jacobian column-energy pruning, and betweenness pruning;
- spectral-radius ablations;
- inference-latency, model-size, and parameter-count measurements;
- autonomous Mackey–Glass evaluation;
- plotting, CSV aggregation, and paper-ready LaTeX table generation;
- the compact supplementary LaTeX source and its associated figures.

---

## Method overview

For a leaky ESN, the reservoir state evolves as

$$
\mathbf{x}(t+1)
=
(1-a)\mathbf{x}(t)
+
a\,f\!\left(
W\mathbf{x}(t)
+
W^{\mathrm{in}}\mathbf{u}(t+1)
\right),
$$

where:

- $\mathbf{x}(t)\in\mathbb{R}^{N}$ is the reservoir state;
- $a$ is the leak rate;
- $W\in\mathbb{R}^{N\times N}$ is the recurrent reservoir matrix;
- $W^{\mathrm{in}}$ is the input matrix;
- $f(\cdot)$ is the nonlinear activation function.

The local transition Jacobian is

$$
J(t)
=
(1-a)I
+
aD(t)W,
$$

with

$$
D(t)
=
\operatorname{diag}
\left(
f'\!\left(\mathbf{z}(t)\right)
\right),
$$

where $\mathbf{z}(t)$ is the reservoir pre-activation.

DMP aggregates the transition Jacobians over the driven trajectory to form the trajectory-averaged Jacobian Gramian

$$
\mathcal{G}
=
\frac{1}{T}
\sum_{t=1}^{T}
J(t)^{\top}J(t).
$$

The Gramian eigendecomposition is

$$
\mathcal{G}
=
V\Lambda V^{\top},
$$

where the eigenvalues are ordered as

$$
\lambda_1
\geq
\lambda_2
\geq
\cdots
\geq
\lambda_N
\geq
0.
$$

The normalized modal energy is

$$
\widetilde{\lambda}_k
=
\frac{\lambda_k}
{\sum_{j=1}^{N}\lambda_j}.
$$

DMP retains the smallest number of leading modes $r$ satisfying

$$
\sum_{k=1}^{r}
\widetilde{\lambda}_k
\geq
\tau,
$$

where $\tau\in(0,1]$ is the prescribed energy threshold.

The importance of reservoir neuron $i$ is then defined as

$$
\eta_i
=
\sum_{k=1}^{r}
\widetilde{\lambda}_k
V_{ik}^{2}.
$$

The highest-scoring neurons are retained. The recurrent reservoir is reduced by extracting the corresponding principal submatrix of $W$, and the input matrix is reduced using the same neuron indices. The recurrent weights remain fixed after pruning, and only the linear readout is refitted.

Spectral-radius rescaling is available as an explicit experimental variant. It is not an implicit part of the main DMP configuration.

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── main_train.py
├── run_supplementary.py
│
├── configs/
│   ├── main_experiments.yml
│   └── supplementary_experiments.yml
│
├── data/
│   ├── electricity.csv
│   ├── mackey.csv
│   ├── solar.csv
│   ├── temperature.csv
│   └── wind.csv
│
├── figures/
│   ├── figure2.pdf
│   ├── figure3.pdf
│   ├── figure4.pdf
│   ├── temperature_accuracy_latency.pdf
│   └── temperature_matched_pruning_nrmse.pdf
│
├── src/
│   ├── dmp_model/
│   │   ├── dmp_esn.py
│   │   └── dmp_results_save.py
│   │
│   ├── sota_models/
│   │   ├── base_esn.py
│   │   ├── leaky_esn.py
│   │   ├── deep_esn.py
│   │   ├── betweenness_esn.py
│   │   └── closeness_esn.py
│   │
│   ├── supplementary/
│   │   ├── dmp_core.py
│   │   ├── evaluation.py
│   │   ├── plotting.py
│   │   └── reporting.py
│   │
│   └── utils/
│       ├── data_cache.py
│       └── overleaf_table_formatter.py
│
├── supplementary/
│   └── supplementary.tex
│
├── results/
└── logs/
```

### Main components

| Path | Purpose |
|---|---|
| `main_train.py` | Runs the broad benchmark, scaling, forecasting-horizon, pruning-ratio, and reference-model experiments. |
| `run_supplementary.py` | Runs the controlled one-shot comparison, efficiency analysis, and autonomous dynamical evaluation. |
| `src/dmp_model/` | Main DMP implementation used in the broad benchmark pipeline. |
| `src/sota_models/` | Base, Leaky, Deep, betweenness-pruned, and closeness-pruned ESN implementations. |
| `src/supplementary/` | Controlled experiment utilities, model evaluation, plotting, and table generation. |
| `configs/` | YAML configuration files for the two experiment pipelines. |
| `supplementary/` | Compact LaTeX source for the supplementary material. |
| `figures/` | Paper and supplementary figures. |
| `results/` | Generated experiment outputs. |
| `logs/` | Runtime logs. |

---

## Installation

Python 3.10 or newer is recommended.

### 1. Clone the repository

```bash
git clone https://github.com/Laudarisd/dmp_esn.git
cd dmp_esn
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

CUDA is optional. When CUDA is available and selected in the configuration, PyTorch uses the GPU. Otherwise, the experiments run on CPU.

---

## Data

The included CSV files make the repository directly runnable and preserve the one-dimensional time-series format used by the experiments.

Each dataset loader:

- identifies a usable numeric column;
- interpolates isolated missing values;
- applies a fixed washout, training, and testing split;
- prevents training targets from crossing into the test interval.

For the controlled experiment pipeline, normalization is leakage-safe. The minimum and maximum values are estimated from the washout and training region only, and the series is linearly scaled to $[-1,1]$.

Dataset paths and split ratios are configured in:

```text
configs/main_experiments.yml
configs/supplementary_experiments.yml
```

The repository includes the following datasets:

- Mackey–Glass;
- electricity;
- temperature;
- wind;
- solar.

---

## Running the broad benchmark suite

The broad benchmark pipeline evaluates DMP together with the Base ESN, Leaky ESN, Deep ESN, betweenness pruning, and closeness pruning.

Run:

```bash
python main_train.py
```

The main settings are stored in:

```text
configs/main_experiments.yml
```

The configuration controls:

- datasets;
- reservoir sizes;
- forecasting horizons;
- pruning ratios;
- random seeds;
- ESN hyperparameters;
- output directories.

Because the complete configuration can contain many dataset, model, seed, horizon, and reservoir-size combinations, it is practical to begin with a reduced configuration containing:

- one dataset;
- one reservoir size;
- one forecasting horizon;
- one pruning ratio;
- one random seed.

After confirming that the environment is working correctly, restore the complete configuration.

---

## Running the controlled and supplementary experiments

The controlled pipeline performs matched comparisons in which pruning methods share the same original reservoir for a fixed seed.

### Quick smoke test

```bash
python run_supplementary.py \
  --config configs/supplementary_experiments.yml \
  --quick
```

This runs a small end-to-end experiment using a reduced reservoir, one dataset, and one seed.

### Complete experiment

```bash
python run_supplementary.py \
  --config configs/supplementary_experiments.yml
```

### Rebuild tables and plots without retraining

```bash
python run_supplementary.py \
  --config configs/supplementary_experiments.yml \
  --tables-only
```

This command rebuilds:

- seed-wise CSV files;
- aggregate CSV summaries;
- LaTeX tables;
- aggregate figures.

Completed runs are skipped when

```yaml
skip_existing: true
```

is enabled in the configuration. This allows interrupted experiments to be resumed without repeating completed runs.

---

## Included comparison methods

### Broad benchmark suite

- Base ESN
- Leaky ESN
- Deep ESN
- Betweenness-pruned ESN
- Closeness-pruned ESN
- Dynamical Mode Pruning

### Controlled one-shot suite

- Full ESN
- Fresh smaller ESN
- Random pruning without spectral-radius rescaling
- Random pruning with spectral-radius rescaling
- Recurrent magnitude pruning
- Activation-variance pruning
- Jacobian column-energy pruning
- Approximate weighted betweenness pruning
- DMP without spectral-radius rescaling
- DMP with spectral-radius rescaling

For a fixed seed, all pruning methods in the controlled suite:

- start from the same full reservoir;
- use the same data split;
- retain the same number of neurons;
- keep the recurrent weights fixed after pruning;
- refit only the linear readout.

---

## Jacobian column-energy baseline

The Jacobian column-energy baseline ranks neuron $i$ using

$$
s_i^{\mathrm{Jac}}
=
\mathcal{G}_{ii}
=
\frac{1}{T}
\sum_{t=1}^{T}
\left\|
J_{:,i}(t)
\right\|_2^2.
$$

This quantity is the trajectory-averaged squared column norm of the transition Jacobian.

The DMP score differs from this baseline because it uses only the dominant retained modes of the Gramian:

$$
\eta_i
=
\sum_{k=1}^{r}
\widetilde{\lambda}_k
V_{ik}^{2}.
$$

---

## Forecasting metrics

The controlled forecasting experiments save:

- final-horizon mean squared error;
- average mean squared error over horizons $1{:}h$;
- final-horizon normalized root mean squared error;
- average normalized root mean squared error over horizons $1{:}h$;
- recurrent parameter count;
- recurrent nonzero count;
- total parameter count;
- model size;
- median inference latency;
- inference-latency interquartile range;
- speedup relative to the full ESN;
- offline pruning time;
- readout-fitting time;
- total model-construction time.

### Mean squared error

At forecasting horizon $h$, the mean squared error is

$$
\operatorname{MSE}_h
=
\frac{1}{n_h}
\sum_{j=1}^{n_h}
\left(
\widehat{y}^{(h)}_j
-
y^{(h)}_j
\right)^2,
$$

where $n_h$ is the number of test predictions at horizon $h$.

### Root mean squared error

The root mean squared error is

$$
\operatorname{RMSE}_h
=
\sqrt{
\frac{1}{n_h}
\sum_{j=1}^{n_h}
\left(
\widehat{y}^{(h)}_j
-
y^{(h)}_j
\right)^2
}.
$$

### Normalized root mean squared error

The normalized root mean squared error is

$$
\operatorname{NRMSE}_h
=
\frac{
\operatorname{RMSE}_h
}{
\sigma\!\left(
\mathbf{y}^{(h)}
\right)
},
$$

where $\sigma(\mathbf{y}^{(h)})$ is the population standard deviation of the corresponding test targets.

---

## Inference-efficiency measurements

Inference latency includes:

1. one recurrent state update;
2. one linear readout evaluation.

The benchmark procedure:

- drives the reservoir to the test boundary;
- uses the same fixed input segment for each repeated measurement;
- resets the reservoir to the same starting state before each repeat;
- performs warm-up repetitions;
- reports median latency per time step;
- reports the interquartile range across measured repetitions;
- synchronizes CUDA before and after the timed section when GPU execution is used.

Model size is calculated from the in-memory sizes of:

- the recurrent matrix;
- the input matrix;
- the trained readout matrix.

---

## Autonomous dynamical evaluation

The autonomous Mackey–Glass experiment uses a one-step readout in closed loop. Each predicted value is fed back as the next input.

The analysis reports:

- one-step NRMSE;
- rollout NRMSE;
- valid prediction length;
- valid prediction fraction;
- power spectral density distance;
- autocorrelation-function distance.

### Rollout NRMSE

For an autonomous rollout of length $L$,

$$
\operatorname{NRMSE}_{\mathrm{rollout}}
=
\frac{
\sqrt{
\frac{1}{L}
\sum_{t=1}^{L}
\left(
\widehat{y}_t-y_t
\right)^2
}
}{
\sigma(\mathbf{y})
}.
$$

### Valid prediction length

The valid prediction length is the number of rollout steps before the normalized absolute error first exceeds a prescribed threshold.

The normalized absolute error at step $t$ is

$$
e_t^{\mathrm{norm}}
=
\frac{
\left|
\widehat{y}_t-y_t
\right|
}{
\sigma(\mathbf{y})
}.
$$

### Power spectral density distance

The power spectral density distance is the total-variation distance between the normalized target and predicted power spectra.

### Autocorrelation-function distance

The autocorrelation-function distance is the root mean squared difference between the target and predicted autocorrelation sequences over the configured lag range.

---

## Generated outputs

The exact output root is configured in the corresponding YAML file.

A controlled run creates a directory structure similar to:

```text
results/<experiment_name>/
├── resolved_config.yml
│
├── runs/
│   └── <dataset>/
│       ├── direct_h<horizon>/
│       │   └── seed_<seed>/
│       │       ├── full_esn/
│       │       ├── fresh_small_esn/
│       │       ├── random_no_rescale/
│       │       ├── random_rho/
│       │       ├── magnitude_rho/
│       │       ├── activation_variance_rho/
│       │       ├── jacobian_diagonal_rho/
│       │       ├── betweenness_rho/
│       │       ├── dmp_no_rescale/
│       │       └── dmp_rho/
│       │
│       └── dynamical_h1/
│           └── seed_<seed>/
│               ├── full_esn/
│               ├── fresh_small_esn/
│               ├── random_rho/
│               └── dmp_rho/
│
├── csv/
│   ├── all_seedwise_metrics.csv
│   ├── table1_matched_comparison_seedwise.csv
│   ├── table1_matched_comparison_mean_std.csv
│   ├── table1_paired_statistics.csv
│   ├── table2_ablation_efficiency_seedwise.csv
│   ├── table2_ablation_efficiency_mean_std.csv
│   ├── table3_dynamical_seedwise.csv
│   └── table3_dynamical_mean_std.csv
│
├── latex/
│   ├── table1_matched_comparison_seedwise.tex
│   ├── table1_matched_comparison_mean_std.tex
│   ├── table2_ablation_efficiency_seedwise.tex
│   ├── table2_ablation_efficiency_mean_std.tex
│   ├── table3_dynamical_seedwise.tex
│   └── table3_dynamical_mean_std.tex
│
└── plots/
    ├── aggregate/
    └── table3_dynamical/
```

Each direct forecasting method directory can contain:

```text
metrics.csv
predictions_all_horizons.csv
prediction_at_final_horizon.png
error_at_final_horizon.png
```

Each autonomous evaluation directory can contain:

```text
metrics.csv
autonomous_rollout.csv
```

---

## Supplementary material

The compact appendix source is located at:

```text
supplementary/supplementary.tex
```

It contains:

- three aggregate tables;
- two Temperature figures;
- concise explanations of the additional comparisons;
- the autonomous dynamical evaluation.

The supplementary figure paths point to the PDF files under:

```text
figures/
```

Include the supplementary source from the main LaTeX document with:

```latex
\input{supplementary/supplementary}
```

---

## Reproducibility notes

- Random seeds are set explicitly for NumPy and PyTorch.
- CUDA seeds are also set when CUDA is available.
- Test values do not influence preprocessing in the controlled pipeline.
- Forecasting targets are prevented from crossing into the test interval.
- Matched pruning methods share the same original reservoir for each seed.
- DMP scoring is one-shot in the controlled experiment pipeline.
- The same target reservoir size is used across matched pruning methods.
- The recurrent weights remain fixed after pruning.
- Only the linear readout is refitted.
- The Jacobian Gramian is evaluated from the driven training trajectory.
- The Jacobian column-energy baseline uses the diagonal of the full Gramian.
- Inference latency includes the recurrent state update and linear readout.
- Dense benchmark timing and streamlined one-shot timing have different computational scopes and are stored separately.
- The resolved configuration is saved with each experiment.
- Existing completed runs can be skipped and resumed safely.

---

## Configuration

The experiment behaviour is controlled through YAML files.

Typical parameters include:

```yaml
experiment:
  reservoir_size: 1000
  prune_ratio: 0.20
  direct_horizon: 20
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

esn:
  leak_rate: 0.4
  spectral_radius: 0.95
  input_scaling: 0.5
  bias_scaling: 0.2
  zero_fraction: 0.10
  ridge_alpha: 1.0e-6

dmp:
  energy_tau: 0.90
```

Refer to the files under `configs/` for the complete settings.

---

## Notes on method names

The public description uses the term **Jacobian column-energy pruning**.

Some internal output keys may retain the name

```text
jacobian_diagonal_rho
```

for compatibility with previously generated result files. This key refers to the same score:

$$
\mathcal{G}_{ii}
=
\frac{1}{T}
\sum_{t=1}^{T}
\left\|
J_{:,i}(t)
\right\|_2^2.
$$

---

## Citation

Please cite the accompanying paper when using this implementation:

```bibtex
@inproceedings{dmp_esn_2026,
  title     = {Rethinking Reservoir Pruning: A Dynamical Perspective for Echo State Networks},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

Replace the anonymous author field with the final camera-ready author list when the paper becomes public.

---

## Licence

Add the repository licence here after selecting the appropriate licence for public release.
