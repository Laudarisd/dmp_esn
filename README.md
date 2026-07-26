# Dynamical Mode Pruning for Echo State Networks

Official research code for **Dynamical Mode Pruning (DMP)**, a structured
reservoir-pruning method for Echo State Networks (ESNs). DMP ranks reservoir
neurons by their participation in dominant input-driven transition modes
obtained from a trajectory-averaged Jacobian Gramian.

The repository contains the complete experimental code used for the paper:

- broad benchmark experiments across reservoir sizes, forecasting horizons,
  and pruning ratios;
- Base, Leaky, and Deep ESN reference models;
- betweenness- and closeness-based reservoir pruning;
- controlled one-shot comparisons with fresh smaller reservoirs, random
  pruning, magnitude pruning, activation-variance pruning, Jacobian
  column-energy pruning, and betweenness pruning;
- spectral-radius ablations, inference-efficiency measurements, and autonomous
  Mackey--Glass evaluation;
- plotting, CSV aggregation, and paper-ready LaTeX table generation;
- the compact supplementary LaTeX source and its two figures.

## Method overview

For a leaky ESN with transition Jacobian

\[
J(t)=(1-a)I+aD(t)W,
\]

DMP forms the trajectory-averaged Jacobian Gramian

\[
\mathcal{G}=\frac{1}{T}\sum_{t=1}^{T}J(t)^{\top}J(t).
\]

After retaining the leading modes that explain a prescribed fraction of the
Gramian energy, neuron \(i\) is scored by

\[
\eta_i=\sum_{k=1}^{r}\widetilde{\lambda}_k V_{ik}^{2}.
\]

The highest-scoring neurons are retained, the corresponding principal
submatrix of the reservoir is extracted, and only the linear readout is
refitted. Spectral-radius rescaling is available as an explicit experimental
variant rather than an implicit part of the main method.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── main_train.py                      # Broad benchmark and baseline suite
├── run_supplementary.py               # Controlled one-shot analyses
├── configs/
│   ├── main_experiments.yml           # Benchmark, scaling, horizon, pruning
│   └── supplementary_experiments.yml  # Matched, efficiency, autonomous runs
├── data/
│   ├── electricity.csv
│   ├── mackey.csv
│   ├── solar.csv
│   ├── temperature.csv
│   └── wind.csv
├── figures/
│   ├── figure2.pdf
│   ├── figure3.pdf
│   ├── figure4.pdf
│   ├── temperature_accuracy_latency.pdf
│   └── temperature_matched_pruning_nrmse.pdf
├── src/
│   ├── dmp_model/                     # Full DMP benchmark implementation
│   │   ├── dmp_esn.py
│   │   └── dmp_results_save.py
│   ├── sota_models/                   # ESN and graph-pruning baselines
│   │   ├── base_esn.py
│   │   ├── leaky_esn.py
│   │   ├── deep_esn.py
│   │   ├── betweenness_esn.py
│   │   └── closeness_esn.py
│   ├── utils/
│   │   ├── data_cache.py
│   │   └── overleaf_table_formatter.py
│   └── supplementary/                 # Controlled one-shot experiment tools
│       ├── dmp_core.py
│       ├── evaluation.py
│       ├── plotting.py
│       └── reporting.py
├── supplementary/
│   └── supplementary.tex              # Compact three-table/two-figure appendix
├── results/                            # Generated outputs; ignored by Git
└── logs/                               # Runtime logs; ignored by Git
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

CUDA is optional. When CUDA is available and selected in the configuration,
PyTorch uses the GPU; otherwise the experiments run on CPU.

## Data

The included CSV files make the repository directly runnable and preserve the
input format used by the experiments. Each dataset is treated as a one-dimensional time series. The loaders automatically select a usable numeric
column.

For the controlled experiment pipeline, preprocessing is leakage-safe:
normalisation statistics are estimated from the washout and training region
only, and forecasting targets are prevented from crossing into the test
interval.

Dataset paths and split ratios are configured in the YAML files under
`configs/`.

## Running the benchmark suite

The broad benchmark pipeline evaluates DMP together with the Base ESN, Leaky
ESN, Deep ESN, betweenness pruning, and closeness pruning.

```bash
python main_train.py
```

Settings are read from:

```text
configs/main_experiments.yml
```

This configuration controls datasets, reservoir sizes, horizons, pruning
ratios, random seeds, model hyperparameters, and output directories.

Because the full configuration contains many model/dataset/seed combinations,
edit the YAML lists to a single dataset, reservoir size, horizon, and seed when
performing an initial environment check.

## Running the controlled and supplementary experiments

Run a small end-to-end smoke test first:

```bash
python run_supplementary.py \
  --config configs/supplementary_experiments.yml \
  --quick
```

Run the complete configured experiment:

```bash
python run_supplementary.py \
  --config configs/supplementary_experiments.yml
```

Rebuild CSV summaries, LaTeX tables, and aggregate plots from existing run
metrics without retraining:

```bash
python run_supplementary.py \
  --config configs/supplementary_experiments.yml \
  --tables-only
```

Completed runs are skipped when `skip_existing: true`, so interrupted
experiments can be resumed safely.

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
- Random pruning, with and without spectral-radius rescaling
- Recurrent magnitude pruning
- Activation-variance pruning
- Jacobian column-energy pruning
- Approximate weighted betweenness pruning
- DMP, with and without spectral-radius rescaling

For a fixed seed, all pruning methods in the controlled suite start from the
same full reservoir and retain the same number of neurons. The recurrent
weights remain fixed after pruning, and only the linear readout is refitted.

## Metrics

The controlled forecasting experiments save:

- final-horizon MSE;
- average MSE over horizons \(1{:}h\);
- final-horizon NRMSE;
- average NRMSE over horizons \(1{:}h\);
- recurrent and total parameter counts;
- exact tensor model size;
- median inference latency and interquartile range;
- speedup relative to the full ESN;
- offline pruning time and readout-fitting time.

At forecasting horizon \(h\), NRMSE is computed as

\[
\operatorname{NRMSE}_h=
\frac{\sqrt{\frac{1}{n_h}\sum_{j=1}^{n_h}
(\widehat{y}^{(h)}_j-y^{(h)}_j)^2}}
{\sigma(\mathbf{y}^{(h)})},
\]

where \(\sigma(\mathbf{y}^{(h)})\) is the population standard deviation of
the corresponding test targets.

The autonomous Mackey--Glass analysis additionally reports rollout NRMSE,
valid prediction length, valid prediction fraction, power spectral density
distance, and autocorrelation-function distance.

## Generated outputs

The exact output root is configured in each YAML file. A controlled run creates
an organized structure similar to:

```text
results/<experiment_name>/
├── resolved_config.yml
├── runs/
│   └── <dataset>/
│       ├── direct_h<horizon>/seed_<seed>/<method>/
│       │   ├── metrics.csv
│       │   ├── predictions_all_horizons.csv
│       │   ├── prediction_at_final_horizon.png
│       │   └── error_at_final_horizon.png
│       └── dynamical_h1/seed_<seed>/<method>/
│           ├── metrics.csv
│           └── autonomous_rollout.csv
├── csv/                                # Seed-wise and aggregate summaries
├── latex/                              # Paper-ready LaTeX tables
└── plots/                              # Per-run and aggregate figures
```

## Supplementary material

The compact appendix source is available at:

```text
supplementary/supplementary.tex
```

It contains only the three aggregate tables and two Temperature figures used
in the final supplementary material. The figure paths already point to the
PDF files under `figures/`.

Include it from the paper source with:

```latex
\input{supplementary/supplementary}
```

## Reproducibility notes

- Random seeds are set explicitly for NumPy and PyTorch.
- Test values do not influence preprocessing in the controlled pipeline.
- Matched pruning methods share the same original reservoir for each seed.
- DMP scoring is one-shot in the controlled pipeline.
- The Jacobian column-energy baseline uses
  \(\mathcal{G}_{ii}=T^{-1}\sum_t\|J_{:,i}(t)\|_2^2\).
- Inference latency includes the recurrent state update and linear readout.
- Dense benchmark timing and streamlined one-shot timing have different
  computational scopes and are intentionally stored separately.
- Numerical experiment settings are preserved in the resolved YAML file saved
  with each run.

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

Replace the anonymous author field with the final camera-ready author list
when the paper becomes public.
