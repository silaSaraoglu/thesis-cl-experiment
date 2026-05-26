# Thesis Project — Continual Learning Benchmark

Comparative study of continual learning (CL) strategies under two evaluation settings:
**Domain-Incremental Learning (DIL)** and **Task-Incremental Learning (TIL)**.
Two model families are benchmarked — **GIM** (Growing and Interpolating Memory networks) and
**ESN** (Echo State Networks) — across two datasets.

---

## Table of Contents

- [Continual Learning Settings](#continual-learning-settings)
- [Datasets](#datasets)
- [Models and Strategies](#models-and-strategies)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Tuning and Evaluation Pipeline](#tuning-and-evaluation-pipeline)
- [Metrics](#metrics)
- [Key Design Decisions](#key-design-decisions)
- [Dependencies](#dependencies)

---

## Continual Learning Settings

Three standard CL settings exist, ordered from hardest to easiest:

| Setting | Abbrev. | Task ID at test | Output head | Challenge |
|---------|---------|-----------------|-------------|-----------|
| Class-Incremental | CIL | Not provided | Single head, grows with classes | Must distinguish all seen classes; random chance degrades each task |
| Domain-Incremental | DIL | Not provided | Single head, fixed output space | Must not forget previous tasks; same output space throughout |
| Task-Incremental | TIL | Provided (oracle) | Separate head per task | Largely trivial with frozen features — serves as upper bound |

> **Note for supervisor:** The original ESN repository (ESANN 2021) operates in the **CIL**
> setting — a single 10-class output head, no task identity provided, new digit classes
> introduced each experience (`nc_benchmark(..., task_labels=False)`). This is the hardest
> of the three settings. Our thesis adapts the ESN to **DIL** (`single_head/`) and **TIL**
> (`multi_head/`) to benchmark it on the same footing as GIM, which was originally designed
> for the DIL setting.
>
> **How the adaptation was made:**
>
> | Dimension | Original ESN (CIL) | Our adaptation |
> |-----------|-------------------|----------------|
> | Task structure | 5 experiences, 2 new digit classes added each time | 5 binary tasks (0vs1, 2vs3, …) — classes fixed per task, no new classes introduced |
> | Output head | Single 10-class linear readout | SH: single 2-class readout shared across tasks · MH: one independent 2-class readout per task (`MultiHeadESN`) |
> | Task ID at test | Not provided | SH: not provided (DIL) · MH: provided as oracle (TIL) |
> | Sequence format | 28 timesteps × 28 features (one full pixel row per step) | 28 timesteps × 28 features (one pixel row per step, full 28×28 image) — same as original ESN format |
> | Second dataset | Not present (MNIST + SSC only) | WISDM added: 3 binary tasks, 128 timesteps × 3 accelerometer channels |
> | Reservoir hyperparams | Tuned via grid search (spectral radius, input scaling, leaky) | Fixed to original paper values (0.99 / 1.0 / 1.0) — not re-tuned |
> | Training hyperparams | Fixed in config (lr=1e-3, units=500, epochs=10, batch=128) | Re-tuned per dataset via Optuna TPE (task structure and sequence length differ) |
>
> The reservoir itself (`DeepReservoirClassifier`) is used unchanged from the original
> repository. Only the task-splitting logic, output head structure, and training loop
> wrapper are new.

### Setting 1 — Domain-Incremental Learning (DIL) · `single_head/`

The model has a **single shared output head** for all tasks. At test time, **no task identity
is provided** — the model must classify correctly without knowing which task it is on.

- GIM routes inputs to the correct module using **autoencoder reconstruction error** (the module
  with the lowest reconstruction error on the input is selected).
- ESN uses a single readout layer updated incrementally across tasks.
- Catastrophic forgetting is the key challenge.

### Setting 2 — Task-Incremental Learning (TIL) · `multi_head/`

The model has **one output head per task** stored in a `ModuleDict`. At test time, the
**oracle task label** (ground truth from the dataset) is provided, and the correct head is
selected automatically.

- GIM routes inputs using the ground-truth task index — autoencoders are **not** trained or used.
- ESN SLDA fits one independent `LinearDiscriminantAnalysis` per task (batch, offline).
- Because heads are separate, there is no cross-task interference; catastrophic forgetting
  is largely eliminated by design. For ESN specifically, the frozen reservoir means there are
  **no shared mutable parameters across tasks** — TIL performance is the theoretical ceiling.

---

## Datasets

### MNIST

Split into **5 sequential binary tasks**:

| Task | Classes |
|------|---------|
| 1 | 0 vs 1 |
| 2 | 2 vs 3 |
| 3 | 4 vs 5 |
| 4 | 6 vs 7 |
| 5 | 8 vs 9 |

Full 28×28 images are used (no downsampling). Each image is fed to the model as a
**row-based sequence** — one pixel row per timestep, giving 28 timesteps with 28 features
each:

```
Pixel grid (28×28):          Sequence fed to model:
┌──────────────────────────┐
│ p1   p2  ...  p28   row1 │         t=1  → model ← [p1, p2, ..., p28]   (28-dim vector)
│ p29  p30 ...  p56   row2 │  →→→   t=2  → model ← [p29, p30, ..., p56]
│ ...                      │         ...
└──────────────────────────┘         t=28 → model ← [p757, ..., p784]
```

At each step the model processes one full row of pixels (28 features), updating its hidden
state with the accumulated spatial context. Only the **final hidden state** (after row 28)
is passed to the output head for classification.

This row-based format (`input_size=28`, 28 timesteps) is the standard sequential MNIST
convention from the original ESN repository. It provides each timestep with meaningful
spatial context (a full image row) rather than a single pixel, making the sequence
structurally richer and more learnable for both LSTM and ESN models.

### WISDM — Wireless Sensor Data Mining

6 smartphone accelerometer activities split into **3 sequential binary tasks**:

| Task | Classes | Activity type |
|------|---------|---------------|
| 1 | Walking vs Jogging | locomotion vs locomotion |
| 2 | Upstairs vs Downstairs | directional locomotion |
| 3 | Sitting vs Standing | static vs static |

Input shape: `(B, 128, 3)` — 128 time steps, 3 accelerometer channels (x, y, z).
Labels are remapped to `{0, 1}` per task.

Data is stored under `data/wisdm/`. The preprocessed file `wisdm_processed.npz` is
generated automatically on first run (z-score normalised using training statistics).

---

## Models and Strategies

### GIM — Growing and Interpolating Memory

Based on the [GIM repository](repos/gim/). Uses adaptive LSTM (`ALSTM`) as the recurrent body.
New modules are grown when validation accuracy on the current task falls below a threshold.

| Variant | SH (DIL) | MH (TIL) |
|---------|-----------|-----------|
| GIM-ALSTM | AE routing | Oracle routing |
| LSTM Naive | Single shared head | Per-task heads |
| LSTM Joint | All tasks simultaneously (upper bound) | All tasks simultaneously (upper bound) |

> **Joint training is not applicable to GIM.** GIM's autoencoder-driven module-growing
> mechanism is inherently sequential — it decides whether to grow a new module based on
> reconstruction error from the *current* task. When all tasks are presented simultaneously
> this decision criterion becomes undefined, so there is no meaningful joint-training variant.
> Joint LSTM is included as an oracle upper bound for the vanilla LSTM baseline only.

### ESN — Echo State Network

Based on the [ESN + Avalanche repository](repos/esn/). Uses a **frozen deep reservoir**
(`DeepReservoirClassifier`) as a fixed feature extractor. Only the readout layer is trained.

#### Single-Head (DIL) ESN strategies

| Variant | Description |
|---------|-------------|
| Naive | No CL strategy — shared readout trained sequentially |
| EWC | Elastic Weight Consolidation on the readout weights |
| LwF | Learning without Forgetting — distillation from previous task outputs |
| Replay | Experience replay buffer |
| SLDA | StreamingLDA — incremental LDA classifier on reservoir features |
| Joint | All tasks simultaneously (oracle upper bound) |

#### Multi-Head (TIL) ESN strategies

| Variant | Description |
|---------|-------------|
| Naive | No CL strategy — per-task independent readout heads |
| Joint | All tasks simultaneously (oracle upper bound) |

> **Multi-head ESN CL strategies (EWC/LwF/Replay) are architecturally vacuous** and are
> therefore not included in the MH pipeline. The ESN reservoir (`DeepReservoir`) has all
> weights registered as `nn.Parameter(..., requires_grad=False)` in the original repository
> — it is frozen by design and cannot learn or forget anything. In the multi-head setting
> each task gets its own independent `nn.Linear` head, and only that head's parameters are
> passed to the optimizer when training on that task. Once a task is done its head is never
> touched again. The result is that there are **no shared mutable parameters across tasks**:
> EWC has no Fisher information to compute, LwF has no shared output to distil from, and
> Replay cannot influence a body that does not change. The MH Naive baseline already captures
> the ceiling performance for this architecture.

---

## Project Structure

```
thesis_project/
├── README.md
│
├── shared/                             # Canonical utilities shared by both settings
│   └── utils.py                        # collect_datasets(), gim_predict()
│
├── single_head/                        # Setting 1 — DIL
│   ├── run_experiment.py               # Optuna tuning + multi-run evaluation orchestrator
│   ├── run_overnight_sh.py             # Launches run_experiment.py as a background daemon
│   ├── utils/
│   │   └── metrics.py                  # CLMetrics, cohen_kappa, save_results
│   └── models/
│       ├── GIM_LSTM/                   # GIM-ALSTM experiments
│       │   ├── gim_lstm_mnist_sh.py
│       │   └── gim_lstm_wisdm_sh.py
│       ├── LSTM/                       # Vanilla LSTM baselines
│       │   ├── model.py                # build_lstm() factory
│       │   ├── naive/
│       │   │   ├── naive_lstm_mnist_sh.py
│       │   │   └── naive_lstm_wisdm_sh.py
│       │   └── joint/
│       │       ├── joint_lstm_mnist_sh.py
│       │       └── joint_lstm_wisdm_sh.py
│       └── ESN/
│           ├── esn_utils.py            # build_model(), predict_step()
│           ├── naive/
│           ├── joint/
│           ├── ewc/
│           ├── lwf/
│           ├── replay/
│           └── slda/
│
├── multi_head/                         # Setting 2 — TIL
│   ├── run_experiment.py               # Same structure as SH orchestrator
│   ├── run_overnight_mh.py
│   ├── utils/
│   │   └── metrics.py
│   └── models/
│       ├── multi_head_utils.py         # MultiHeadLSTM, MultiHeadESN, _forward_mh, _predict_mh
│       ├── GIM_LSTM/                   # GIM-ALSTM experiments
│       │   ├── gim_lstm_mnist_mh.py
│       │   └── gim_lstm_wisdm_mh.py
│       ├── LSTM/                       # Vanilla LSTM baselines
│       │   ├── naive/
│       │   │   ├── naive_lstm_mnist_mh.py
│       │   │   └── naive_lstm_wisdm_mh.py
│       │   └── joint/
│       │       ├── joint_lstm_mnist_mh.py
│       │       └── joint_lstm_wisdm_mh.py
│       └── ESN/
│           ├── naive/
│           └── joint/
│
├── repos/
│   ├── gim/                            # Original GIM codebase (unmodified)
│   └── esn/                            # Original ESN + Avalanche codebase (unmodified)
│
├── data/
│   ├── mnist/                          # Auto-downloaded by torchvision
│   └── wisdm/                          # WISDM Dataset + wisdm_processed.npz
│
└── results/
    ├── single_head/
    │   └── experiment_YYYYMMDD_HHMMSS/ # One folder per run
    │       ├── {model}_{dataset}.json  # Per-combination results
    │       ├── plots/                  # PNG metric charts and R-matrix heatmaps
    │       └── summary.json            # Aggregated results for all combinations
    └── multi_head/
        └── experiment_YYYYMMDD_HHMMSS/
```

---

## How to Run

### Overnight pipeline (recommended)

Launches the full experiment suite as a background daemon with its own log file:

```powershell
# Single-head (DIL) — runs all model-dataset combinations
cd single_head
python run_overnight_sh.py

# Multi-head (TIL) — same structure
cd multi_head
python run_overnight_mh.py
```

Each overnight script prints the log path and PID on startup:

```
Single-Head (DIL) experiment started — safe to close this terminal.
  PID     : 1234
  Log     : single_head/logs/overnight_sh_YYYYMMDD_HHMMSS.log
  Monitor : Get-Content -Wait 'single_head/logs/overnight_sh_....log'
  Stop    : Stop-Process -Id 1234
```

### Direct invocation

```powershell
# Run full pipeline (tuning + evaluation) with optional overrides
python single_head/run_experiment.py
python single_head/run_experiment.py --n_trials 5 --tune_subset 100
python single_head/run_experiment.py --num_runs 3 --subset 500
```

---

## Tuning and Evaluation Pipeline

Each model-dataset combination goes through a two-phase pipeline inside `run_experiment.py`:

### Phase 1 — Hyperparameter tuning (Optuna TPE)

- `N_TRIALS = 10` trials per study (GIM, LSTM, Joint-LSTM, ESN-Base, Joint-ESN)
- `N_STRATEGY_TRIALS = 5` trials for ESN strategy-specific studies (EWC/LwF/Replay — fewer
  parameters to tune)
- `TUNE_SUBSET_MNIST = 500` samples per task during MNIST tuning
- `TUNE_SUBSET_WISDM = 500` samples per task during WISDM tuning
- Up to 4 parallel workers (Phase 1a: independent studies; Phase 1b: ESN secondary studies
  that depend on ESN-Base results)
- GIM on MNIST uses fixed original-paper hyperparameters (not tuned)
- ESN strategies (EWC/LwF/Replay) share the ESN-Base reservoir parameters and only tune
  their strategy-specific extras

#### MNIST search spaces

| Model | hidden_size / esn_units | epochs | batch_size |
|-------|------------------------|--------|------------|
| LSTM Naive | [64, 128, 256] | 3–7 | [32, 64, 128] |
| LSTM Joint | [64, 128, 256] | 2–5 | [32, 64, 128] |
| ESN | [200, 500, 1000] | 2–5 | [32, 64, 128] |

#### WISDM search spaces

| Model | hidden_size / esn_units | epochs | batch_size |
|-------|------------------------|--------|------------|
| LSTM Naive | [32, 64, 128, 256] | 3–7 | [16, 32, 64] |
| LSTM Joint | [32, 64, 128, 256] | 3–7 | [16, 32, 64] |
| ESN | [100, 200, 500] | 3–7 | [16, 32, 64] |
| GIM | [32, 64, 128] (rnn) | 3–7 | [16, 32, 64] |

### Phase 2 — Multi-run evaluation

- `SUBSET_MNIST = 5000` samples per task for full MNIST runs
- `SUBSET_WISDM = 2000` samples per task for full WISDM runs
- `NUM_RUNS = 1` full evaluation run with best hyperparameters
- Results for all combinations saved to a single timestamped `experiment_YYYYMMDD_HHMMSS/`
  folder containing individual JSONs, plots, and a `summary.json`

---

## Metrics

Implemented in `utils/metrics.py` (both SH and MH). All metrics are computed from the
accuracy matrix `R` where `R[i][j]` = test accuracy on task `j` after training on task `i`.

| Metric | Formula | Meaning |
|--------|---------|---------|
| **ACC_final** | mean of last row of R | Average accuracy after all tasks |
| **ACC_avg** | mean of lower-left triangle of R | Average accuracy over all evaluation points |
| **BWT_final** | mean of `R[N-1][j] - R[j][j]` for j < N-1 | Backward transfer (forgetting) after all tasks |
| **BWT_avg** | mean of `R[i][j] - R[j][j]` for i > j | Average backward transfer |
| **FWT** | mean of `R[i-1][i] - baseline[i]` | Forward transfer (zero-shot to next task) |
| **Cohen's κ** | mean κ on final model across all tasks | Agreement beyond chance |
| **Plasticity** | mean of diagonal of R | How well each task is learned right after training |
| **Stability** | mean of `R[N-1][j]` for all j | How well all tasks are retained at the end |

---

## Key Design Decisions

### WISDM task pairing
Activities are paired within similar motion categories: Walking/Jogging (both locomotion),
Upstairs/Downstairs (directional locomotion), Sitting/Standing (both static). Each task is a
binary classification problem with labels remapped to `{0, 1}`.

### `shared/utils.py`
All utilities shared between `single_head/` and `multi_head/` live in `shared/utils.py`:
- `collect_datasets(train_cl, test_cl, task_list, max_samples, batch_size, reshape)` —
  materialises `TensorDataset` objects for all tasks upfront. Used by all ESN files.
  `reshape=True` removes the channel dim `(B, 1, H, W) → (B, H, W)` for MNIST sequences.
- `gim_predict(train_models, x, task_id)` — one forward pass through the GIM-ALSTM model,
  returning predicted class labels.

### GIM autoencoders in SH vs MH
- **SH**: Autoencoders are trained every epoch and used at test time to route inputs to
  the module with the lowest reconstruction error (unsupervised module selection).
- **MH**: Autoencoders are allocated by the model factory but never trained and never
  used — the oracle task label provides routing directly.

### ESN SLDA in SH vs MH
- **SH**: One `StreamingLDA` classifier updated incrementally across all tasks (no task
  boundary information used).
- **MH**: One independent `sklearn.LinearDiscriminantAnalysis` fitted per task after
  extracting all reservoir features for that task (offline batch fit, oracle task ID used
  at test time).

### Joint training as oracle upper bound
Joint training (all tasks simultaneously) serves as the oracle upper bound for vanilla LSTM
and ESN models. It is not applicable to GIM because GIM's module-growing criterion depends
on sequential task presentation.

### Val/test loader shuffle
Validation and test `DataLoader` objects always use `shuffle=False`. Only training loaders
use `shuffle=True`. This is consistent across all model files in both SH and MH.

---

## Dependencies

```
torch
torchvision
scikit-learn
numpy
optuna
avalanche-lib        # from repos/esn/
```

The original GIM and ESN repositories in `repos/` are used directly via `sys.path`
injection — no installation required.
