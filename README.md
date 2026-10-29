# Thesis Project — Continual Learning Benchmark

A comparative study of continual learning (CL) strategies across four sequence
datasets, evaluated under two settings — **Domain-Incremental Learning (DIL,
[single_head/](single_head/))** and **Task-Incremental Learning (TIL,
[multi_head/](multi_head/))**.

Every dataset is turned into a sequence of **tasks** presented one after another, so all
model families (GIM, LSTM, ESN) see the same recurrent input format:
`(batch, timesteps, features)`. Each dataset is run under **15 task configurations**
(orderings / pairings), and results are kept **per configuration — never averaged**.
Those 15 configurations are also what the statistical tests treat as matched blocks
(see [Statistical testing](#statistical-testing)).

### What differs between the two settings

|  | `single_head/` (DIL) | `multi_head/` (TIL) |
|---|---|---|
| Output head | **one shared head**, no task ID at test time | **one head per task**, oracle task ID at test time |
| Datasets | MNIST, HHAR, Fashion-SW | MNIST, HHAR, Fashion-SW, **WISDM** |
| Models | 10 (full ESN roster) | 6 (ESN Naive + Joint only) |
| Forward transfer | reported | **not** reported (≈ chance by construction) |

The dataset and model lists are the `DATASETS` / `MODELS` constants in each setting's
`config.py` ([single_head/config.py](single_head/config.py),
[multi_head/config.py](multi_head/config.py)).

### Original repositories

The two model families are built on top of their authors' original codebases (vendored
under `repos/`, lightly adapted for this thesis):

- **GIM** — Gated Incremental Memory:
  [AndreaCossu/ContinualLearning-SequentialProcessing](https://github.com/AndreaCossu/ContinualLearning-SequentialProcessing)
  · Cossu et al., *Continual Learning with Gated Incremental Memories for Sequential
  Data Processing*, IJCNN 2020 ([arXiv:2004.04077](https://arxiv.org/abs/2004.04077))
- **ESN** — Echo State Networks:
  [Pervasive-AI-Lab/ContinualLearning-EchoStateNetworks](https://github.com/Pervasive-AI-Lab/ContinualLearning-EchoStateNetworks)
  · Cossu et al., *Continual Learning with Echo State Networks*, ESANN 2021
  ([arXiv:2105.07674](https://arxiv.org/abs/2105.07674))

---

## Project layout

```
thesis_project/
├── shared/                     code used by both settings
│   ├── dataset_cl.py           dataset classes + sensor preprocessing
│   ├── utils.py                input plumbing (permutation, loaders)
│   ├── replay_buffer.py        hand-rolled LSTM replay memory
│   ├── metrics.py              CLMetrics: the R matrix and every CL metric
│   ├── stats_tests.py          permutation / Holm / Friedman / Nemenyi
│   ├── cd_diagram.py           Critical Difference diagram
│   ├── epoch_analysis.py       cross-budget epoch charts
│   └── plot_style.py           shared print-sized figure style
│
├── single_head/                DIL — one shared head, no task ID
│   ├── config.py               datasets, models, seed, 15 task configs
│   ├── run_tuning.py           Optuna search -> tuned_params.json
│   ├── run_experiment.py       main comparison (reads the cache)
│   ├── run_epoch_analysis.py   epoch-budget sweep (reads the cache)
│   ├── resume_experiment.py    fill in missing combinations of an existing run
│   └── models/                 GIM_LSTM/ · LSTM/{naive,joint,replay} · ESN/{naive,ewc,lwf,replay,slda,joint}
│
├── multi_head/                 TIL — one head per task, oracle task ID
│   ├── config.py · run_tuning.py · run_experiment.py · run_epoch_analysis.py
│   └── models/                 GIM_LSTM/ · LSTM/{naive,joint,replay} · ESN/{naive,joint}
│
├── run_significance.py         statistical tests over finished results
├── replot.py                   rebuild figures from a finished run
├── repos/                      vendored GIM + ESN codebases
├── data/                       raw + cached preprocessed datasets
└── results/{single_head,multi_head}/experiment_<ts>/
```

---

## Running the pipeline

Each setting is driven by **three scripts that share one `config.py`** and communicate
through a cached hyper-parameter file. Run them in order — commands shown for
`single_head/`, identical for `multi_head/`:

```bash
# 1. Tune once  — writes tuned_params.json (the shared, seed-tagged cache)
python single_head/run_tuning.py

# 2. Main comparison — reads the cache, runs every model × dataset × 15 configs
python single_head/run_experiment.py

# 3. Epoch-sensitivity sweep — reads the cache, sweeps the epoch budget
python single_head/run_epoch_analysis.py
```

**Why three scripts (the tuning-cache split).** Tuning is the slow part and only needs to
happen once. `run_tuning.py` **owns all tuning logic** — the search spaces, the Optuna
studies, the tuner — and writes the best hyper-parameters to `tuned_params.json`, tagged
with the `SEED` they were found under. `run_experiment.py` and `run_epoch_analysis.py`
contain **no tuning code**; they only *read* that cache, so both experiment scripts reuse
one identical set of hyper-parameters and never re-tune. If the cache is missing, was built
under a different `SEED`, or doesn't cover a requested `(model, dataset)`, the readers stop
with a `TuningCacheError` telling you to run `run_tuning.py` first. (`run_tuning.py` itself
tunes **incrementally** — it only tunes datasets not already present in the cache, then
appends them.)

`--datasets` restricts any of the three to a subset of datasets:

```bash
python single_head/run_tuning.py     --datasets hhar              # tune just HHAR
python single_head/run_experiment.py --datasets hhar fashion_sw   # run a subset
```

### Post-processing (no retraining)

Three scripts read a *finished* experiment directory and never touch a model:

```bash
# statistical comparison of the models — see the Statistical testing section
python run_significance.py results/single_head/experiment_20260810_155645

# regenerate every figure at the current print size
python replot.py results/single_head/experiment_20260810_155645

# fill in only the (model, dataset) combinations missing from an existing run
python single_head/resume_experiment.py --exp-dir experiment_20260810_155645 --datasets hhar
```

`resume_experiment.py` exists for `single_head/` only. It skips any combination whose
`<model>_<dataset>.json` is already present, runs the rest, then rebuilds `summary.json`,
the plots and the tables from the full set — reusing the tuning cache throughout.

---

## Datasets

Four datasets are used, each presented as a sequence of **tasks**. Three are **binary**
(the two raw labels per task are remapped to `{0, 1}` and share a 2-class head); **HHAR**
is the exception — a **6-way** activity-classification problem whose tasks are *devices*
(shared 6-class head). Every dataset is run under **15 task configurations**
(orderings / pairings) and results are kept **per configuration — not averaged**. The
tables below show the first three of each as examples; the full lists live in each
setting's `config.py` (`*_DATASET_CONFIGS`).

| Dataset | Domain | Type | # Tasks | Per-task classes | Input (T × F) | Split basis |
|---------|--------|------|:-------:|:----------------:|:-------------:|-------------|
| **MNIST** | handwritten digits | image→seq | 5 | 2 (digit pair) | 28 × 28 | official train/test · **random** val |
| **Fashion-MNIST SW** | clothing: summer vs winter | image→seq | 3 | 2 | 28 × 28 | official train/test · **random** val |
| **WISDM** | phone accelerometer (HAR) | sensor | 3 | 2 (activity pair) | 200 × 3 | **subject-independent** (by user) |
| **HHAR** | phone accelerometer (HAR) | sensor | 4 (devices) | 6 activities | 128 × 3 | **subject-independent** (by user) |

> **WISDM is multi-head only.** `single_head/config.py` lists `mnist`, `hhar`,
> `fashion_sw`; WISDM appears only in `multi_head/`. There are therefore no single-head
> WISDM results to compare against.

Dataset classes live in [shared/dataset_cl.py](shared/dataset_cl.py); task lists and the
per-dataset configurations are defined in each setting's `config.py` as the
`*_DATASET_CONFIGS` dicts. See the **Inputs** section at the end for how a sample is fed
to the model.

---

### MNIST — 5 tasks (digit pairs)

The standard 60,000 / 10,000 train/test split is grouped into **5 binary tasks**, one
per digit pair. Default pairing (`config_1`):

| Task | Classes | Train | Test |
|:----:|:-------:|:-----:|:----:|
| 1 | 0 vs 1 | 12,665 | 2,115 |
| 2 | 2 vs 3 | 12,089 | 2,042 |
| 3 | 4 vs 5 | 11,263 | 1,874 |
| 4 | 6 vs 7 | 12,183 | 1,986 |
| 5 | 8 vs 9 | 11,800 | 1,983 |
| **Total** | | **60,000** | **10,000** |

- **Class balance:** near-balanced within each task (digit counts range 5,421–6,742
  in train), so no resampling is applied.
- **Per-task size:** all tasks are roughly equal (~11–13k train each).

**Task configurations.** To probe the effect of *which* digits are paired, MNIST is
run with **15** pairings (`config_1`–`config_15`); results are reported **per
configuration (not averaged)**. The first three:

| Config | T1 | T2 | T3 | T4 | T5 |
|--------|----|----|----|----|----|
| `config_1` | 0/1 | 2/3 | 4/5 | 6/7 | 8/9 |
| `config_2` | 1/2 | 3/4 | 5/6 | 7/8 | 0/9 |
| `config_3` | 0/3 | 2/5 | 4/7 | 6/9 | 8/1 |

#### Preprocessing & permutation

1. **Source.** Loaded via `torchvision.datasets.MNIST` (`MNIST_CL` subclass), with
   `ToTensor()` scaling pixels to `[0, 1]`. No downsampling — full 28×28 images.
2. **Fixed pixel permutation.** Every image is shuffled by a **single fixed
   permutation of the 784 pixels**, taken from the original GIM paper
   (`permutations.npy[0]`). This is *not* a different permutation per task — it is one
   global shuffle applied identically to all images, reproducing the GIM
   "permuted-MNIST" protocol. The permuted pixels are then reshaped **back to 28×28**,
   so the sequence format is preserved. The permutation is loaded once as `MNIST_PERM`
   ([utils.py:23](shared/utils.py#L23)) and applied inline wherever MNIST is loaded
   ([utils.py:35](shared/utils.py#L35), [utils.py:43](shared/utils.py#L43)) or as the
   per-model `px()` helper (e.g.
   [naive_lstm_mnist_sh.py:36](single_head/models/LSTM/naive/naive_lstm_mnist_sh.py#L36)).
3. **Row-as-timestep sequence.** The 28×28 image is fed as a sequence of **28
   timesteps × 28 features** (`input_size=28`) — one full pixel row per step. Only the
   final hidden state is classified.

```
28×28 image (after fixed permutation)        sequence to the model
┌────────────────────────────┐
│  row 0  →  [p0 … p27]       │   t=0  →  [28-dim row vector]
│  row 1  →  [p28 … p55]      │   t=1  →  [28-dim row vector]
│   …                         │   …
│  row 27 →  [p756 … p783]    │   t=27 →  [28-dim row vector]
└────────────────────────────┘
```

The same permutation + reshape is applied identically by every MNIST model — GIM, LSTM
(naive/joint/replay), and ESN — so all models train on the exact same inputs.

#### Relation to the original GIM / ESN papers (and why we re-tune)

This MNIST setup borrows from two sources but matches neither exactly:

- The **GIM paper** uses *permuted MNIST* fed pixel-by-pixel (784 steps). We keep its
  fixed pixel permutation but reshape to 28×28, so the sequence is 28 steps × 28
  features.
- The **ESN paper** uses *Split MNIST* fed one **natural (unpermuted) row** at a time, as
  a **class-incremental** 10-way problem. We instead present **binary** tasks
  (Domain-Incremental in `single_head/`, Task-Incremental in `multi_head/`).

Because both the input ordering (permuted vs natural) and the CL scenario differ from
those papers, the per-strategy hyperparameters in the original ESN configs
([repos/esn/CONFIGS/smnist_28/](repos/esn/CONFIGS/smnist_28)) are used only as a loose
reference, not copied. Every `(model, dataset)` configuration is tuned independently with
Optuna — including the ESN CL knobs `ewc_lambda`, `lwf_alpha`, and `lwf_temperature` — so
no hard-coded strategy default is carried over. Replay sizes its memory from the data
(10% of each task's training set) rather than from a fixed constant.

---

### Fashion-MNIST Summer-vs-Winter — 3 tasks (season)

The standard Fashion-MNIST images (60,000 / 10,000, 28×28) under a **summer(0) vs
winter(1)** label scheme (`FashionMNISTSeason_CL`), with summer = {Tshirt, Dress, Sandal}
and winter = {Pullover, Coat, Ankle-boot}. Each of the 3 tasks pairs one summer class with
one winter class, so the shared head is binary; preprocessing/permutation is identical to
MNIST (28 rows × 28 features). Each Fashion-MNIST class holds exactly **6,000 train /
1,000 test**, so every binary task is **12,000 train / 2,000 test**, perfectly balanced.
It is run with **15** configurations (`config_1`–`config_15`); only 6 distinct
summer→winter matchings of the three pairs exist, so the configs vary both the matching
and the task order. The first three:

| Config | T1 | T2 | T3 |
|--------|----|----|----|
| `config_1` | Tshirt/Pullover | Dress/Coat | Sandal/Ankle-boot |
| `config_2` | Sandal/Pullover | Dress/Ankle-boot | Tshirt/Coat |
| `config_3` | Dress/Pullover | Sandal/Coat | Tshirt/Ankle-boot |

The first element of every pair is the summer class, the second the winter class, so the
`{0,1}` mapping is consistent across all tasks and configs.

---

### WISDM — 3 tasks (activity pairs)

Wireless Sensor Data Mining accelerometer dataset (`WISDM_ar_v1.1`). 6 activities are
paired into **3 binary tasks**, with **non-overlapping** 200-step windows. After
balancing there are **159 train / 72 test windows per class**, i.e. **318 train / 144
test per task** (954 / 432 total — matching the cached
`data/wisdm/wisdm_processed.npz`):

| Task | Classes | Train | Test |
|:----:|:-------:|:-----:|:----:|
| 1 | Walking vs Jogging   | 318 | 144 |
| 2 | Upstairs vs Downstairs | 318 | 144 |
| 3 | Sitting vs Standing  | 318 | 144 |
| **Total** | | **954** | **432** |

- **Class balance:** every class is downsampled to the smallest class count (seed 0),
  so the three tasks are equal in size.

**Task configurations.** WISDM is run with **15** configurations (`config_1`–`config_15`).
`config_1`–`config_3` keep the three semantic activity pairs (Walk/Jog · Up/Down ·
Sit/Stand) in different presentation orders, while `config_4`–`config_15` **mix the
pairings** (like MNIST / Fashion-SW), each still using all 6 classes once. There are
exactly 15 ways to split 6 classes into 3 pairs, so no configuration repeats a pairing.
The first three:

| Config | T1 | T2 | T3 |
|--------|----|----|----|
| `config_1` | Walk/Jog | Up/Down | Sit/Stand |
| `config_2` | Walk/Jog | Sit/Stand | Up/Down |
| `config_3` | Up/Down | Walk/Jog | Sit/Stand |

#### Preprocessing

Run automatically on first use; the result is cached to
`data/wisdm/wisdm_processed.npz` (`wisdm_preprocess`,
[shared/dataset_cl.py](shared/dataset_cl.py)):

1. **Parse** the raw text file (`user_id, activity, x, y, z` per line); only the 6
   mapped activities are kept.
2. **Sliding windows** per `(user, activity)`: window length **200** (10 s at 20 Hz),
   stride **200** — i.e. **non-overlapping** (the 200-step length follows Azghan et al.,
   *Gated Adaptation for Continual Learning in HAR*, arXiv:2603.10046; we drop the
   50% overlap to avoid near-duplicate windows leaking across splits).
3. **Subject-independent split:** users are split **36 → 28 train / 8 test**; all of a
   user's windows stay on one side, so no person appears in both train and test. The
   user id of every window is saved so the validation split can also hold out whole users.
4. **Class balancing:** every class is downsampled to the smallest class count (seed 0).
5. **Z-score normalisation** per channel, using **train statistics only**.

---

### HHAR — 4 tasks (devices), 6-way activity classification

Heterogeneity Activity Recognition phone accelerometer data. Unlike the others, HHAR is
**not** binary: each task is a **device model** and the model classifies **6 activities**
(bike, sit, stairsdown, stairsup, stand, walk) through a shared 6-class head — a
domain-incremental setting where the *domain* is the phone. Windows are **non-overlapping**
128-step segments of (x, y, z). After capping and balancing (below) every device holds the
same number of windows. Totals: **27,288 train / 8,928 test** windows (matching the cached
`data/hhar/hhar_processed.npz`).

| Task | Device | Train windows | Test windows |
|:----:|--------|:-------------:|:------------:|
| 1 | nexus4 | 6,822 | 2,232 |
| 2 | s3 | 6,822 | 2,232 |
| 3 | s3mini | 6,822 | 2,232 |
| 4 | samsungold | 6,822 | 2,232 |

- **Class-balanced per device:** windows are first capped at 1,500 per
  (user, device, activity), then each device's 6 activities are downsampled to an equal
  count (seed 0) — so within every device all 6 activities are balanced, and all four
  devices end up the same size (train 1,137 / test 372 per activity per device).

**Task configurations.** The four devices are presented in **15** orders
(`config_1`–`config_15`); results are reported **per configuration (not averaged)**. The
first three:

| Config | T1 | T2 | T3 | T4 |
|--------|----|----|----|----|
| `config_1` | nexus4 | s3 | s3mini | samsungold |
| `config_2` | samsungold | s3mini | s3 | nexus4 |
| `config_3` | nexus4 | s3mini | s3 | samsungold |

#### Preprocessing

Run automatically on first use; cached to `data/hhar/hhar_processed.npz`
(`hhar_preprocess`, [shared/dataset_cl.py](shared/dataset_cl.py)):

1. **Parse** `Phones_accelerometer.csv`, keeping `User, Model, gt, x, y, z` and the 4
   phone models (nexus4, s3, s3mini, samsungold) and 6 activities.
2. **Sliding windows** per `(user, model, activity)`, sorted by time: length **128**,
   stride **128** — **non-overlapping**. Each window → one `(128, 3)` sample.
3. **Subject-independent split:** the 9 users are split **7 train / 2 test**; all of a
   user's windows stay on one side. The user id of each window is saved so the validation
   split holds out whole users **within each device** (task = device, so a task's
   train/val/test all come from the same device).
4. **Cap & class-balance:** windows are capped at **1,500** per
   (user, device, activity), then each device's 6 activities are downsampled (seed 0) to an
   equal count — applied per split, so no window crosses a user boundary and every device
   ends up the same size.
5. **Z-score normalisation** per channel, using **train statistics only**.

---

### Train / validation / tuning splits (all datasets)

Three **disjoint** splits, each specified in a different place — the tuning holdout and
the validation set are **not** the same and serve different purposes:

- **Tuning holdout (~10%, fixed seed `TUNE_SPLIT_SEED = 0`).** The per-dataset
  `TUNE_SUBSET_*` sizes and `TUNE_SPLIT_SEED` live in each setting's `config.py`; the
  `holdout_n` / `holdout_seed` / `use_holdout` flags are set in
  [single_head/run_experiment.py](single_head/run_experiment.py) and applied in
  `ContinualDatasetMixin.active_indices`. A fixed slice of each task's
  training data is reserved as a holdout pool used **only** for the Optuna
  hyperparameter search (`use_holdout=True`); the final training runs use everything
  *outside* the holdout (`use_holdout=False`), so the tuning data never leaks into final
  training.
- **Validation split (`perc_val = 0.25`).** A **separate** split — set in each per-model
  run file (e.g. [naive_lstm_mnist_sh.py](single_head/models/LSTM/naive/naive_lstm_mnist_sh.py))
  and applied by `ContinualDatasetMixin.get_train_val_loader`. It holds out **25% of the
  active training pool** for validation: during tuning that pool is the ~10% holdout
  (so val = 25% of the holdout, and mean val accuracy is the Optuna objective); during
  final runs it is the training data minus the holdout (val = 25% of that, used for
  per-epoch monitoring). It is **dataset-aware**:
  - **MNIST / Fashion-SW** (i.i.d. images): a **random** stratified split
    (these samples are independent, so a random split is unbiased).
  - **WISDM / HHAR** (sensor windows): a **subject-independent** split — whole *users*
    are held out for validation (deterministic), so no person appears in both train and
    val. For HHAR this happens **within each device**, keeping every task tied to its
    own device. This avoids the optimistic bias a random split would cause when
    neighbouring windows are near-identical.
- **Test set.** Fixed and never trained on; used after every task to fill the accuracy
  matrix `R` from which all CL metrics are computed. Image datasets use the official
  torchvision test split (MNIST / Fashion-SW: 60k train / 10k test). WISDM / HHAR use a
  **subject-independent** split — `sensor_train_fraction = 0.8` of users → train, the
  rest → test (WISDM 36 → 28 / 8, HHAR 9 → 7 / 2), so the test users never appear in
  train or val.

#### What the tuning objective optimizes

The Optuna objective is **the mean validation accuracy across *all* tasks** — not task 1
alone. In [run_experiment.py](single_head/run_experiment.py) the objective returns
`np.mean(runner(...)[0])`, where `runner(...)[0]` is `subtask_val_accs`: a list with one
entry per task.

The subtlety is in *what each entry measures*. Each `subtask_val_accs[i]` is task *i*'s
**own** validation accuracy, measured **immediately after** training on task *i* (it is
appended inside the per-task loop, e.g.
[naive_lstm_mnist_sh.py:103](single_head/models/LSTM/naive/naive_lstm_mnist_sh.py#L103)),
and it is **not** re-measured after later tasks. So the averaged objective is a
**plasticity / learning-quality signal** — how well each task is learned at its own time —
and it does **not** reflect retention of earlier tasks (it is not a forgetting-aware
metric like `Acc_final`, `BWT`, or `Acc_avg`, all of which require evaluating earlier
tasks *after* later ones). Hyper-parameters are therefore selected to maximise average
per-task learning, with the anti-forgetting metrics reported (not optimised) afterwards.

---

## Metrics

After every task *i*, each model is evaluated on all tasks seen so far, filling the
**accuracy matrix** `R[i][j]` = test accuracy on task *j* after training through task *i*
(with *T* tasks total). All metrics are computed from `R`
([shared/metrics.py](shared/metrics.py)) and are reported **per task configuration**
(no averaging across configs).

| Metric | Definition | Meaning |
|--------|-----------|---------|
| **Acc_final** | `mean_j R[T-1][j]` | final accuracy across all tasks |
| **Acc_avg** | `mean R[i][j] for all i >= j` | average accuracy over the trajectory |
| **BWT_final** | `mean(R[T-1][j] - R[j][j]) for j < T-1` | net change to earlier tasks by the end (− = forgetting) |
| **BWT_avg** | `mean(R[i][j] - R[j][j]) for all i > j` | average backward transfer over the trajectory |
| **FWT** | `mean(pretrain[j] - chance) for j > 0` | zero-shot head start on an unseen task from prior learning |
| **Cohen's κ** | agreement above chance (multi-class) | accuracy corrected for chance agreement |
| **Plasticity** | `mean_j R[j][j]` (diagonal) | how well each task is learned when first trained |
| **Stability** | `mean R[i][j] for i > j` (off-diagonal) | how well past tasks are retained |

Every accuracy-valued metric above has a **Cohen's κ twin** computed from a parallel
task × task κ matrix (`final_cohen_kappa`, `average_kappa_over_time`,
`final_backward_transfer_kappa`, …), so chance-corrected versions of the whole suite are
available alongside the raw accuracies.

**Chance baseline.** `chance = 1/num_classes` — **0.5** for the binary tasks
(MNIST / Fashion-SW / WISDM) and **1/6** for HHAR's 6 activity classes. It is
both the forward-transfer baseline and the reference line on per-task accuracy plots.

**Forward transfer (FWT).** `pretrain[j]` is the **measured** zero-shot accuracy on task
*j*'s test set using the model trained through task *j*−1 (for SLDA, the shared LDA fitted
on tasks 0…*j*−1). FWT subtracts the hardcoded `chance` baseline — the expected accuracy of
a random classifier, a simplification of GEM's measured random-initialized baseline — and
averages only over *j > 0* (task 0 has no prior knowledge to transfer). The κ version
needs no baseline term: κ = 0 *is* chance agreement, so it reduces to the mean zero-shot κ.

**Cohen's κ is multi-class.** The expected-agreement term sums over **every** class
present, so κ is correct for the 6-way HHAR tasks as well as the binary ones (it reduces
to the usual two-class formula when only labels {0,1} appear).

**Where each metric applies.**
- **Joint** models pool all tasks at once, so there is no task order → only **Acc_final**
  and **Cohen's κ** are meaningful; BWT / FWT / Plasticity / Stability are not reported.
- **FWT is reported for single-head (DIL) only.** In the multi-head (TIL) setting each
  task gets its own freshly-initialized head, so zero-shot accuracy on an unseen task is
  ≈ chance by construction — FWT is not meaningful there and is **not reported**.

---

## Statistical testing

Model comparisons are backed by significance tests rather than by reading means off a
table. Because **every model is run on the identical 15 task configurations**, each
config is a **matched block** — the comparison is a *dependent-samples* (paired) design,
and the between-config variance (some task orders are simply harder for every model)
cancels out.

Run it on any finished experiment directory; nothing is retrained:

```bash
python run_significance.py results/single_head/experiment_20260810_155645
python run_significance.py results/multi_head/experiment_20260801_154816 --alpha 0.01
```

Options: `--metric` (default `final_accuracy`), `--alpha` (default `0.05`), `--out`,
`--no-plot`. The tests currently target **`final_accuracy`** as the single headline
metric; any other key in `per_config[...]["summary"]` can be selected with `--metric`.

### Two complementary views

**1 · Pairwise, directional — "does A outperform B?"**

For each ordered pair the paired differences *d_i* = A_i − B_i over the 15 configs are
tested one-sided:

> H₀: µ_A ≤ µ_B  vs  H₁: µ_A > µ_B, at α = 0.05

using an **exact sign-flip permutation test** on the mean difference. Under H₀ the sign of
each *d_i* is exchangeable, so all 2¹⁵ = 32,768 sign assignments are enumerated in full —
the p-value is exact, with no sampling error and no normality assumption. The mean
difference is the test statistic, keeping the test faithful to hypotheses stated about
means (a signed-rank test would instead target the pseudo-median).

Because both directions of every pair are tested, the family is corrected with
**Holm** within each (head, dataset). Output tables mark `*` for significant uncorrected
and `**` for still significant after correction, so both readings are available.

**2 · Omnibus + post-hoc — the Critical Difference diagram**

The **Friedman test** ranks the models within each config and asks one global question:
are the average ranks more spread out than chance allows? It is the non-parametric
counterpart of a repeated-measures ANOVA and assumes exactly this dependent sampling.
Only if it rejects does the **Nemenyi post-hoc** run, which yields a single **Critical
Difference**: two models differ significantly iff their average ranks differ by more than
CD. This is summarized as a CD diagram, where models joined by a bar are *not*
significantly different.

Ranks discard magnitude — a 30-point win and a 0.1-point win are both "one rank apart" —
so this view is more conservative and less informative than the pairwise tests, but it
needs no multiplicity bookkeeping and condenses into one figure. The two views are
reported together; a pair the pairwise test separates but the CD does not is the expected
conservatism of a rank-based post-hoc, not a contradiction.

### Output

Written to `<exp_dir>/significance/`, one set per dataset:

| File | Contents |
|---|---|
| `<ds>_<metric>_matrix.csv` | the win matrix (`*` / `**` per ordered pair) |
| `<ds>_<metric>_pvalues.csv` | every pair: means, mean difference, wins/15, raw p, Holm p |
| `<ds>_<metric>_table.tex` | booktabs table ready to `\input{}` |
| `<ds>_<metric>_friedman_nemenyi.csv` | χ², df, p, CD, and each model's average rank |
| `<ds>_<metric>_cd_diagram.png` / `.pdf` | the Critical Difference diagram |

Implementation: [shared/stats_tests.py](shared/stats_tests.py) (the tests, numpy-only —
the χ² tail is computed directly rather than adding a scipy dependency) and
[shared/cd_diagram.py](shared/cd_diagram.py) (the figure).

> **Scope of the inference.** `SEED = 42` is fixed, so the 15 replicates vary **task
> ordering only**. The tests therefore generalize over task orderings, not over random
> initializations — worth stating explicitly when reporting them. Covering initialization
> variance too would mean re-running the grid under several seeds.

---

## Single-Head (DIL) Models

In the single-head / Domain-Incremental setting, every model has **one shared output
head** used for all tasks, and **no task ID is given at test time**. Tasks are
presented sequentially; after each task the model is evaluated on all tasks seen so far
to fill the accuracy matrix `R`. Files live under [single_head/models/](single_head/models/).

### LSTM (baseline family)

A plain single-layer LSTM with one shared linear readout
([SingleHeadLSTM](single_head/models/LSTM/model.py), wrapping the GIM repo's
`BaselineRNN`). The sequence's **final hidden state** is fed to the readout. Trained
with Adam + cross-entropy and gradient clipping (max-norm 5.0). Three variants:

| Variant | What it does | Role |
|---------|--------------|------|
| **Naive** ([naive/](single_head/models/LSTM/naive/)) | One LSTM trained sequentially over all tasks, no CL mechanism | Lower bound — shows raw catastrophic forgetting |
| **Replay** ([replay/](single_head/models/LSTM/replay/)) | Naive LSTM + an experience-replay buffer (see below) | Rehearsal baseline |
| **Joint** ([joint/](single_head/models/LSTM/joint/)) | One LSTM trained on **all tasks at once** (`ConcatDataset`) | Oracle upper bound (no task boundaries → no forgetting). Only `Acc_final` and Cohen's κ are meaningful |

### GIM — Gated Incremental Memory

[GIM_LSTM/](single_head/models/GIM_LSTM/) runs the **original GIM repo code unmodified**
(`CL_Experiment`, `train`, `test`, autoencoders). We use the **ALSTM** variant: each
*module* is an LSTM, and the architecture **grows one module per task**.

- **Module growing:** after each task a new module is added (growth threshold `1.01`, so
  growth always fires) — past modules are frozen, so old knowledge is structurally
  protected.
- **Test-time routing (the key idea):** since no task ID is given, GIM trains **one
  autoencoder per task** and routes each test input to the module whose autoencoder gives
  the **lowest reconstruction error**; that module then classifies via `gim_predict`.
- **No joint variant:** the growth criterion depends on sequential task presentation, so
  joint training is undefined for GIM — the LSTM-Joint serves as the shared upper bound.

GIM is the main "method" under study; Naive-LSTM (lower bound) and Joint-LSTM (upper
bound) bracket its performance.

### ESN — Echo State Network (+ CL strategies)

[ESN/](single_head/models/ESN/) wraps the ESN repo's `DeepReservoirClassifier`
([SingleHeadESN](single_head/models/ESN/esn_utils.py)): a **frozen random reservoir**
(units tuned; `spectral_radius=0.99`, `leaky=1.0`, `input_scaling=1.0`, 1 layer) acts as
a fixed feature extractor — **only the readout is trained**. The reservoir hyper-params
are fixed to the paper's values; only training/readout params are tuned. CL strategies
(via the ESN repo's `get_strategy`, Avalanche-backed):

| Strategy | Mechanism |
|----------|-----------|
| **Naive** | Shared readout trained sequentially, no protection (lower bound) |
| **EWC** | Elastic Weight Consolidation: Fisher-weighted quadratic penalty on readout weights (`ewc_mode='separate'`, `ewc_lambda` tuned) |
| **LwF** | Learning without Forgetting: knowledge distillation from the previous model's softened outputs (`lwf_alpha`, `lwf_temperature` tuned) |
| **Replay** | Avalanche experience replay buffer (see below) |
| **SLDA** | StreamingLDA on reservoir features: `ESNWrapper` extracts the final reservoir state, one **shared** LDA (means + covariance, shrinkage `1e-4`) updated incrementally — no gradients, no task ID |
| **Joint** | Readout trained on all tasks at once — oracle upper bound (only `Acc_final`/κ valid) |

> Because the reservoir is frozen, all forgetting in the ESN happens in the small readout
> only, which makes EWC/LwF exact and cheap.

### How Replay differs: LSTM vs ESN

Both store a small memory of past samples and rehearse it, but the implementations differ:

- **LSTM Replay** ([replay_lstm_mnist_sh.py](single_head/models/LSTM/replay/replay_lstm_mnist_sh.py)) — **hand-rolled**. A per-task buffer holds a balanced quota of `mem_size // num_tasks` examples (random subsample) from each past task. When training task *t*, the current task's data is **concatenated** with the entire buffer into one dataset and trained jointly. `mem_size` ≈ 10% of the total training data (`len(first_task) × num_tasks × 0.1`).
- **ESN Replay** ([replay_esn_mnist_sh.py](single_head/models/ESN/replay/replay_esn_mnist_sh.py)) — uses **Avalanche's `ReplayPlugin`** through `get_strategy(strategy='replay')`. The buffer is maintained by **reservoir sampling** and **mixed into every minibatch** during training (rather than concatenated once per task). `mem_size = Σ (10% of each task's training set)`. Replay only updates the readout, since the reservoir is frozen.

So the **memory budget is comparable (~10% of the data)**, but LSTM replay does explicit per-task balanced storage + one-shot concatenation, whereas ESN replay delegates to Avalanche's reservoir-sampled, per-batch rehearsal.

> **Joint training = replay with an unlimited buffer.** The **Joint** variants
> (LSTM-Joint, ESN-Joint) are conceptually the limiting case of replay: instead of
> rehearsing a small ~10% memory of past tasks, they train on **100% of every task's data
> at once**. There is no forgetting because nothing is ever "past" — all tasks are present
> in every batch. This is exactly why Joint is the **oracle upper bound** for the
> rehearsal family: Replay approximates Joint with a tiny, sampled memory, and Joint is
> what Replay would converge to if the buffer held the entire dataset. (Only `Acc_final`
> and Cohen's κ are meaningful for Joint, since BWT/FWT need a task order that joint
> training removes.)

### Model & training differences at a glance

The shared scaffolding is identical across every model: same permuted-MNIST / windowed
sensor input, same per-task loop, same `R`-matrix evaluation, the same tuned holdout
split, and a shared head. What differs is **which parameters update** and **how
the loss/teacher is formed**:

| Model | Trainable parameters | Optimizer / update rule | Loss | Inference routing | Special mechanism |
|-------|----------------------|-------------------------|------|-------------------|-------------------|
| **LSTM Naive/Replay** | Whole LSTM + readout | Adam + CE, grad-clip 5.0 | Cross-entropy (+ replay rehearsal) | Single shared head | — |
| **LSTM Joint** | Whole LSTM + readout | Adam + CE, grad-clip 5.0 | Cross-entropy | Single shared head | All tasks concatenated (no boundaries) |
| **GIM-ALSTM** | Current module's LSTM + readout + that task's autoencoder | Adam + CE, grad-clip 5.0 (`lr_ae=1e-4` for AEs) | Cross-entropy + AE reconstruction | **Autoencoder picks the module** | Grows one frozen module per task |
| **ESN Naive/EWC/LwF/Replay** | **Readout only** (reservoir frozen) | Adam + CE on readout | CE (+EWC Fisher penalty / +LwF distillation / +replay) | Single shared head | Reservoir = fixed feature extractor |
| **ESN SLDA** | **None via gradient** | Closed-form incremental LDA | — (analytic mean/cov updates) | Shared streaming LDA | No backprop at all |
| **ESN Joint** | Readout only | Adam + CE on readout | Cross-entropy | Single shared head | All tasks at once |

Concretely, the training-loop differences are:

- **What learns.** LSTM and GIM update *recurrent weights + readout*; every ESN variant updates *only the readout* (the reservoir is `requires_grad=False`); **SLDA updates no weights by gradient** — it's a closed-form classifier.
- **Sequential vs joint.** All CL variants train task-by-task; the two **Joint** models pool all tasks into one training set, so only `Acc_final` and Cohen's κ are meaningful for them (BWT/FWT are undefined without task order).
- **Where the anti-forgetting pressure lives.** GIM = architectural (new module + AE routing); EWC = parameter penalty; LwF = output distillation; Replay = data rehearsal; SLDA = a single shared statistical model; Naive = nothing.
- **Per-task epochs** are **not** tuned: every other hyper-parameter is tuned with epochs held fixed at 5, and the epoch budget is then studied on its own (see *Epoch-sensitivity analysis* below). GIM-on-MNIST still uses the paper's fixed hyper-parameters.
- **Grad clipping (max-norm 5.0)** is applied wherever gradients flow through time — the LSTM and GIM loops, which train recurrent weights via backprop-through-time (prone to exploding gradients over the 28/200/128-step sequences). ESN skips it because only a linear readout is trained on the frozen reservoir (no BPTT), and SLDA has no gradient step at all.

---

## Multi-Head (TIL) Models

The multi-head setting keeps the same recurrent cores and the same training loop, and
changes exactly one thing: **the output layer is a `ModuleDict` of per-task linear heads
instead of one shared head**, and the **oracle task ID is supplied at test time** to route
through the right one. Before each task the runner calls `add_head(task_id)` then
`set_task(task_id)`.

| Wrapper | Base | What is trained |
|---|---|---|
| [MultiHeadLSTM](multi_head/models/LSTM/multi_head_lstm_model.py) | GIM repo's `BaselineRNN` (LSTM) | recurrent weights + the active task's head |
| [MultiHeadESN](multi_head/models/ESN/multi_head_esn_model.py) | ESN repo's `DeepReservoirClassifier` | the active task's head only (reservoir frozen) |

Six models are run ([multi_head/config.py](multi_head/config.py) `MODELS`): GIM-ALSTM,
LSTM (naive), LSTM-Joint, LSTM-Replay, ESN-Naive, ESN-Joint. **The ESN CL strategies
(EWC / LwF / Replay / SLDA) are not run here** — with a private head per task there is far
less for a readout-level penalty to protect, so the multi-head roster keeps only the
Naive lower bound and the Joint upper bound for the ESN family.

Because each task's head is freshly initialized and never sees another task's data, a
task's zero-shot accuracy before training is ≈ chance by construction, which is why
**FWT is not reported in this setting**.

---

## Epoch-sensitivity analysis (`run_epoch_analysis.py`)

Per-task **epochs are deliberately not tuned**. `run_tuning.py` searches every *other*
hyper-parameter with the epoch budget pinned at 5, and the main comparison
(`run_experiment.py`) trains at that same fixed budget (`EPOCHS = 5` in `single_head/`,
`REFERENCE_EPOCHS = 5` in `multi_head/`). Training length is then studied **on its own**
here, so the epoch question neither multiplies the cost of the headline grid nor
entangles itself with the hyper-parameter search.

**What it does.** It reads the *same* `tuned_params.json` as the main comparison (no
re-tuning) and re-runs the full `model × dataset` grid at each budget in
`EPOCH_NUMBERS = [5, 10, 20]`, on the **first `CONFIGS` task configurations**
of each dataset (`CONFIGS = 3` in both settings). Holding the budget as the *only* thing
that varies — same hyper-parameters, same configs, same `SEED`, full training set —
isolates the effect of training length. Restricting to the first few configs keeps the
trend lines internally consistent and keeps cost down (3 budgets × 3 configs instead of
3 budgets × all 15).

**Outputs**, under `epoch_analysis_<ts>/`:
- `epoch_sweep/epochs_<E>/` — one standard result set (`summary.json` + `plots/`) per
  budget `E`, produced by the *same* `run_combination` / `save_plots` code the main
  comparison uses, so each budget is directly comparable to the headline run.
- `epoch_analysis/` — the cross-budget charts: **every CL metric vs. epochs, one line per
  model, per dataset** (`save_epoch_analysis_plots`, [shared/epoch_analysis.py](shared/epoch_analysis.py)).

**What to read from it.** The headline story is the **plasticity–stability trade-off**: as
the epoch budget grows, **plasticity** (how well each task is learned at its own time — the
`R` diagonal) tends to *rise*, while **stability / backward transfer** (retention of
earlier tasks) tends to *degrade*, because more fitting of the current task overwrites more
of the past. Plotting each metric against epochs makes that curve explicit per model, so
you can see which strategies trade the two off gracefully (replay, EWC, GIM) versus the
Naive lower bound, which simply forgets faster the longer it trains.

> **A flat line is a result, not a bug.** ESN-SLDA (single-head only) is a closed-form,
> single-pass classifier with no gradient training, so the epoch budget cannot affect it —
> its flat curve is the intended contrast (insensitive to training length). The other
> frozen-reservoir ESN variants are likewise far less epoch-sensitive than the fully
> trained LSTM and GIM, since only their small readout sees the extra epochs.

---

## Reproducibility

The pipeline is built to be **fully deterministic**: re-running
[single_head/run_experiment.py](single_head/run_experiment.py) on the same machine
reproduces the same hyper-parameters *and* the same headline numbers, bit-for-bit.

**CPU-only math.** `DEVICE = torch.device("cpu")` on purpose — it avoids the
non-determinism of cuDNN/cuBLAS and atomic GPU reductions. Because the whole pipeline runs
**serially** (see below), each run is free to use all CPU cores for its own intra-op math
without any cross-run contention reordering floating-point reductions.

**One global seed, reseeded per run.** `SEED = 42`. Every experiment
run reseeds both RNGs (`torch.manual_seed(SEED)`, `np.random.seed(SEED)`) **immediately
before** the model is built and trained, so weight init, batch shuffling, and reservoir
sampling are identical regardless of which combination ran first.

**Serial execution, not pools.** Both phases run **one unit of work at a time** — one
Optuna trial in tuning, one `(dataset, model)` combination in the experiments — with no
process or thread pool. The runner models read the *global* NumPy/PyTorch RNG (they don't
take a private generator), so any concurrency would let trials/combinations overwrite each
other's seed and make results depend on scheduling. Running serially and reseeding before
each unit of work is exactly what keeps both the *selected* hyper-parameters and the
headline numbers reproducible.

**Seeded tuning.** Every Optuna trial reseeds the global RNG to `SEED` before evaluating,
the sampler is itself seeded (`TPESampler(seed=SEED)`), and `MedianPruner` is fair under
serial execution because it compares each trial only against previously *completed* trials.
Because the chosen hyper-parameters are cached and feed both experiment scripts, this is
what makes the whole pipeline reproducible end-to-end.

**Deterministic data splits.** The three disjoint splits (see *Train / validation / tuning
splits*) are all fixed:
- The **tuning holdout** uses a fixed `TUNE_SPLIT_SEED = 0` and fixed per-dataset sizes
  (`TUNE_SUBSET_*`), so the held-out tuning pool is identical every run and never leaks
  into final training.
- The **validation split** is deterministic per dataset — a seeded stratified split for
  the i.i.d. image datasets, and a fixed **subject-independent** (whole-user) hold-out for
  the sensor datasets (WISDM / HHAR).
- The **test set** is fixed (official torchvision split for images; subject-independent
  user split for sensors) and never trained on.

**Per-config, never averaged.** Each of the 15 task configurations is run once under the
fixed seed and reported on its own — there is no averaging across configs or across seeds
to obscure run-to-run variation. This is also what makes the paired statistical tests
possible: the configs are matched blocks shared by every model.

**Full datasets.** The `SUBSET_*` constants in both `config.py` files are `None`, so the
main comparison trains on the **complete** per-task training data (no per-task cap). Each
`summary.json` still records `seed`, `epochs`, and the per-dataset `subset` /
`tune_subset` values for whatever `--datasets` were selected, so a run stays fully
described by its own metadata.

> **Scope of guarantee.** Determinism holds on a fixed machine + library stack
> (PyTorch / NumPy / Optuna versions). Different CPUs or BLAS builds can still differ in
> the last floating-point digits; the seeding and split logic above remove every
> *controllable* source of variation.

---

## Execution model (three scripts, sequential)

The pipeline is **three scripts sharing one `config.py`** and a cached hyper-parameter
file — there is **no process or thread pool**; every unit of work runs one at a time. The
reason is determinism: the runner models read the *global* NumPy/PyTorch RNG (they don't
take a private generator), so any concurrency would let trials or combinations overwrite
each other's seed and make results depend on scheduling. To keep a serial run fast, each
run is left free to use all CPU cores for its own intra-op math.

### `run_tuning.py` — tuning (writes the cache)

Each `(model, dataset)` is an independent Optuna study (`TPESampler(seed=SEED)`,
`MedianPruner`), optimized over `N_TRIALS = 15` trials at a 5-epoch budget on a small
tuning holdout. Every trial reseeds the global RNG to `SEED` before evaluating, so which
trial wins is fixed run-to-run, and `MedianPruner` cuts weak trials short (fair under
serial execution — it only compares against *completed* trials). The best params are
written to the seed-tagged `tuned_params.json`, **tuned incrementally** (only datasets not
already in the cache), and reused by both experiment scripts.

- **single_head** tunes in two ordered passes: **1a** the independent studies (GIM, LSTM,
  Joint-LSTM, ESN-Base, Joint-ESN), then **1b** the ESN strategy studies (EWC, LwF), each
  of which *merges in* the params of its corresponding `ESN-Base on <dataset>` study and so
  must run **after** 1a.
- **multi_head** has only Naive + Joint ESN (no EWC/LwF), so there is no 1a/1b ordering —
  every study is independent.

### `run_experiment.py` — main comparison (reads the cache)

The full `dataset × model` grid runs **sequentially**, one combination at a time, at the
fixed 5-epoch budget, across **all 15** task configs. Each `run_combination` reseeds both
RNGs immediately before building/training the model, so its result is independent of run
order. Results land in `results/<head>/experiment_<ts>/` as one JSON per
`(model, dataset)` — each holding the hyper-parameters plus a `per_config` block with the
val/test curves, the `R` matrix and the full metric summary for all 15 configs — alongside
`summary.json`, `plots/`, and printed metric + hyper-parameter tables.

### `run_epoch_analysis.py` — epoch sweep (reads the cache)

The same grid re-run across `EPOCH_NUMBERS = [5, 10, 20]`, but only on the first
`CONFIGS = 3` configs of each dataset, so the epoch-trend study stays cheap. Each
`(dataset, model, epochs)` triple is one sequential run; results and the cross-budget
charts are described in the
[Epoch-sensitivity analysis](#epoch-sensitivity-analysis-run_epoch_analysispy) section above.

---

## References

**Datasets**
- **MNIST** — Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE, 1998.
- **WISDM** — J. R. Kwapisz, G. M. Weiss, S. A. Moore. *Activity Recognition using Cell Phone Accelerometers.* SIGKDD Explorations, 2011. ([WISDM Lab](https://www.cis.fordham.edu/wisdm/dataset.php))
- **HHAR** — A. Stisen et al. *Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition.* SenSys, 2015. ([UCI dataset 344](https://archive.ics.uci.edu/dataset/344))
- **Fashion-MNIST** — H. Xiao, K. Rasul, R. Vollgraf. *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747, 2017.

**GIM — Gated Incremental Memory**
- A. Cossu, A. Carta, D. Bacciu. *Continual Learning with Gated Incremental Memories for Sequential Data Processing.* IJCNN, 2020. ([arXiv:2004.04077](https://arxiv.org/abs/2004.04077)) — the architecture re-implemented here (one grown module per task + autoencoder routing).

**ESN — Echo State Network**
- *Foundations:* H. Jaeger. *The "Echo State" Approach to Analysing and Training Recurrent Neural Networks.* GMD Report 148, German National Research Center for Information Technology, 2001. · H. Jaeger, H. Haas. *Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication.* Science 304(5667):78–80, 2004.
- *Deep reservoir (the architecture used here — `clrnn/deep_esn.py` is a PyTorch port of Gallicchio's [DeepRC-TF](https://github.com/gallicch/DeepRC-TF), and the file asks for exactly this citation):* C. Gallicchio, A. Micheli, L. Pedrelli. *Deep Reservoir Computing: A Critical Experimental Analysis.* Neurocomputing 268:87–99, 2017. ([doi:10.1016/j.neucom.2016.12.089](https://doi.org/10.1016/j.neucom.2016.12.089))
- *ESN for continual learning (source of the CL strategies + configs):* A. Cossu, D. Bacciu, A. Carta, C. Gallicchio, V. Lomonaco. *Continual Learning with Echo State Networks.* ESANN, 2021.

**Statistical methodology**
- J. Demšar. *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR 7:1–30, 2006. — the Friedman + Nemenyi protocol and the Critical Difference diagram; source of the tabulated `q_α` critical values in [shared/stats_tests.py](shared/stats_tests.py).
- S. Holm. *A Simple Sequentially Rejective Multiple Test Procedure.* Scandinavian Journal of Statistics 6(2):65–70, 1979. — the step-down family-wise correction applied to the pairwise tests.

**Preprocessing reference**
- **WISDM windowing (200-step windows)** — R. R. Azghan et al. *Gated Adaptation for Continual Learning in Human Activity Recognition.* [arXiv:2603.10046](https://arxiv.org/abs/2603.10046). (We use **non-overlapping** windows here, not the 50%-overlap variant — see the Inputs section below.)

---

## Inputs: what one sample is and how it is fed to the model

Every model here is **recurrent** (LSTM, ESN, GIM). It does not see a whole sample at
once — it reads it **one time step at a time**, updating a hidden state, and classifies
from the state at the end. So every input has the shape:

```
(batch, time_steps, features_per_step)
```

What "one time step" and "one sample" mean differs by dataset:

| Dataset | One sample = (time steps × features) | One time step is… | `input_size` | Real time? |
|---------|--------------------------------------|-------------------|:------------:|:----------:|
| **MNIST / Fashion-SW** | 28 × 28 | one **row of 28 pixels** of the image | 28 | ❌ no |
| **HHAR** | 128 × 3 | one accelerometer reading **(x, y, z)** | 3 | ✅ yes |
| **WISDM** | 200 × 3 | one accelerometer reading **(x, y, z)** | 3 | ✅ yes |

### Images (MNIST / Fashion-SW)
A 28×28 image is read **row by row**: 28 steps, each a 28-pixel row
([`collect_datasets` / `MNIST_PERM`](shared/utils.py)). For MNIST the 784 pixels are
first permuted with the fixed GIM permutation, then reshaped back to 28×28 so the
sequence format is unchanged. **Important:** these "time steps" are just the rows of one
still image — they are *not* real time, and different images are independent of each
other. That is why a random train/val split is fine for images.

### Sensors (HHAR / WISDM)
A long continuous accelerometer recording is cut into fixed-length **windows**, and
**one window = one sample** fed to the model:
- **HHAR:** 128 readings per window, **non-overlapping** (`stride = window = 128`).
- **WISDM:** 200 readings per window, **non-overlapping** (`stride = window = 200`).

Each window carries **one label** (the activity during it). Here the time steps are
**real time** (WISDM is 20 Hz, so a 200-step window is ~10 s; HHAR's rate varies by
device), so neighbouring windows are highly correlated — which is why these datasets use
**subject-independent** train/val/test splits, never a random split of windows.

### How the model consumes a sample
1. The sequence `(time_steps, features)` is fed in step by step.
2. The recurrent core (LSTM / frozen ESN reservoir / GIM module) updates its hidden state at each step.
3. The **final hidden state** is passed to the classifier head — a linear readout (LSTM / ESN / GIM), one of the per-task heads (multi-head), or the streaming LDA (SLDA) — which outputs the class for that whole sample. (`SingleHeadLSTM` / `SingleHeadESN` use `states[-1]` / the last output.)

### Splits recap (per dataset)
- **MNIST / Fashion-SW** — i.i.d. images: official train/test split; **random** stratified validation.
- **HHAR** — task = device; **subject-independent**: 9 users → 7-user train pool / 2 test, and within each device-task users 0–5 train / user 6 validation / 7–8 test (`hhar_n_val_users = 1`; same device per task, no user shared across splits).
- **WISDM** — task = activity-pair; **subject-independent**: 36 users → 28 train / 8 test, with the later train-users held out for validation.

In all cases the chance baseline used by metrics is `1/num_classes` — **0.5** for the
binary tasks (MNIST / Fashion-SW / WISDM) and **1/6** for HHAR's 6 activities.
