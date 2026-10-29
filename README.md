# Continual Learning with Recurrent Models: Thesis Experiments

This repository contains the experiment code for a thesis on **continual learning (CL)**
with recurrent models. It compares CL strategies for three model families: **LSTM**,
**GIM** (Gated Incremental Memory) and **ESN** (Echo State Networks), on four sequence
datasets and in two settings:

| Setting | Folder | Output layer | Task id at test time | Datasets | Models |
|---|---|---|---|---|---|
| Domain-incremental (DIL) | [single_head/](single_head/) | one shared head | not given | MNIST, Fashion-SW, HHAR | 10 |
| Task-incremental (TIL) | [multi_head/](multi_head/) | one head per task | given | MNIST, Fashion-SW, HHAR, WISDM | 6 |

Each dataset is split into a sequence of **tasks**. Every model is run on **15 task
configurations** (different task orders and pairings), and results are reported for each
configuration separately.

## Contents

1. [Requirements](#1-requirements)
2. [Download the code](#2-download-the-code)
3. [Set up the environment](#3-set-up-the-environment)
4. [Download the datasets](#4-download-the-datasets)
5. [Run the experiments](#5-run-the-experiments)
6. [Output files](#6-output-files)
7. [Project layout](#7-project-layout)
8. [Experimental setup](#8-experimental-setup)
9. [References](#9-references)

---

## 1. Requirements

- **Python 3.11**
- **CPU only.** No GPU is needed; the code always runs on the CPU (see
  [Reproducibility](#reproducibility)).
- About **2 GB of free disk space** for the datasets.

Package versions used for the thesis results:

| Package | Version |
|---|---|
| torch | 2.5.1 |
| torchvision | 0.20.1 |
| numpy | 2.4.3 |
| scikit-learn | 1.7.1 |
| pandas | 2.3.3 |
| matplotlib | 3.10.6 |
| optuna | 4.8.0 |
| avalanche-lib | 0.6.0 |
| pyyaml | 6.0.2 |
| tensorboard | 2.20.0 |

`pyyaml` and `tensorboard` are needed by the bundled ESN and GIM code in `repos/`.

---

## 2. Download the code

With git:

```bash
git clone https://github.com/silaSaraoglu/thesis-cl-experiment.git
cd thesis-cl-experiment
```

Without git: on the GitHub page click **Code → Download ZIP**, unzip it and open a
terminal in the unzipped folder.

The original GIM and ESN code is already included under `repos/`, so nothing else has
to be cloned.

---

## 3. Set up the environment

Create a Python 3.11 environment with **one** of the two options below.

**Option A: conda**

```bash
conda create -n thesis-cl python=3.11
conda activate thesis-cl
```

**Option B: venv**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Then install the packages:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.4.3 scikit-learn==1.7.1 pandas==2.3.3 matplotlib==3.10.6 optuna==4.8.0 avalanche-lib==0.6.0 pyyaml==6.0.2 tensorboard==2.20.0
```

The first line installs the CPU build of PyTorch.

---

## 4. Download the datasets

The `data/` folder is **not** included in the repository. The scripts expect it at the
repository root with this layout (the paths are set in each setting's `config.py`):

```
data/
├── mnist/MNIST_CL/raw/                 MNIST files
├── fashion_mnist/FashionMNIST_CL/raw/  Fashion-MNIST files
├── hhar/Phones_accelerometer.csv       HHAR raw file
└── wisdm/WISDM_ar_v1.1_raw.txt         WISDM raw file (multi-head only)
```

### MNIST and Fashion-MNIST

Run this from the repository root. It downloads both datasets into the folders above:

```bash
python -c "from shared.dataset_cl import MNIST_CL, FashionMNISTSeason_CL; [C(r, download=True, train=t) for C, r in [(MNIST_CL, 'data/mnist'), (FashionMNISTSeason_CL, 'data/fashion_mnist')] for t in (True, False)]"
```

### HHAR

1. Download the **Heterogeneity Activity Recognition** dataset from UCI:
   <https://archive.ics.uci.edu/dataset/344>
2. Extract `Phones_accelerometer.csv` (about 1.3 GB) into `data/hhar/`.

### WISDM (only used by `multi_head/`)

1. Download the **WISDM Activity Prediction** dataset (v1.1):
   <https://www.cis.fordham.edu/wisdm/dataset.php>
2. Extract `WISDM_ar_v1.1_raw.txt` into `data/wisdm/`.

### Preprocessing

HHAR and WISDM are preprocessed automatically the first time they are used (windowing,
subject-independent split, class balancing, z-score normalisation). The result is
cached as `data/hhar/hhar_processed.npz` and `data/wisdm/wisdm_processed.npz`. Delete
these files to rebuild them.

---

## 5. Run the experiments

Run all commands from the **repository root**. Both settings have the same scripts, so
replace `single_head` with `multi_head` to run the task-incremental setting.

| Step | Script | What it does | Writes |
|:---:|---|---|---|
| 1 | `run_tuning.py` | Optuna hyperparameter search | `<setting>/tuned_params.json` |
| 2 | `run_experiment.py` | Main comparison: every model × dataset × 15 configs, 5 epochs | `results/<setting>/experiment_<timestamp>/` |
| 3 | `run_epoch_analysis.py` | Same grid at 5, 10 and 20 epochs (first 3 configs) | `results/<setting>/epoch_analysis_<timestamp>/` |
| 4 | `run_significance.py` (repository root) | Statistical tests on a finished experiment | `<experiment folder>/significance/` |

### Step 1: Tune the hyperparameters (optional)

The repository already contains the tuned hyperparameters used in the thesis
(`single_head/tuned_params.json` and `multi_head/tuned_params.json`, found with
`SEED = 42`). **You can skip this step** to reproduce the thesis results with the same
hyperparameters.

To tune again:

```bash
python single_head/run_tuning.py
python multi_head/run_tuning.py
```

- Only datasets that are missing from `tuned_params.json` are tuned; datasets already
  in the file are kept. To tune everything from scratch, delete `tuned_params.json`
  first. A file created with a different `SEED` is ignored.
- Each (model, dataset) pair gets its own Optuna study: 15 trials, 5 epochs per task,
  on a small tuning holdout (see [Data splits](#data-splits)).
- Steps 2 and 3 only **read** this file. If a (model, dataset) pair is missing from it,
  they stop with a `KeyError`. Run step 1 for that dataset first.

### Step 2: Run the main experiment

```bash
python single_head/run_experiment.py
python multi_head/run_experiment.py
```

Every model is trained on every dataset and all 15 task configurations, with 5 epochs
per task. At the end the script prints the metric tables and the hyperparameters, and
the path of the results folder.

### Step 3: Run the epoch analysis

```bash
python single_head/run_epoch_analysis.py
python multi_head/run_epoch_analysis.py
```

The same models are trained with 5, 10 and 20 epochs per task on the first 3 task
configurations of each dataset. The script then plots each metric against the number
of epochs.

### Step 4: Run the statistical tests

Use the results folder printed by step 2:

```bash
python run_significance.py results/single_head/experiment_<timestamp>
python run_significance.py results/multi_head/experiment_<timestamp>
```

Nothing is retrained; the script only reads the saved results.

| Option | Meaning | Default |
|---|---|---|
| `--metric` | Metric to compare (any key in a config's `summary`) | `final_accuracy` |
| `--alpha` | Significance level | `0.05` |
| `--out` | Output folder | `<experiment folder>/significance` |
| `--no-plot` | Do not draw the Critical Difference diagram | off |

### Run only some datasets

The scripts in `single_head/` and `multi_head/` accept `--datasets`:

```bash
python single_head/run_tuning.py --datasets hhar
python single_head/run_experiment.py --datasets mnist fashion_sw
python multi_head/run_experiment.py --datasets wisdm
```

| Setting | Allowed values |
|---|---|
| `single_head/` | `mnist`, `hhar`, `fashion_sw` |
| `multi_head/` | `mnist`, `wisdm`, `hhar`, `fashion_sw` |

### Running time

Everything runs on the CPU, one run at a time. A full run of every model, dataset and
configuration can take many hours. To split the work, run one dataset at a time with
`--datasets`; each call creates its own results folder.

---

## 6. Output files

The `results/` folder is created when you run the scripts (it is not in the repository).

```
results/<setting>/
├── experiment_<timestamp>/
│   ├── <model>_<dataset>.json          hyperparameters + results of every config
│   │                                   (R and K matrices, all metrics)
│   ├── summary.json                    all results of this run
│   ├── plots/<dataset>/<config>/       metric bars, per-task bars, R/K heatmaps
│   ├── plots/<dataset>/comparison/     one figure per metric, all configs side by side
│   ├── plots/<dataset>/aggregate/      mean ± std over all configs
│   └── significance/                   written by run_significance.py
└── epoch_analysis_<timestamp>/
    ├── epoch_sweep/epochs_<E>/         a full result set for each epoch budget E
    └── epoch_analysis/                 metric-vs-epochs charts per dataset
```

Files written by `run_significance.py`, one set per dataset:

| File | Contents |
|---|---|
| `<dataset>_<metric>_pvalues.csv` | every model pair: means, mean difference, wins, raw and Holm p-values |
| `<dataset>_<metric>_matrix.csv` | win matrix (`*` significant, `**` still significant after Holm) |
| `<dataset>_<metric>_table.tex` | LaTeX table of the win matrix |
| `<dataset>_<metric>_friedman_nemenyi.csv` | Friedman statistic, p-value, Critical Difference, average ranks |
| `<dataset>_<metric>_cd_diagram.png` / `.pdf` | Critical Difference diagram (only when the Friedman test is significant) |

---

## 7. Project layout

```
thesis-cl-experiment/
├── shared/                     code used by both settings
│   ├── dataset_cl.py           dataset classes + HHAR/WISDM preprocessing
│   ├── utils.py                pixel permutation, data collection, tuning cache
│   ├── replay_buffer.py        replay memory for the LSTM replay models
│   ├── metrics.py              CLMetrics: R/K matrices and all CL metrics
│   ├── stats_tests.py          permutation test, Holm, Friedman, Nemenyi
│   ├── cd_diagram.py           Critical Difference diagram
│   ├── epoch_analysis.py       metric-vs-epochs charts
│   └── plot_style.py           figure sizes and fonts for the thesis
│
├── single_head/                domain-incremental setting (one shared head)
│   ├── config.py               datasets, models, seed, 15 task configs, paths
│   ├── tuned_params.json       tuned hyperparameters (output of run_tuning.py)
│   ├── run_tuning.py
│   ├── run_experiment.py
│   ├── run_epoch_analysis.py
│   └── models/                 GIM_LSTM/, LSTM/{naive,joint,replay}/,
│                               ESN/{naive,ewc,lwf,replay,slda,joint}/
│
├── multi_head/                 task-incremental setting (one head per task)
│   ├── config.py, tuned_params.json, run_tuning.py,
│   │   run_experiment.py, run_epoch_analysis.py
│   └── models/                 GIM_LSTM/, LSTM/{naive,joint,replay}/, ESN/{naive,joint}/
│
├── run_significance.py         statistical tests on finished results
├── repos/                      original GIM and ESN code (lightly adapted)
├── data/                       datasets (download separately, see section 4)
└── results/                    experiment output (created when you run)
```

Each file in `models/` runs one model on one dataset, for example
`single_head/models/ESN/ewc/ewc_esn_hhar_sh.py` is ESN + EWC on HHAR, single-head.

---

## 8. Experimental setup

### Datasets

| Dataset | Type | Tasks | Classes per task | Input (time steps × features) | Train / test split |
|---|---|:---:|:---:|:---:|---|
| **MNIST** | handwritten digits | 5 (digit pairs) | 2 | 28 × 28 | official split |
| **Fashion-SW** | Fashion-MNIST, summer vs winter | 3 | 2 | 28 × 28 | official split |
| **HHAR** | phone accelerometer | 4 (phone models) | 6 activities | 128 × 3 | by user |
| **WISDM** | phone accelerometer | 3 (activity pairs) | 2 | 200 × 3 | by user |

- **MNIST:** each task is a pair of digits, for example 0 vs 1. Labels are mapped to
  even digit → 0 and odd digit → 1. Every image is shuffled with one fixed pixel
  permutation from the original GIM paper and then read row by row (28 steps of 28
  pixels).
- **Fashion-SW:** each task pairs one summer class (T-shirt, Dress, Sandal → 0) with one
  winter class (Pullover, Coat, Ankle boot → 1). Same pixel permutation and row-by-row
  input as MNIST. Every task has 12,000 training and 2,000 test images.
- **HHAR:** each task is one phone model (nexus4, s3, s3mini, samsungold) with the same
  6 activities (bike, sit, stairs down, stairs up, stand, walk). Non-overlapping windows
  of 128 readings (x, y, z), capped at 1,500 windows per (user, phone, activity) and
  balanced so every phone and activity has the same number of windows (6,822 train /
  2,232 test windows per phone).
- **WISDM:** 6 activities paired into 3 binary tasks (default: Walking/Jogging,
  Upstairs/Downstairs, Sitting/Standing). Non-overlapping windows of 200 readings
  (10 s at 20 Hz), balanced to 318 train / 144 test windows per task.
- HHAR and WISDM are normalised with the mean and standard deviation of the training
  users only.

### Task configurations

Each dataset has 15 task configurations, `config_1` to `config_15`, defined in the
`*_DATASET_CONFIGS` dictionaries of each `config.py`. They change which classes are
paired and the order of the tasks. Example (`config_1`):

| Dataset | Tasks in order |
|---|---|
| MNIST | 0/1, 2/3, 4/5, 6/7, 8/9 |
| Fashion-SW | T-shirt/Pullover, Dress/Coat, Sandal/Ankle boot |
| HHAR | nexus4, s3, s3mini, samsungold |
| WISDM | Walking/Jogging, Upstairs/Downstairs, Sitting/Standing |

### Data splits

Three separate splits are used:

1. **Tuning holdout.** A fixed part of each task's training data (seed 0) is used
   only for hyperparameter tuning. Size per task: MNIST 1,200, Fashion-SW 1,200,
   HHAR 675, WISDM 62 (`TUNE_SUBSET_*` in `config.py`). The final runs train on the
   rest of the data, so tuning data never appears in final training.
2. **Validation.** 25% of the active training data. For images it is a random
   stratified split. For HHAR and WISDM whole users are held out, so no person is in
   both train and validation.
3. **Test.** Never used for training. Images use the official test set. For HHAR and
   WISDM, 80% of the users form the training pool and the other 20% are test users
   (HHAR: 9 users → 7 / 2, WISDM: 36 users → 28 / 8).

**Tuning objective:** the mean validation accuracy over all tasks, where each task's
accuracy is measured right after training on it. It rewards learning each task well;
forgetting is measured afterwards with the test metrics below.

### Models

**Single-head (DIL), 10 models:**

| Model | ID | Description |
|---|---|---|
| LSTM | `lstm_sh` | Single-layer LSTM trained on the tasks one after another. Lower bound. |
| LSTM-Replay | `lstm_replay_sh` | LSTM with a replay memory of 10% of the training data, shared equally by the tasks seen so far. Each mini-batch mixes new samples with samples from memory. |
| Joint-LSTM | `joint_lstm_sh` | LSTM trained on all tasks at once. Upper bound. |
| GIM-ALSTM | `gim_alstm_sh` | Adds a new LSTM module per task and freezes the old ones. An autoencoder per task sends each test input to the right module, since no task id is given. |
| ESN-Naive | `esn_naive_sh` | Frozen random reservoir; only the readout is trained, with no protection against forgetting. |
| ESN-EWC | `esn_ewc_sh` | ESN + Elastic Weight Consolidation (penalty on changing important readout weights). |
| ESN-LwF | `esn_lwf_sh` | ESN + Learning without Forgetting (distillation from the previous model). |
| ESN-Replay | `esn_replay_sh` | ESN + Avalanche replay with a memory of 10% of the training data. |
| ESN-SLDA | `esn_slda_sh` | Streaming LDA on the reservoir states; no gradient training. |
| Joint-ESN | `joint_esn_sh` | ESN readout trained on all tasks at once. Upper bound. |

**Multi-head (TIL), 6 models:** GIM-ALSTM, LSTM, Joint-LSTM, LSTM-Replay, ESN-Naive
and Joint-ESN. Each task gets its own linear head and the true task id picks the head
at test time. Differences from single-head:

- GIM uses the task id to pick the module, so no autoencoders are trained.
- In LSTM-Replay every stored sample keeps its task id and is trained through its own
  task's head.
- The ESN strategies EWC, LwF, Replay and SLDA are not run: with a frozen reservoir and
  a separate head per task, ESN-Naive already does not forget.

**Shared training details:**

- LSTM and GIM: Adam, cross-entropy loss, gradient clipping at 5.0.
- ESN reservoir: 1 layer, spectral radius 0.99, leaky rate 1.0, input scaling 1.0.
  The number of units is tuned.
- 5 epochs per task in the main experiment. The number of epochs is not tuned; it is
  studied separately in the epoch analysis.
- GIM on MNIST uses the fixed hyperparameters of the original GIM paper; everywhere
  else the hyperparameters are tuned.

### Metrics

After training on each task *i*, the model is tested on every task *j* seen so far.
This fills the accuracy matrix `R[i][j]` (and a Cohen's kappa matrix `K[i][j]`).
All metrics are computed from these matrices ([shared/metrics.py](shared/metrics.py)),
with *T* tasks in total:

| Metric | Definition | Meaning |
|---|---|---|
| Acc_final | `mean_j R[T-1][j]` | accuracy on all tasks at the end |
| Acc_avg | `mean R[i][j]` for all `i >= j` | average accuracy over the whole sequence |
| BWT_final | `mean(R[T-1][j] - R[j][j])` for `j < T-1` | change on earlier tasks by the end (negative = forgetting) |
| BWT_avg | `mean(R[i][j] - R[j][j])` for all `i > j` | average backward transfer over the sequence |
| FWT | `mean(pretrain[j] - chance)` for `j > 0` | accuracy on a task before training on it, above chance |
| Cohen's kappa | agreement above chance | accuracy corrected for chance |
| Plasticity | `mean_j R[j][j]` | how well each task is learned when it is trained |
| Stability | `mean R[i][j]` for `i > j` | how well earlier tasks are kept |

- Every accuracy metric also has a kappa version computed from `K`.
- Chance is 0.5 for the binary tasks and 1/6 for HHAR.
- **Joint models** have no task order, so only Acc_final and Cohen's kappa are
  reported for them.
- **FWT is reported for single-head only.** In multi-head a new task's head is
  untrained, so its accuracy before training is at chance by design.

### Statistical testing

Every model is run on the same 15 task configurations, so the configurations are
matched pairs. [run_significance.py](run_significance.py) uses two tests
([shared/stats_tests.py](shared/stats_tests.py)):

1. **Pairwise test: "is model A better than model B?"** A one-sided exact paired
   permutation test on the 15 differences A − B (H₀: µ_A ≤ µ_B, H₁: µ_A > µ_B,
   α = 0.05). All 2¹⁵ sign flips are enumerated, so the p-value is exact. The p-values
   are corrected for multiple comparisons with the **Holm** method.
2. **Friedman test + Nemenyi post-hoc test.** The Friedman test ranks all models in
   each configuration and checks whether the average ranks differ more than chance. If
   it does, the Nemenyi test gives a Critical Difference (CD). In the CD diagram, models
   joined by a bar are not significantly different.

The main metric is `final_accuracy`. Because `SEED = 42` is fixed, the 15 replicates
differ only in task order, so the tests say something about task orders, not about
random initialisations.

### Reproducibility

- **CPU only** (`DEVICE = torch.device("cpu")`), which avoids non-deterministic GPU
  operations.
- **One seed, `SEED = 42`.** PyTorch and NumPy are reseeded right before every
  (model, dataset, configuration) run, so the results do not depend on run order.
- **Serial runs.** One run at a time, with no parallel workers, because the models use
  the global random number generators.
- **Seeded tuning.** The Optuna sampler is seeded (`TPESampler(seed=SEED)`).
  `single_head/run_tuning.py` also reseeds PyTorch and NumPy before every trial;
  `multi_head/run_tuning.py` does not.
- **Fixed data splits.** The tuning holdout uses seed 0, the sensor validation users
  are fixed, and the test sets never change.
- The same machine and package versions give the same numbers. Other CPUs or BLAS
  libraries can differ in the last decimal places.

---

## 9. References

**Original code**

- GIM: [AndreaCossu/ContinualLearning-SequentialProcessing](https://github.com/AndreaCossu/ContinualLearning-SequentialProcessing)
- ESN: [Pervasive-AI-Lab/ContinualLearning-EchoStateNetworks](https://github.com/Pervasive-AI-Lab/ContinualLearning-EchoStateNetworks)

**Models**

- A. Cossu, A. Carta, D. Bacciu. *Continual Learning with Gated Incremental Memories for
  Sequential Data Processing.* IJCNN, 2020. ([arXiv:2004.04077](https://arxiv.org/abs/2004.04077))
- A. Cossu, D. Bacciu, A. Carta, C. Gallicchio, V. Lomonaco. *Continual Learning with
  Echo State Networks.* ESANN, 2021. ([arXiv:2105.07674](https://arxiv.org/abs/2105.07674))
- H. Jaeger. *The "Echo State" Approach to Analysing and Training Recurrent Neural
  Networks.* GMD Report 148, 2001.
- H. Jaeger, H. Haas. *Harnessing Nonlinearity: Predicting Chaotic Systems and Saving
  Energy in Wireless Communication.* Science 304(5667):78–80, 2004.
- C. Gallicchio, A. Micheli, L. Pedrelli. *Deep Reservoir Computing: A Critical
  Experimental Analysis.* Neurocomputing 268:87–99, 2017.
  ([doi:10.1016/j.neucom.2016.12.089](https://doi.org/10.1016/j.neucom.2016.12.089))
  — `repos/esn/clrnn/deep_esn.py` is a PyTorch port of
  [DeepRC-TF](https://github.com/gallicch/DeepRC-TF).

**Datasets**

- MNIST: Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. *Gradient-Based Learning Applied
  to Document Recognition.* Proceedings of the IEEE, 1998.
- Fashion-MNIST: H. Xiao, K. Rasul, R. Vollgraf. *Fashion-MNIST: a Novel Image Dataset
  for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747, 2017.
- HHAR: A. Stisen et al. *Smart Devices are Different: Assessing and Mitigating Mobile
  Sensing Heterogeneities for Activity Recognition.* SenSys, 2015.
  ([UCI dataset 344](https://archive.ics.uci.edu/dataset/344))
- WISDM: J. R. Kwapisz, G. M. Weiss, S. A. Moore. *Activity Recognition using Cell Phone
  Accelerometers.* SIGKDD Explorations, 2011.
  ([WISDM Lab](https://www.cis.fordham.edu/wisdm/dataset.php))
- WISDM 200-step windows: R. R. Azghan et al. *Gated Adaptation for Continual Learning
  in Human Activity Recognition.* [arXiv:2603.10046](https://arxiv.org/abs/2603.10046).
  (This project uses non-overlapping windows instead of 50% overlap.)

**Statistical methods**

- J. Demšar. *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR
  7:1–30, 2006. (Friedman + Nemenyi and the Critical Difference diagram)
- S. Holm. *A Simple Sequentially Rejective Multiple Test Procedure.* Scandinavian
  Journal of Statistics 6(2):65–70, 1979.
