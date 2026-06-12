# Thesis Project — Continual Learning Benchmark

A comparative study of continual learning (CL) strategies across three sequence
datasets, evaluated under two settings — **Domain-Incremental Learning (DIL,
`single_head/`)** and **Task-Incremental Learning (TIL, `multi_head/`)**.

Every dataset is turned into a sequence of **binary classification tasks** presented
one after another, so all model families (GIM, LSTM, ESN) see the same recurrent
input format: `(batch, timesteps, features)`.

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

## Datasets

Three datasets are used. Each is split into a fixed number of **binary tasks**, and
within every task the two raw class labels are remapped to `{0, 1}`.

| Dataset | Domain | # Tasks | Classes | Train total | Test total | Input shape (T × F) |
|---------|--------|:-------:|:-------:|:-----------:|:----------:|:-------------------:|
| **MNIST** | Handwritten digits (image → sequence) | 5 | 10 digits | 60,000 | 10,000 | 28 × 28 |
| **WISDM** | Smartphone accelerometer (HAR) | 3 | 6 activities | 1,866 | 828 | 200 × 3 |
| **UWave** | Wiimote accelerometer (gestures) | 4 | 8 gestures | 896 | 3,582 | 315 × 3 |

Defined in [repos/gim/tasks/dataset_cl.py](repos/gim/tasks/dataset_cl.py); task counts
and names are registered in [single_head/run_experiment.py](single_head/run_experiment.py#L284-L288).

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

**Task configurations.** To remove the effect of *which* digits are paired, MNIST is
run with three pairings and the results are averaged
([run_experiment.py:83-87](single_head/run_experiment.py#L83-L87)):

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
   so the sequence format is preserved (`shared/utils.py::permute_mnist`,
   [utils.py:21-26](shared/utils.py#L21-L26)).
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

### WISDM — 3 tasks (activity pairs)

Wireless Sensor Data Mining accelerometer dataset (`WISDM_ar_v1.1`). 6 activities are
paired into **3 binary tasks** by motion category:

| Task | Classes | Train | Test |
|:----:|:-------:|:-----:|:----:|
| 1 | Walking vs Jogging   | 622 | 276 |
| 2 | Upstairs vs Downstairs | 622 | 276 |
| 3 | Sitting vs Standing  | 622 | 276 |
| **Total** | | **1,866** | **828** |

- **Class balance:** every class is downsampled to the smallest class count, giving
  **exactly 311 train / 138 test samples per class** (perfectly balanced). All three
  tasks are therefore equal in size.

**Task configurations.** Same three activity pairs in three presentation orders,
averaged ([run_experiment.py:90-94](single_head/run_experiment.py#L90-L94)):

| Config | T1 | T2 | T3 |
|--------|----|----|----|
| `config_1` | Walk/Jog | Up/Down | Sit/Stand |
| `config_2` | Walk/Jog | Sit/Stand | Up/Down |
| `config_3` | Up/Down | Walk/Jog | Sit/Stand |

#### Preprocessing

Run automatically on first use; the result is cached to
`data/wisdm/wisdm_processed.npz` (`_wisdm_preprocess`,
[dataset_cl.py:131-206](repos/gim/tasks/dataset_cl.py#L131-L206)):

1. **Parse** the raw text file (`user_id, activity, x, y, z` per line); only the 6
   mapped activities are kept.
2. **Sliding windows** per `(user, activity)`: window length **200** (10 s at 20 Hz),
   stride **100** (50% overlap). Each window → one sample of shape `(200, 3)`. The
   200-step / 50%-overlap setting follows the continual-learning HAR convention of
   Azghan et al. (*Gated Adaptation for Continual Learning in HAR*, arXiv:2603.10046).
3. **User-disjoint split:** the first **80% of users** form the train set, the rest the
   test set (windows never leak a user across the split).
4. **Class balancing:** every class is downsampled to the smallest class count
   (seed 0), so the binary tasks are equal in size.
5. **Z-score normalisation** per channel, using **train statistics only**.

---

### UWave — 4 tasks (gesture pairs)

UWave Gesture Library (UEA "All" variant). 8 gestures paired into **4 binary tasks**:

| Task | Classes | Train | Test |
|:----:|:-------:|:-----:|:----:|
| 1 | G1 vs G2 | 230 | 889 |
| 2 | G3 vs G4 | 216 | 904 |
| 3 | G5 vs G6 | 238 | 882 |
| 4 | G7 vs G8 | 212 | 907 |
| **Total** | | **896** | **3,582** |

- **Class balance:** near-balanced within each task (per-class train counts 100–127).
- **Inverted train/test ratio:** the archive ships **far more test than train**
  (~900 vs ~220 per task). This is the fixed UEA split and is kept as-is.
- **Difficulty varies by pair**, so task order matters — hence multiple orderings.

**Task configurations.** Same four pairs, three orders, averaged. The code treats
G1/G2 and G7/G8 as the *hard* pairs and G3/G4, G5/G6 as the *easy* ones
([run_experiment.py:97-102](single_head/run_experiment.py#L97-L102)):

| Config | T1 | T2 | T3 | T4 | Order |
|--------|----|----|----|----|-------|
| `config_1` | G3/G4 | G5/G6 | G1/G2 | G7/G8 | easy → hard |
| `config_2` | G7/G8 | G1/G2 | G5/G6 | G3/G4 | reverse |
| `config_3` | G1/G2 | G3/G4 | G7/G8 | G5/G6 | alternating |

#### Preprocessing

Run automatically on first use; cached to `data/uwave/uwave_processed.npz`
(`_uwave_preprocess`, [dataset_cl.py:335-384](repos/gim/tasks/dataset_cl.py#L335-L384)):

1. **Parse** the UEA `.ts` files (`_TRAIN.ts` / `_TEST.ts`).
2. **Reshape from univariate to triaxial:** the "All" variant stores each gesture as a
   univariate series of length **945 = 315 × 3** (X, then Y, then Z concatenated). It is
   reshaped back to `(315, 3)` so the input matches the WISDM convention
   (`input_size=3`).
3. **Z-score normalisation** per channel, using **train statistics only**.

---

### Train / validation / tuning splits (all datasets)

Applied at load time, per task, inside each `*_CL` dataset class:

- **Tuning holdout (fixed seed `TUNE_SPLIT_SEED = 0`).** A fixed slice of each task's
  training data is reserved as a holdout pool used **only** for Optuna hyperparameter
  search: 1,200 samples/task for MNIST, 62 for WISDM, 22 for UWave (~10% each,
  [run_experiment.py:126-129](single_head/run_experiment.py#L126-L129)). Final training
  runs use everything *outside* the holdout.
- **Validation split.** The remaining training data is split into train/val (val
  fraction 20% for MNIST, 25% for WISDM/UWave), stratified by label.
- **Test set.** The held-out test split is fixed and used after every task to fill the
  accuracy matrix `R` that all CL metrics are computed from.

---

## Single-Head (DIL) Models

In the single-head / Domain-Incremental setting, every model has **one shared 2-class
output head** used for all tasks, and **no task ID is given at test time**. Tasks are
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
split, and a 2-class shared head. What differs is **which parameters update** and **how
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
- **Per-task epochs** are tuned per (model, dataset) via Optuna, except GIM-on-MNIST which uses the paper's fixed hyper-parameters.
- **Grad clipping (max-norm 5.0)** is applied wherever gradients flow through time — the LSTM and GIM loops, which train recurrent weights via backprop-through-time (prone to exploding gradients over the 28/200/315-step sequences). ESN skips it because only a linear readout is trained on the frozen reservoir (no BPTT), and SLDA has no gradient step at all.

---

## References

**Datasets**
- **MNIST** — Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE, 1998.
- **WISDM** — J. R. Kwapisz, G. M. Weiss, S. A. Moore. *Activity Recognition using Cell Phone Accelerometers.* SIGKDD Explorations, 2011. ([WISDM Lab](https://www.cis.fordham.edu/wisdm/dataset.php))
- **UWave** — J. Liu, L. Zhong, J. Wickramasuriya, V. Vasudevan. *uWave: Accelerometer-based Personalized Gesture Recognition and Its Applications.* Pervasive and Mobile Computing, 2009.

**Models & CL strategies**
- **GIM** — A. Cossu, A. Carta, D. Bacciu. *Continual Learning with Gated Incremental Memories for Sequential Data Processing.* IJCNN, 2020.
- **ESN for CL** — A. Cossu, D. Bacciu, A. Carta, C. Gallicchio, V. Lomonaco. *Continual Learning with Echo State Networks.* ESANN, 2021.

**Preprocessing reference**
- **WISDM windowing (200 steps, 50% overlap)** — R. R. Azghan et al. *Gated Adaptation for Continual Learning in Human Activity Recognition.* [arXiv:2603.10046](https://arxiv.org/abs/2603.10046).
