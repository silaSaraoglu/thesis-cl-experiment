"""
Multi-Head (TIL) -- shared configuration.

Global constants used across the three entry points (run_tuning.py, run_experiment.py,
run_epoch_analysis.py).
"""
import warnings
warnings.filterwarnings("ignore")  # quiet torch.sparse + Avalanche UserWarnings during runs
# Avalanche's @deprecated decorator calls warnings.simplefilter("once", DeprecationWarning)
# on every invocation (e.g. replay's storage_policy.update), re-enabling the warning ahead
# of the ignore filter above. Silence the output sink so no library can bring warnings back.
warnings.showwarning = lambda *args, **kwargs: None

import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# Global settings
DATASETS    = ["mnist", "wisdm", "hhar", "fashion_sw"]
MODELS      = ["gim_alstm_mh", "lstm_mh", "joint_lstm_mh",
                "lstm_replay_mh",
                "esn_naive_mh", "joint_esn_mh"]
SEED        = 42
RESULTS_DIR = os.path.join(HERE, "..", "results", "multi_head")
MNIST_DIR   = os.path.join(HERE, "..", "data", "mnist")
WISDM_DIR     = os.path.join(HERE, "..", "data", "wisdm")
HHAR_DIR      = os.path.join(HERE, "..", "data", "hhar")
FASHION_DIR   = os.path.join(HERE, "..", "data", "fashion_mnist")
SUBSET_MNIST      = None   # samples per task in Phase 2 (None = full dataset, ~12k/task)
SUBSET_WISDM      = None   # samples per task in Phase 2 (None = full dataset, balanced)
SUBSET_HHAR       = None   # samples per task in Phase 2 (None = full dataset; balanced to ~6750 windows/device at preprocessing)
SUBSET_FASHION    = None   # samples per task in Phase 2 (None = full dataset, ~12k/task)

# MNIST task-pair configurations
# 15 digit-pairing / order variants (each task pairs one even + one odd digit,
# all 10 digits used once, so the even->0 / odd->1 label map is identical in every
# task and only the digit domain changes); results stay per-config.
MNIST_DATASET_CONFIGS = {
    "config_1": [[0,1],[2,3],[4,5],[6,7],[8,9]],  # 0v1, 2v3, 4v5, 6v7, 8v9
    "config_2": [[1,2],[3,4],[5,6],[7,8],[0,9]],  # 1v2, 3v4, 5v6, 7v8, 0v9
    "config_3": [[0,3],[2,5],[4,7],[6,9],[8,1]],  # 0v3, 2v5, 4v7, 6v9, 8v1 (mixed-parity, matches SH)
    "config_4": [[8,3],[2,7],[4,9],[6,1],[0,5]],
    "config_5": [[6,9],[0,7],[8,1],[2,3],[4,5]],
    "config_6": [[8,5],[2,7],[6,3],[0,9],[4,1]],
    "config_7": [[2,5],[4,7],[0,3],[6,9],[8,1]],
    "config_8": [[8,7],[4,5],[0,9],[2,1],[6,3]],
    "config_9": [[0,7],[4,9],[8,1],[6,3],[2,5]],
    "config_10": [[0,3],[6,9],[2,5],[4,1],[8,7]],
    "config_11": [[4,1],[8,5],[0,7],[6,3],[2,9]],
    "config_12": [[6,7],[0,5],[8,1],[4,3],[2,9]],
    "config_13": [[2,7],[8,9],[0,1],[6,5],[4,3]],
    "config_14": [[8,7],[0,3],[6,5],[2,1],[4,9]],
    "config_15": [[6,1],[4,7],[2,3],[8,9],[0,5]],
}

# WISDM task-pairing / ordering configurations
# 15 variants over the 6 activity classes (0=Walk,1=Jog,2=Up,3=Down,4=Sit,5=Stand).
# config_1-3 keep the semantic pairs (Walk/Jog · Up/Down · Sit/Stand) in different orders;
# config_4-15 mix the pairings (like MNIST/Fashion-SW). Each task pairs two classes and all
# 6 classes are used once per config; results stay per-config. All 15 use distinct class
# pairings (WISDM has 15 possible pairings of 6 classes into 3 tasks), so no config repeats.
WISDM_DATASET_CONFIGS = {
    "config_1": [[0,1],[2,3],[4,5]],  # Walk/Jog · Up/Down · Sit/Stand
    "config_2": [[0,1],[4,5],[2,3]],  # Walk/Jog · Sit/Stand · Up/Down
    "config_3": [[2,3],[0,1],[4,5]],  # Up/Down  · Walk/Jog · Sit/Stand
    "config_4": [[0,2],[1,4],[3,5]],  # Walk/Up   · Jog/Sit    · Down/Stand
    "config_5": [[1,3],[0,5],[2,4]],  # Jog/Down  · Walk/Stand · Up/Sit
    "config_6": [[0,4],[2,5],[1,3]],  # Walk/Sit  · Up/Stand   · Jog/Down
    "config_7": [[0,3],[1,5],[2,4]],  # Walk/Down · Jog/Stand  · Up/Sit
    "config_8": [[1,2],[0,3],[4,5]],  # Jog/Up    · Walk/Down  · Sit/Stand
    "config_9": [[3,4],[0,5],[1,2]],  # Down/Sit  · Walk/Stand · Jog/Up
    "config_10": [[0,5],[2,3],[1,4]], # Walk/Stand · Up/Down   · Jog/Sit
    "config_11": [[1,5],[0,2],[3,4]], # Jog/Stand  · Walk/Up   · Down/Sit
    "config_12": [[0,3],[1,4],[2,5]], # Walk/Down  · Jog/Sit   · Up/Stand
    "config_13": [[3,5],[0,4],[1,2]], # Down/Stand · Walk/Sit  · Jog/Up
    "config_14": [[2,4],[3,5],[0,1]], # Up/Sit     · Down/Stand · Walk/Jog
    "config_15": [[1,5],[2,3],[0,4]], # Jog/Stand  · Up/Down   · Walk/Sit
}

# HHAR task orderings (device-model indices: 0=nexus4, 1=s3, 2=s3mini, 3=samsungold).
HHAR_DATASET_CONFIGS = {
    "config_1": [0, 1, 2, 3],   # nexus4 -> s3 -> s3mini -> samsungold
    "config_2": [3, 2, 1, 0],   # reverse
    "config_3": [0, 2, 1, 3],   # alternating cross-cluster (max domain jumps)
    "config_4": [2, 1, 3, 0],
    "config_5": [1, 0, 2, 3],
    "config_6": [3, 1, 2, 0],
    "config_7": [3, 1, 0, 2],
    "config_8": [1, 3, 0, 2],
    "config_9": [1, 2, 0, 3],
    "config_10": [0, 2, 3, 1],
    "config_11": [2, 0, 1, 3],
    "config_12": [2, 3, 0, 1],
    "config_13": [0, 3, 2, 1],
    "config_14": [1, 2, 3, 0],
    "config_15": [3, 0, 1, 2],
}

# Fashion-MNIST Summer-vs-Winter task orderings (within-category season pairs).
# Every task pairs one summer {0,3,5} with one winter {2,4,9} class, so the summer->0 /
# winter->1 label map is fixed and only the garment domain changes across tasks. Only 6
# such pairings exist, so the 15 configs vary both pairing and task order.
FASHION_SW_DATASET_CONFIGS = {
    "config_1": [[0,2],[3,4],[5,9]],   # tops -> long-garments -> footwear
    "config_2": [[5,2],[3,9],[0,4]],
    "config_3": [[3,2],[5,4],[0,9]],
    "config_4": [[0,4],[5,9],[3,2]],
    "config_5": [[0,4],[5,2],[3,9]],
    "config_6": [[3,9],[5,4],[0,2]],
    "config_7": [[0,4],[3,2],[5,9]],
    "config_8": [[3,4],[5,9],[0,2]],
    "config_9": [[3,2],[0,4],[5,9]],
    "config_10": [[5,2],[0,4],[3,9]],
    "config_11": [[0,9],[3,4],[5,2]],
    "config_12": [[5,2],[0,9],[3,4]],
    "config_13": [[0,9],[5,4],[3,2]],
    "config_14": [[0,2],[5,4],[3,9]],
    "config_15": [[5,9],[0,2],[3,4]],
}

DATASET_CONFIGS = {"mnist": MNIST_DATASET_CONFIGS, "wisdm": WISDM_DATASET_CONFIGS, "hhar": HHAR_DATASET_CONFIGS, "fashion_sw": FASHION_SW_DATASET_CONFIGS}

# CPU-only: fully deterministic (no cuDNN/cuBLAS/atomic GPU jitter). Runs are executed
# sequentially and torch uses all CPU cores per run.
DEVICE            = torch.device("cpu")

# Epochs are fixed at 5 for the main comparison; the epoch-sensitivity sweep is a separate
# study in run_epoch_analysis.py.
REFERENCE_EPOCHS = 5
# Tuning-holdout config (consumed by run_tuning.py; also logged in run metadata).
# Samples per task reserved as the tuning holdout pool (~10% of each task's training data).
TUNE_SUBSET_MNIST = 1200   # ~10% of 12,000 per digit pair
TUNE_SUBSET_WISDM = 62     # ~10% of 622 per activity pair (balanced)
TUNE_SUBSET_HHAR  = 675    # ~10% of ~6750 windows per device (held out for tuning)
TUNE_SUBSET_FASHION = 1200  # ~10% of 12,000 per garment/accessory pair
TUNE_SPLIT_SEED   = 0      # fixed seed for tuning/training data split

# Cached tuning results; produced by run_tuning.py, read by run_experiment / run_epoch_analysis.
TUNED_PARAMS_CACHE = os.path.join(HERE, "tuned_params.json")
