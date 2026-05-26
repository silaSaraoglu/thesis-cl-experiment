"""
Shared utilities for all ESN continual-learning experiment files.

Architecture — DeepReservoirClassifier (models/ESN/repo/clrnn/deep_esn.py):

    Input (B, T, input_size)
        │
    DeepReservoir  [fixed weights, no gradients]
        ├─ ReservoirLayer 1  (ReservoirCell: leaky-integration, random sparse weights)
        ├─ ReservoirLayer 2  ...
        └─ ...
        │   concat=True  → output uses all layer states concatenated (richer features)
        │   return_sequences=False → only the LAST time-step hidden state is forwarded
        ▼
    h_last: shape (B, esn_units)   [last reservoir layer, last time-step]
        │
    Linear readout  [only trainable part]
        ▼
    Logits (B, num_classes=2)

Design choices matching the original splitmnist_esn.py:
  - concat=True            matches DeepReservoirClassifier default
  - feedforward_layers=1   single linear readout, no hidden MLP layer
  - return_sequences=False classifier sees only the final state, not the full sequence
  - connectivity_input=10, connectivity_recurrent=10  original repo defaults (not tuned)

The reservoir weights (kernel, recurrent_kernel, bias) all have requires_grad=False.
Only the readout Linear layer is updated during training.

Original repo: models/ESN/repo/experiments/splitmnist_esn.py
"""
import os
import sys
import torch

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from clrnn.deep_esn import DeepReservoirClassifier

def build_model(input_size, args, device):
    """
    Build a DeepReservoirClassifier matching the original repo's setup.

    Differences from the original splitmnist_esn.py call:
      - arg name: esn_units  (original used args.esn_units directly)
      - concat=True          (original default; we make it explicit)
      - connectivity params  omitted → repo defaults of 10 apply
    """
    model = DeepReservoirClassifier(
        input_size=input_size,
        num_classes=2,
        units=args.esn_units,           # reservoir neurons per layer
        layers=1,         # number of stacked reservoir layers
        concat=True,                    # concatenate all layer states (original default)
        feedforward_layers=1,           # single linear readout
        spectral_radius=0.99,
        leaky=1.0,               # leakage rate: 1.0 = no leakage, <1 = smoothed state
        input_scaling=1.0,
        return_sequences=False,         # use only last time-step state for classification
    )
    return model

@torch.no_grad()
def predict_step(model, x):
    """Return predicted class indices for a batch."""
    model.eval()
    device = next(model.parameters()).device
    return model(x.to(device)).argmax(dim=1).cpu().numpy()
