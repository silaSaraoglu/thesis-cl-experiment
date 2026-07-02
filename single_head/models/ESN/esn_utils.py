"""
Wrapper around DeepReservoirClassifier for single-head ESN experiments.
Original repo: repos/esn/clrnn/deep_esn.py (splitmnist_esn.py)
"""
import os
import sys
import torch

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import shared.utils

from clrnn.deep_esn import DeepReservoirClassifier


class SingleHeadESN(DeepReservoirClassifier):

    def __init__(self, input_size, args, num_classes=None):
        if num_classes is None:
            num_classes = getattr(args, 'output_size', 2)
        super().__init__(
            input_size=input_size,
            num_classes=num_classes,
            units=args.esn_units,
            layers=1,
            concat=True,
            feedforward_layers=1,
            spectral_radius=0.99,
            leaky=1.0,
            input_scaling=1.0,
            return_sequences=False,
        )

    def forward(self, x):
        _, h = self.hidden(x)
        return self.output(h[-1])

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        device = next(self.parameters()).device
        return self.forward(x.to(device)).argmax(dim=1).cpu().numpy()
