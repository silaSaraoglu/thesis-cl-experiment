import os
import sys

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from core.BaselineRNN import BaselineRNN


class SingleHeadLSTM(BaselineRNN):
    """BaselineRNN with a standard forward(x) → logits interface (single shared head)."""

    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super().__init__(
            input_size, hidden_size, output_size, 'cpu',
            lstm=True, batch_size=batch_size, num_layers=1, orthogonal=False,
        )

    def forward(self, x):
        hidden_state = self.reset_memory_state(batch_size=x.size(0))
        out, _       = self.rnn_module(x, hidden_state)
        return self.linear(out[:, -1, :])
