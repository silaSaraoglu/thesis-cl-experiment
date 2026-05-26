"""
Baseline LSTM classifier wrapping BaselineRNN from the GIM repository.
"""
import os
import sys

_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths

from core.BaselineRNN import BaselineRNN 

def build_lstm(input_size, hidden_size, output_size, batch_size):
    return BaselineRNN(
        input_size, hidden_size, output_size, 'cpu',
        lstm=True, batch_size=batch_size, num_layers=1, orthogonal=False,
    )


def lstm_forward(model, x):
    """Single forward pass: allocates hidden state, returns last-timestep logits (B, output_size)."""
    hidden_state = model.reset_memory_state(batch_size=x.size(0))
    _, out = model(x, hidden_state)
    return out[:, -1, :]
