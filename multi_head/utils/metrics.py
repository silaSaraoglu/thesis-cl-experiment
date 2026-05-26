import os, sys
_p = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_p, 'repos')): _p = os.path.dirname(_p)
if _p not in sys.path: sys.path.insert(0, _p)
import setup_paths
from shared.metrics import CLMetrics, cohen_kappa, save_results

__all__ = ["CLMetrics", "cohen_kappa", "save_results"]
