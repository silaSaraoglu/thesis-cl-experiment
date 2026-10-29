"""Rebuild all plots for a finished experiment from its summary.json.

Training results are already serialized, so figures can be regenerated at the
current print size without re-running any model:

    python replot.py results/single_head/experiment_20260630_022910
    python replot.py results/multi_head/experiment_20260629_154332 --out /tmp/figs

The head (single/multi) is inferred from the path unless --head is given.
Writes <out>/plots/... as both .png (300 dpi) and .pdf (vector).
"""

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_run_experiment(head):
    path = os.path.join(HERE, head, "run_experiment.py")
    if not os.path.isfile(path):
        sys.exit("no run_experiment.py for head %r at %s" % (head, path))
    # run_experiment.py bootstraps its own sys.path from __file__, so loading it
    # by location resolves its `models.*` imports the same way a direct run does.
    spec = importlib.util.spec_from_file_location("_replot_%s" % head, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _infer_head(exp_dir):
    parts = os.path.normpath(os.path.abspath(exp_dir)).split(os.sep)
    for head in ("single_head", "multi_head"):
        if head in parts:
            return head
    sys.exit("cannot infer head from %s; pass --head" % exp_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", help="experiment dir containing summary.json")
    ap.add_argument("--head", choices=["single_head", "multi_head"],
                    help="defaults to whichever appears in exp_dir")
    ap.add_argument("--out", help="output dir (default: exp_dir, overwrites in place)")
    a = ap.parse_args()

    summary_path = os.path.join(a.exp_dir, "summary.json")
    if not os.path.isfile(summary_path):
        sys.exit("no summary.json in %s" % a.exp_dir)

    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    results = summary["results"]

    head = a.head or _infer_head(a.exp_dir)
    out = a.out or a.exp_dir
    os.makedirs(out, exist_ok=True)

    mod = _load_run_experiment(head)
    mod.save_plots(results, out)
    print("wrote plots for %d results -> %s" % (len(results), os.path.join(out, "plots")))


if __name__ == "__main__":
    main()
