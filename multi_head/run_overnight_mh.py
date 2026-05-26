"""
Launch the Multi-Head (TIL) pipeline detached from the terminal.

The spawned process survives terminal close — safe to leave overnight.
stdout + stderr are written to a timestamped log file under logs/.

Usage:
    python run_overnight_mh.py
    python run_overnight_mh.py --subset 2000 --n_trials 30 --num_runs 3
    python run_overnight_mh.py --results_dir ../results/my_mh_run

Monitor progress (PowerShell):
    Get-Content -Wait logs\overnight_mh_<TIMESTAMP>.log

Stop the run:
    Stop-Process -Id <PID shown on launch>
"""
import os
import sys
import subprocess
from datetime import datetime

HERE    = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = os.path.join(LOG_DIR, f"overnight_mh_{timestamp}.log")
script  = os.path.join(HERE, "run_experiment.py")

DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000

env                     = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

cmd = [sys.executable, "-u", script] + sys.argv[1:]

with open(logfile, "w", encoding="utf-8") as log:
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        cwd=HERE,
        env=env,
        creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
    )

print("\nMulti-Head (TIL) experiment started — safe to close this terminal.\n")
print(f"  PID     : {proc.pid}")
print(f"  Log     : {logfile}")
print(f"  Monitor : Get-Content -Wait '{logfile}'")
print(f"  Stop    : Stop-Process -Id {proc.pid}")
