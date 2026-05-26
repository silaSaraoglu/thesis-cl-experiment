# single_head/run_overnight_sh.ps1  --  Single-Head (DIL) pipeline
#
# Launches run_experiment.py detached from this terminal.
# Safe to close PowerShell after running — the experiment keeps going.
#
# Monitor progress:
#   Get-Content -Wait logs\overnight_sh_<TIMESTAMP>.log
#
# Stop early:
#   Stop-Process -Id <PID printed below>

Set-Location $PSScriptRoot

python run_overnight_sh.py
