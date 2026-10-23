"""How much of the cost is the simulator's sampling noise? (analysis only)

Runs controllers on dev segments 10000-10099 twice with the lockstep evaluator:
with the official stochastic sampling, and with a deterministic variant where
each step returns the token nearest to the expected lataccel. The deterministic
plant is used ONLY for this analysis; no reported score uses it.

  python scripts/noise_floor.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctrl.batch_eval import BatchSim  # noqa: E402
from ctrl.ilqr import BatchILQR  # noqa: E402

SEGS = range(10000, 10100)


def make(name, n):
  if name == "ilqr":
    from controllers.ilqr import PARAMS
    return BatchILQR(n, device="cuda" if torch.cuda.is_available() else "cpu", **PARAMS)
  C = importlib.import_module(f"controllers.{name}").Controller
  return [C() for _ in range(n)]


def main():
  summary = {}
  for name in ["pid_ff", "mpc", "ilqr"]:
    for det in (False, True):
      rows = BatchSim(SEGS, deterministic=det).run(make(name, len(SEGS)))
      key = f"{name}/{'deterministic' if det else 'stochastic'}"
      summary[key] = {k: float(np.mean([r[k] for r in rows])) for k in ("lataccel_cost", "jerk_cost", "total_cost")}
      v = summary[key]
      print(f"{key:24s} total {v['total_cost']:7.2f}  lataccel {v['lataccel_cost']:.3f}  jerk {v['jerk_cost']:.2f}", flush=True)
  json.dump(dict(segs="10000-10099", summary=summary), open(ROOT / "results" / "noise_floor.json", "w"), indent=1)


if __name__ == "__main__":
  main()
