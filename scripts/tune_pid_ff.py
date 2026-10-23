"""Tune the global PID+FF gains once, on dev segments only.

Coordinate descent: Nelder-Mead on the continuous gains (kp, ki, kd, k_r,
k_roll) with the preview window fixed, then a grid over the window (s, L),
then Nelder-Mead again. Objective = mean total_cost on dev segs 10000-10199.
The final dev score is reported on 10000-10499. Test segs 0-4999 are never used.
"""
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctrl.evaluate import evaluate  # noqa: E402
from controllers.pid_ff import Controller  # noqa: E402

TUNE_SEGS = range(10000, 10200)
DEV_SEGS = range(10000, 10500)
GAINS = ["kp", "ki", "kd", "k_r", "k_roll"]
log = []


def cost(params, segs=TUNE_SEGS):
  df = evaluate(partial(Controller, **params), segs, workers=32)
  c = df.total_cost.mean()
  log.append({**params, "total_cost": c, "n_segs": len(df)})
  print(f"{c:8.3f}  " + "  ".join(f"{k}={v:.4g}" for k, v in params.items()), flush=True)
  return c


def nelder_mead(params, iters):
  x0 = np.array([params[k] for k in GAINS])
  f = lambda x: cost({**params, **dict(zip(GAINS, x))})
  # initial simplex: +/-20% steps (or 0.02 for near-zero gains)
  simplex = [x0] + [x0 + np.eye(len(x0))[i] * max(0.2 * abs(x0[i]), 0.02) for i in range(len(x0))]
  res = minimize(f, x0, method="Nelder-Mead",
                 options=dict(maxfev=iters, initial_simplex=np.array(simplex), xatol=1e-3, fatol=0.05))
  return {**params, **dict(zip(GAINS, res.x))}


def main():
  params = dict(kp=0.25, ki=0.05, kd=-0.05, k_r=0.55, k_roll=-0.6, s=1, L=3)
  params = nelder_mead(params, 100)
  best = (cost(params), params)
  for s in range(0, 5):
    for L in [1, 2, 3, 5, 8]:
      p = {**params, "s": s, "L": L}
      c = cost(p)
      if c < best[0]:
        best = (c, p)
  params = nelder_mead(best[1], 100)
  dev = evaluate(partial(Controller, **params), DEV_SEGS, workers=32)
  m = dev[["lataccel_cost", "jerk_cost", "total_cost"]].mean().to_dict()
  print("final params", params, "dev(500):", m)
  out = ROOT / "results" / "tuning"
  out.mkdir(parents=True, exist_ok=True)
  json.dump({"params": params, "dev_10000-10499": m, "tune_segs": "10000-10199", "log": log},
            open(out / "pid_ff.json", "w"), indent=1)


if __name__ == "__main__":
  main()
