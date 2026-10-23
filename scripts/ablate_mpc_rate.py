"""MPC steer-rate weight with and without the observer, on dev segments.

Backs two README statements with a saved run: (1) without the observer the MPC is
far worse at every steer-rate weight, not just at the weight tuned with the
observer; (2) small steer-rate weights are unstable because the ARX model is
only valid for smooth steering.

  python scripts/ablate_mpc_rate.py
"""
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctrl.arx_controller import ARXController  # noqa: E402
from ctrl.evaluate import evaluate  # noqa: E402

SEGS = range(10000, 10200)


def main():
  rows = []
  for observer in (True, False):
    for w_du in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0):
      df = evaluate(partial(ARXController, solver="osqp", observer=observer, w_du=w_du), SEGS, workers=32)
      m = df[["lataccel_cost", "jerk_cost", "total_cost"]].mean().to_dict()
      rows.append(dict(observer=observer, w_du=w_du, **m))
      print(f"observer={observer!s:5} w_du={w_du:<4}: total {m['total_cost']:8.2f}  "
            f"lataccel {m['lataccel_cost']:.3f}  jerk {m['jerk_cost']:.2f}", flush=True)
  json.dump(dict(segs="10000-10199", rows=rows), open(ROOT / "results" / "tuning" / "mpc_rate_ablation.json", "w"), indent=1)


if __name__ == "__main__":
  main()
