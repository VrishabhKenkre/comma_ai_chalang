"""Plot target vs achieved lateral acceleration (and steer) for a few test segments.

Runs each controller through the official simulator on segments 00000-00004
(the same ones eval.py shows as sample rollouts) and saves
results/plots/segments.png plus the per-segment costs it printed.

  python scripts/plot_segments.py pid_ff mpc ilqr
"""
import importlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctrl.evaluate import seg_path  # noqa: E402
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, COST_END_IDX  # noqa: E402

SEGS = [0, 1, 2, 3, 4]
COLORS = {"pid": "#7f8c8d", "pid_ff": "#2980b9", "mpc": "#e67e22", "mpc_admm": "#d35400", "ilqr": "#c0392b", "lqr": "#8e44ad"}


def run(args):
  name, i = args
  ctrl = importlib.import_module(f"controllers.{name}").Controller()
  sim = TinyPhysicsSimulator(TinyPhysicsModel("./models/tinyphysics.onnx", debug=False), seg_path(i), controller=ctrl)
  cost = sim.rollout()
  return name, i, cost, sim.target_lataccel_history, sim.current_lataccel_history, sim.action_history


def main():
  names = sys.argv[1:]
  with Pool(min(32, len(names) * len(SEGS))) as pool:
    out = pool.map(run, [(n, i) for n in names for i in SEGS])
  res = {(n, i): (c, tgt, cur, act) for n, i, c, tgt, cur, act in out}
  fig, axs = plt.subplots(len(SEGS), 2, figsize=(16, 3.0 * len(SEGS)), gridspec_kw=dict(width_ratios=[3, 2]))
  costs = {}
  for row, i in enumerate(SEGS):
    ax, ax2 = axs[row]
    tgt = res[(names[0], i)][1]
    ax.plot(tgt, color="#27ae60", lw=2, label="target")
    for n in names:
      c, _, cur, act = res[(n, i)]
      costs[f"{n}/{i:05d}"] = c
      ax.plot(cur, color=COLORS.get(n), lw=1, label=f"{n} (total {c['total_cost']:.1f})")
      ax2.plot(act, color=COLORS.get(n), lw=1, label=n)
    for a in (ax, ax2):
      a.axvspan(CONTROL_START_IDX, COST_END_IDX, color="k", alpha=0.04)
      a.axvline(CONTROL_START_IDX, color="k", ls="--", lw=0.8)
    ax.set_title(f"segment {i:05d}: lateral acceleration (scored window shaded)")
    ax2.set_title("steer command")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("m/s^2")
  axs[-1][0].set_xlabel("step (0.1 s)")
  axs[-1][1].set_xlabel("step (0.1 s)")
  fig.tight_layout()
  out_dir = ROOT / "results" / "plots"
  out_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_dir / "segments.png", dpi=110)
  json.dump(costs, open(out_dir / "segments_costs.json", "w"), indent=1)
  for k, v in costs.items():
    print(k, {kk: round(vv, 2) for kk, vv in v.items()})


if __name__ == "__main__":
  main()
