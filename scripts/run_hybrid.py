"""Step 2: hybrid MPC on the PWA model (GCS relaxation + rounding), on dev segments.

  python scripts/run_hybrid.py --eval --segs 10000 10032 --model results/pwa/pwa_yu_8.json --H 8 --name hybrid_gcs_yu8
  python scripts/run_hybrid.py --crosscheck --segs 10000 10008 --every 10

--eval        closed loop in the official simulator (one process per segment): per-segment
              costs -> results/hybrid/<name>_<range>.csv, per-step solver log -> <name>_steps.json
--crosscheck  at every N-th controlled step also solve the same problem exactly as a
              mixed-integer QP (SCIP) and record relaxed / rounded / optimal cost and times
              -> results/hybrid/crosscheck.json
"""
import argparse
import json
import logging
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from ctrl import hybrid_mpc  # noqa: E402
from ctrl.evaluate import seg_path  # noqa: E402
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator  # noqa: E402

OUT = ROOT / "results" / "hybrid"


def _eval_one(i, model, H, w_du, paths):
  logging.getLogger("drake").setLevel(logging.WARNING)
  c = hybrid_mpc.HybridMPCController(model, H=H, w_du=w_du, max_rounded_paths=paths, log=True)
  t0 = time.time()
  cost = TinyPhysicsSimulator(TinyPhysicsModel("./models/tinyphysics.onnx", debug=False), seg_path(i), controller=c).rollout()
  return {"seg": f"{i:05d}", **cost, "wall_s": time.time() - t0}, c.log


def _crosscheck_one(i, model, H, w_du, paths, every):
  logging.getLogger("drake").setLevel(logging.WARNING)
  rows, count = [], [0]
  orig = hybrid_mpc.HybridProblem.solve_gcs

  def wrapped(self, *a, **k):
    sol = orig(self, *a, **k)
    count[0] += 1
    if count[0] % every == 1 and "failed" not in sol:
      mi = self.solve_miqp(time_limit=120)
      opt = self.solve_fixed_modes(mi["modes"])[0] if mi["modes"] is not None else None
      rows.append(dict(seg=i, call=count[0], relaxed=sol["relaxed"], rounded=sol["rounded"], rounding=sol["rounding"],
                       miqp_optimal=opt, miqp_status=mi["status"], t_gcs=sol["t_total"], t_miqp=mi["t"],
                       same_first_mode=(mi["modes"] or [None])[0] == sol["modes"][0],
                       u0_gcs=sol["u0"], u0_miqp=mi["u0"]))
    return sol
  hybrid_mpc.HybridProblem.solve_gcs = wrapped
  c = hybrid_mpc.HybridMPCController(model, H=H, w_du=w_du, max_rounded_paths=paths)
  TinyPhysicsSimulator(TinyPhysicsModel("./models/tinyphysics.onnx", debug=False), seg_path(i), controller=c).rollout()
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--eval", action="store_true")
  ap.add_argument("--crosscheck", action="store_true")
  ap.add_argument("--segs", type=int, nargs=2, default=[10000, 10032])
  ap.add_argument("--model", default="results/pwa/pwa_yu_8.json")
  ap.add_argument("--H", type=int, default=8)
  ap.add_argument("--w_du", type=float, default=4.0)
  ap.add_argument("--paths", type=int, default=5)
  ap.add_argument("--every", type=int, default=10)
  ap.add_argument("--workers", type=int, default=32)
  ap.add_argument("--name", default="hybrid_gcs")
  args = ap.parse_args()
  OUT.mkdir(parents=True, exist_ok=True)
  segs = list(range(*args.segs))
  if args.eval:
    with Pool(min(args.workers, len(segs))) as pool:
      out = pool.map(partial(_eval_one, model=args.model, H=args.H, w_du=args.w_du, paths=args.paths), segs, chunksize=1)
    import pandas as pd
    df = pd.DataFrame([o[0] for o in out])
    tag = f"{segs[0]:05d}-{segs[-1]:05d}"
    df.to_csv(OUT / f"{args.name}_{tag}.csv", index=False)
    steps = [s for o in out for s in o[1]]
    ok = [s for s in steps if not s["failed"]]
    gap = [(s["rounded"] - s["relaxed"]) / max(s["rounded"], 1e-12) for s in ok]
    summary = dict(model=args.model, H=args.H, w_du=args.w_du, paths=args.paths, segs=tag,
                   mean=df[["lataccel_cost", "jerk_cost", "total_cost"]].mean().to_dict(),
                   n_steps=len(steps), n_failed=len(steps) - len(ok),
                   rel_gap_median=float(np.median(gap)), rel_gap_mean=float(np.mean(gap)),
                   ms_per_step_mean=1e3 * float(np.mean([s["t_total"] for s in ok])),
                   ms_per_step_p99=1e3 * float(np.percentile([s["t_total"] for s in ok], 99)),
                   rounding_used={r: sum(s.get("rounding") == r for s in ok) for r in ("random", "simulated", "given")},
                   note="ms/step measured inside parallel worker processes (machine shared), not an idle single-thread timing")
    json.dump(summary, open(OUT / f"{args.name}_{tag}_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
  if args.crosscheck:
    with Pool(min(args.workers, len(segs))) as pool:
      rows = [r for rs in pool.map(partial(_crosscheck_one, model=args.model, H=args.H, w_du=args.w_du,
                                           paths=args.paths, every=args.every), segs, chunksize=1) for r in rs]
    ok = [r for r in rows if r["miqp_optimal"] is not None]
    lb_gap = [(r["miqp_optimal"] - r["relaxed"]) / r["miqp_optimal"] for r in ok]
    ub_gap = [(r["rounded"] - r["miqp_optimal"]) / r["miqp_optimal"] for r in ok]
    summary = dict(model=args.model, H=args.H, n_instances=len(rows), n_with_optimum=len(ok),
                   relaxation_below_optimum_median=float(np.median(lb_gap)), relaxation_below_optimum_mean=float(np.mean(lb_gap)),
                   rounded_above_optimum_median=float(np.median(ub_gap)), rounded_above_optimum_mean=float(np.mean(ub_gap)),
                   rounded_within_1pct=float(np.mean(np.array(ub_gap) <= 0.01)),
                   same_first_mode=float(np.mean([r["same_first_mode"] for r in ok])),
                   mean_abs_u0_diff=float(np.mean([abs(r["u0_gcs"] - r["u0_miqp"]) for r in ok if r["u0_miqp"] is not None])),
                   t_gcs_mean_s=float(np.mean([r["t_gcs"] for r in ok])), t_miqp_mean_s=float(np.mean([r["t_miqp"] for r in ok])),
                   rows=rows)
    json.dump(summary, open(OUT / "crosscheck.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
  main()
