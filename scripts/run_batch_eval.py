"""Evaluate the batched iLQR on many segments at once (GPU) with ctrl/batch_eval.py.

The lockstep evaluator reproduces the official simulator exactly (checked by
scripts/verify_batch_eval.py); only the controller computation is batched.

  python scripts/run_batch_eval.py --start 0 --num_segs 5000 --H 12 --iters 1 --w_du 1.0 --name ilqr

Writes results/runs/<name>_<first>-<last>.csv (same format as run_eval.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctrl.batch_eval import BatchSim  # noqa: E402
from ctrl.ilqr import BatchILQR  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--start", type=int, default=10000)
  p.add_argument("--num_segs", type=int, default=100)
  p.add_argument("--H", type=int, default=12)
  p.add_argument("--iters", type=int, default=1)
  p.add_argument("--w_du", type=float, default=1.0)
  p.add_argument("--shifts", type=float, nargs="*", default=[], help="steer offsets tried around the warm start")
  p.add_argument("--var_weight", type=float, default=0.0, help="risk-sensitive weight on predicted variance")
  p.add_argument("--name", default=None)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--segs_file", default=None, help="text file with one segment index per line (overrides --start)")
  p.add_argument("--ctrl", default="ilqr", choices=["ilqr", "lookahead", "mppi"])
  p.add_argument("--kw", default="{}", help="extra controller keyword arguments as JSON")
  p.add_argument("--out", default=None, help="results subfolder (default: runs; dev sweeps use tuning/<name>)")
  p.add_argument("--deterministic", action="store_true",
                 help="ANALYSIS ONLY: mean-token plant; results go to results/noise_floor/, never results/runs/")
  args = p.parse_args()
  segs = range(args.start, args.start + args.num_segs)
  if args.segs_file:
    segs = [int(x) for x in open(args.segs_file).read().split()]
  if args.ctrl == "ilqr":
    params = dict(H=args.H, iters=args.iters, w_du=args.w_du, shifts=tuple(args.shifts))
    if args.var_weight:
      params["var_weight"] = args.var_weight
    params.update(json.loads(args.kw))
    make = lambda n: BatchILQR(n, device=args.device, **params)
  else:
    from ctrl.variance_ctrl import BatchLookahead, BatchMPPI
    params = json.loads(args.kw)
    cls = BatchLookahead if args.ctrl == "lookahead" else BatchMPPI
    make = lambda n: cls(n, device=args.device, **params)
  t0 = time.time()
  sim = BatchSim(segs, deterministic=args.deterministic)
  ctrl = make(len(segs))
  df = pd.DataFrame(sim.run(ctrl))
  wall = time.time() - t0
  name = args.name or f"ilqr_H{args.H}_it{args.iters}_w{args.w_du:g}"
  tag = Path(args.segs_file).stem if args.segs_file else f"{segs[0]:05d}-{segs[-1]:05d}"
  out_dir = ROOT / "results" / ("noise_floor" if args.deterministic else args.out or ("stress" if args.segs_file else "runs"))
  out_dir.mkdir(parents=True, exist_ok=True)
  out = out_dir / f"{name}_{tag}.csv"
  df.to_csv(out, index=False)
  meta_file = out_dir / "batch_eval_meta.json"
  meta = json.load(open(meta_file)) if meta_file.exists() else {}
  meta[out.name] = dict(ctrl=args.ctrl, params={k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
                        evaluator="ctrl/batch_eval.py (lockstep, GPU controller)", deterministic_plant=args.deterministic, wall_s=wall,
                        device=torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu")
  json.dump(meta, open(meta_file, "w"), indent=1)
  m = df[["lataccel_cost", "jerk_cost", "total_cost"]].mean()
  print(f"{name} on {len(df)} segs ({tag}): lataccel {m.lataccel_cost:.3f}  "
        f"jerk {m.jerk_cost:.3f}  total {m.total_cost:.3f}   wall {wall:.0f}s")


if __name__ == "__main__":
  main()
