"""Small sweep of the iLQR settings on dev segments (results appended as they finish).

  python scripts/tune_ilqr.py --segs 10000 10100
"""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctrl.evaluate import evaluate  # noqa: E402
from ctrl.ilqr import ILQRController  # noqa: E402

GRID = [dict(H=12, iters=1, w_du=w) for w in (0.5, 1.0, 2.0, 4.0)] + \
       [dict(H=20, iters=2, w_du=w) for w in (1.0, 2.0, 4.0)]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--segs", type=int, nargs=2, default=[10000, 10100])
  ap.add_argument("--out", default="results/tuning/ilqr_sweep.json")
  args = ap.parse_args()
  out = ROOT / args.out
  rows = json.load(open(out)) if out.exists() else []
  done = {json.dumps(r["params"], sort_keys=True) for r in rows if r["segs"] == args.segs}
  for params in GRID:
    if json.dumps(params, sort_keys=True) in done:
      continue
    df = evaluate(partial(ILQRController, **params), range(*args.segs), workers=32)
    m = df[["lataccel_cost", "jerk_cost", "total_cost"]].mean().to_dict()
    rows.append(dict(params=params, segs=args.segs, **m))
    json.dump(rows, open(out, "w"), indent=1)
    print(params, {k: round(v, 3) for k, v in m.items()}, flush=True)


if __name__ == "__main__":
  main()
