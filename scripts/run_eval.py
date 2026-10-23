"""Batch evaluation that saves per-segment costs.

Uses the official tinyphysics.run_rollout, so every number is produced by the
unmodified simulator. Paths are built exactly like eval.py builds them
("data/00000.csv"), because the simulator seeds its RNG from the path string.

  python scripts/run_eval.py --controller pid --start 0 --num_segs 5000

Writes results/runs/<controller>_<first>-<last>.csv with one row per segment.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one core per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm.contrib.concurrent import process_map

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from tinyphysics import run_rollout  # noqa: E402

MODEL_PATH = "./models/tinyphysics.onnx"


def one_segment(data_file, controller):
  cost, _, _ = run_rollout(data_file, controller, MODEL_PATH, debug=False)
  return {"seg": data_file.stem, **cost}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--controller", required=True)
  p.add_argument("--start", type=int, default=0, help="first segment index")
  p.add_argument("--num_segs", type=int, default=500)
  p.add_argument("--workers", type=int, default=32)
  p.add_argument("--tag", default="", help="optional suffix for the output file")
  args = p.parse_args()

  # Same construction as eval.py: sorted(Path('./data').iterdir())
  files = sorted(Path("./data").iterdir())[args.start:args.start + args.num_segs]
  rows = process_map(partial(one_segment, controller=args.controller), files,
                     max_workers=args.workers, chunksize=10)
  df = pd.DataFrame(rows)

  out_dir = ROOT / "results" / "runs"
  out_dir.mkdir(parents=True, exist_ok=True)
  name = f"{args.controller}{args.tag}_{files[0].stem}-{files[-1].stem}.csv"
  df.to_csv(out_dir / name, index=False)
  m = df[["lataccel_cost", "jerk_cost", "total_cost"]].mean()
  print(f"{args.controller}{args.tag} on {len(df)} segs ({files[0].stem}-{files[-1].stem}): "
        f"lataccel {m.lataccel_cost:.3f}  jerk {m.jerk_cost:.3f}  total {m.total_cost:.3f}")
  print(f"saved results/runs/{name}")


if __name__ == "__main__":
  main()
