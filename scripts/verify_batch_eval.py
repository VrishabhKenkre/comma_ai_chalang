"""Check that ctrl/batch_eval.py reproduces the official simulator exactly.

Runs controllers that were already evaluated with scripts/run_eval.py (official
tinyphysics.run_rollout, one process per segment) through the lockstep
evaluator and compares per-segment costs. Also checks that batched ONNX calls
give bit-identical logits to batch-1 calls.

  python scripts/verify_batch_eval.py pid:10000:500 pid_ff:10000:500 mpc:10000:500
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctrl.batch_eval import BatchSim  # noqa: E402


def onnx_bitwise_check(sim, n=400):
  rng = np.random.default_rng(0)   # only used to build test inputs here
  states = rng.normal(0, 1, (n, 20, 4)).astype(np.float32)
  tokens = rng.integers(0, 1024, (n, 20)).astype(np.int64)
  sess = sim.models[0].ort_session
  single = np.concatenate([sess.run(None, {"states": states[i:i + 1], "tokens": tokens[i:i + 1]})[0] for i in range(n)])
  return bool(np.array_equal(single, sim._logits(states, tokens)))


def main():
  report = {}
  for spec in sys.argv[1:]:
    name, start, n = spec.split(":")
    segs = range(int(start), int(start) + int(n))
    runs = [pd.read_csv(f, dtype={"seg": str}, float_precision="round_trip") for f in (ROOT / "results" / "runs").glob(f"{name}_[0-9]*.csv")]
    official = pd.concat(runs).drop_duplicates("seg")
    official = official[official.seg.isin([f"{i:05d}" for i in segs])]
    assert len(official) == len(segs), f"no official run covers {spec}"
    Controller = importlib.import_module(f"controllers.{name}").Controller
    sim = BatchSim(segs)
    if "onnx_bitwise" not in report:
      report["onnx_bitwise"] = onnx_bitwise_check(sim)
      print("batched ONNX logits bit-identical to batch-1:", report["onnx_bitwise"], flush=True)
    batch = pd.DataFrame(sim.run([Controller() for _ in segs]))
    m = official.merge(batch, on="seg", suffixes=("_official", "_batch"))
    diffs = {k: float(np.abs(m[f"{k}_official"] - m[f"{k}_batch"]).max())
             for k in ("lataccel_cost", "jerk_cost", "total_cost")}
    exact = int((m["total_cost_official"] == m["total_cost_batch"]).sum())
    report[spec] = dict(n=len(m), max_abs_diff=diffs, n_bitwise_equal_total=exact,
                        mean_official=float(m.total_cost_official.mean()), mean_batch=float(m.total_cost_batch.mean()))
    print(spec, report[spec], flush=True)
  json.dump(report, open(ROOT / "results" / "batch_eval_verification.json", "w"), indent=1)


if __name__ == "__main__":
  main()
