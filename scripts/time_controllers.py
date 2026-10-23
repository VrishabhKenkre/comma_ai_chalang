"""Measure controller compute time per step (update() only, simulator excluded).

Runs each controller single-process on a few dev segments and records the mean
wall-clock time of update() over all controlled steps. Results are merged into
results/timing.json. Run on an otherwise idle machine.

  python scripts/time_controllers.py pid pid_ff mpc
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from ctrl.evaluate import seg_path  # noqa: E402
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX  # noqa: E402

SEGS = range(10000, 10010)


class Timed:
  def __init__(self, inner):
    self.inner, self.times = inner, []

  def update(self, *args, **kw):
    t0 = time.perf_counter()
    out = self.inner.update(*args, **kw)
    self.times.append(time.perf_counter() - t0)
    return out


def main():
  model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
  out_file = ROOT / "results" / "timing.json"
  timing = json.load(open(out_file)) if out_file.exists() else {}
  for name in sys.argv[1:]:
    Controller = importlib.import_module(f"controllers.{name}").Controller
    times = []
    for i in SEGS:
      c = Timed(Controller())
      TinyPhysicsSimulator(model, seg_path(i), controller=c).rollout()
      times += c.times[CONTROL_START_IDX - 20:]   # controlled steps only
    times = np.array(times[50:])                   # drop first calls (JIT / cache warm-up)
    timing[name] = dict(ms_per_step=1e3 * times.mean(), ms_p99=1e3 * np.percentile(times, 99),
                        n_steps=len(times), cpu=platform.processor() or platform.machine())
    print(f"{name}: {timing[name]['ms_per_step']:.3f} ms/step (p99 {timing[name]['ms_p99']:.3f})")
  json.dump(timing, open(out_file, "w"), indent=1)


if __name__ == "__main__":
  main()
