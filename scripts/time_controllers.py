"""Measure controller compute time per step (update() only, simulator excluded).

Runs each controller single-process on a few dev segments and records the mean,
p99 and max wall-clock time of update() over the controlled steps after the
engagement step, and separately the engagement step itself (step 100, where the
iLQR does extra iterations). The first segment is warm-up and not counted.
Results are merged into results/timing.json. Run on an otherwise idle machine.

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

SEGS = range(10000, 10011)   # first one is warm-up


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
    times, first = [], []
    for k, i in enumerate(SEGS):
      c = Timed(Controller())
      TinyPhysicsSimulator(model, seg_path(i), controller=c).rollout()
      ctrl_steps = c.times[CONTROL_START_IDX - 20:]   # controlled steps only
      if k > 0:                                      # first segment = JIT / cache warm-up
        first.append(ctrl_steps[0])                  # the engagement step (step 100)
        times += ctrl_steps[1:]
    times, first = np.array(times), np.array(first)
    timing[name] = dict(ms_per_step=1e3 * times.mean(), ms_p99=1e3 * np.percentile(times, 99),
                        ms_max=1e3 * times.max(), ms_engage_step_mean=1e3 * first.mean(),
                        n_steps=len(times), cpu=platform.processor() or platform.machine())
    t = timing[name]
    print(f"{name}: {t['ms_per_step']:.3f} ms/step (p99 {t['ms_p99']:.3f}, max {t['ms_max']:.3f}; "
          f"engagement step {t['ms_engage_step_mean']:.3f})")
  json.dump(timing, open(out_file, "w"), indent=1)


if __name__ == "__main__":
  main()
