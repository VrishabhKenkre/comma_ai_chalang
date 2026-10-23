"""Tune the 10 global PID+FF parameters (controllers/pid_ff_cma.py) with CMA-ES on dev segments.

Objective = mean total_cost on dev segments 10000-10199 (same tuning set as every other
method). CMA-ES starts at the Nelder-Mead PID+FF (controllers/pid_ff.py). At the end the
best evaluated point and the CMA-ES mean are compared on dev 10200-10499 (not used by
the search) and the better one is kept. Test segments are never used.

The worker pool is created before CMA-ES initializes its own RNG; the simulator reseeds
numpy's RNG from the segment path at every rollout, so tuning randomness cannot reach it.

  python scripts/tune_pid_ff_cma.py
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctrl import evaluate as ev  # noqa: E402
from controllers.pid_ff_cma import DEFAULTS, NAMES, Controller  # noqa: E402
from tinyphysics import TinyPhysicsSimulator  # noqa: E402

TUNE_SEGS = range(10000, 10200)
CHECK_SEGS = range(10200, 10500)
STDS = dict(kp=0.05, ki=0.03, kd=0.03, w0=0.1, w1=0.1, w2=0.1, w3=0.1, w4=0.1, k_roll=0.1, k_v=0.1)


def _rollout(job):
  params, i = job
  return TinyPhysicsSimulator(ev._model, ev.seg_path(i), controller=Controller(**params)).rollout()["total_cost"]


def mean_cost(pool, list_of_params, segs):
  jobs = [(p, i) for p in list_of_params for i in segs]
  costs = np.array(pool.map(_rollout, jobs, chunksize=8)).reshape(len(list_of_params), len(segs))
  return costs.mean(1)


def main():
  pool = Pool(32, initializer=ev._init_worker)
  import cma                                    # imported after the pool exists
  x0 = np.array([DEFAULTS[n] for n in NAMES])
  es = cma.CMAEvolutionStrategy(x0, 1.0, dict(CMA_stds=[STDS[n] for n in NAMES], popsize=12,
                                              maxiter=60, seed=12345, verbose=-9))
  log, best = [], (np.inf, None)
  t0 = time.time()
  while not es.stop():
    X = es.ask()
    f = mean_cost(pool, [dict(zip(NAMES, x)) for x in X], TUNE_SEGS)
    es.tell(X, f.tolist())
    i = int(np.argmin(f))
    if f[i] < best[0]:
      best = (float(f[i]), X[i].copy())
    log.append(dict(gen=es.countiter, best_gen=float(f[i]), median_gen=float(np.median(f)), best_so_far=best[0]))
    print(f"gen {es.countiter:3d}  best {f[i]:.3f}  median {np.median(f):.3f}  best so far {best[0]:.3f}  "
          f"({time.time() - t0:.0f}s)", flush=True)
  candidates = {"best_evaluated": dict(zip(NAMES, best[1])), "cma_mean": dict(zip(NAMES, es.result.xfavorite)),
                "start (pid_ff)": dict(DEFAULTS)}
  check = dict(zip(candidates, mean_cost(pool, list(candidates.values()), CHECK_SEGS)))
  chosen = min(["best_evaluated", "cma_mean"], key=lambda k: check[k])
  print("check on 10200-10499:", {k: round(v, 3) for k, v in check.items()}, "-> chosen", chosen)
  out = dict(chosen=chosen, params={k: float(v) for k, v in candidates[chosen].items()},
             tune_segs="10000-10199", check_segs="10200-10499", check_mean_total=check,
             candidates={k: {kk: float(vv) for kk, vv in v.items()} for k, v in candidates.items()}, log=log)
  json.dump(out, open(ROOT / "results" / "tuning" / "pid_ff_cma.json", "w"), indent=1)
  pool.close()


if __name__ == "__main__":
  main()
