"""CMA-ES tuning of the preview MPC + disturbance observer (controllers/mpc.py) on dev segments.

Parameters (14): tracking weight (jerk weight fixed at 1, so only the ratio matters), steer-rate
weight, horizon length, the observer's three noise variances (q_d, q_k, r) and the speed-scheduling
breakpoints. The breakpoints are parametrized by 8 logits that set each speed bin's share of the
sysID data (floored at F_MIN), and the breakpoints are the matching speed quantiles, so every
candidate is valid. (A first attempt with the 7 breakpoints as raw parameters wasted about half of
each generation on invalid bins; results/tuning/mpc_cma_attempt1.log.) New bins mean refitted
ARX models (least squares, same data and lags as results/sysid/arx_models_round1.json:
sysID segments 15000-15299, PID+FF and MPC closed-loop runs).

Objective: mean total cost on dev segments 10500-10999 (never used for tuning before).
Overfitting check: dev 11000-11499, where the best evaluated point, the CMA-ES mean and the
hand-tuned start are compared; the better of the first two is kept. Test segments are never used.

  python scripts/tune_mpc_cma.py
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ctrl import arx  # noqa: E402
from ctrl import evaluate as ev  # noqa: E402
from ctrl.arx_controller import ARXController  # noqa: E402
from tinyphysics import TinyPhysicsSimulator, CONTROL_START_IDX  # noqa: E402

OBJ_SEGS = range(10500, 11000)
CHECK_SEGS = range(11000, 11500)
NA, NB, NC = 1, 6, 6
F_MIN = 0.015                              # minimum share of the sysID data per speed bin
CACHE = ROOT / "cache_arx_dataset.npz"
# name, start, CMA std (in the search coordinates), decoding
NAMES = ["log10_w_track", "log10_w_du", "H", "log10_q_d", "log10_q_k", "log10_r"] + [f"bin_logit{i}" for i in range(8)]
STDS = [0.3, 0.3, 8.0, 0.5, 0.5, 0.5] + [0.5] * 8


def start_point(V):
  """Hand-tuned settings; bin logits that reproduce the current breakpoints 5, 10, ..., 35 m/s."""
  share = np.bincount([arx.speed_bin(v) for v in V], minlength=8) / len(V)
  logits = np.log((share - F_MIN) / (1 - 8 * F_MIN))
  return [0.0, np.log10(4.0), 40.0, np.log10(3e-3), np.log10(1e-5), np.log10(2e-3)] + list(logits - logits.mean())


def build_dataset():
  """Regressors / targets / speeds of the round-0 + round-1 sysID runs (as sysid_closed_loop.py)."""
  if CACHE.exists():
    z = np.load(CACHE)
    return z["X"], z["Y"], z["V"]
  import sysid as S0
  from ctrl.torch_model import TinyPhysicsTorch
  model = TinyPhysicsTorch().to(S0.DEV)
  sets = [np.load(ROOT / "cache_sysid_data.npz")["D_fit"], np.load(ROOT / "cache_sysid_round1.npz")["fit"]]
  t_all = np.arange(CONTROL_START_IDX, S0.SEG_LEN)
  X, Y, V = [], [], []
  for D in sets:
    E = S0.expected(model, *S0.contexts(D, t_all))
    x, y, v = S0.arx_rows(D, E, t_all, NA, NB, NC)
    X.append(x); Y.append(y); V.append(v)
  X, Y, V = np.vstack(X), np.concatenate(Y), np.concatenate(V)
  np.savez_compressed(CACHE, X=X, Y=Y, V=V)
  return X, Y, V


def refit(edges, X, Y, V):
  """ARX per speed bin for the given breakpoints."""
  B = np.array([arx.speed_bin(v, edges) for v in V])
  return [arx.fit(Y[B == b], X[B == b], NA, NB, NC) for b in range(len(edges) - 1)]


def decode(x, V):
  p = dict(zip(NAMES, x))
  z = np.array([p[f"bin_logit{i}"] for i in range(8)])
  share = F_MIN + (1 - 8 * F_MIN) * np.exp(z - z.max()) / np.exp(z - z.max()).sum()
  inner = np.quantile(V, np.cumsum(share)[:-1])
  return dict(w_track=10 ** p["log10_w_track"], w_du=10 ** p["log10_w_du"], H=int(np.clip(round(p["H"]), 8, 50)),
              kf=dict(q_d=10 ** p["log10_q_d"], q_k=10 ** p["log10_q_k"], r=10 ** p["log10_r"]),
              edges=[0.0] + [float(e) for e in inner] + [100.0])


def _rollout(job):
  cfg, models, i = job
  c = ARXController(solver="osqp", observer=True, H=cfg["H"], w_du=cfg["w_du"], w_track=cfg["w_track"],
                    kf=cfg["kf"], models=models, speed_edges=cfg["edges"])
  return TinyPhysicsSimulator(ev._model, ev.seg_path(i), controller=c).rollout()["total_cost"]


def mean_costs(pool, cfgs, data, segs):
  fitted = [refit(c["edges"], *data) for c in cfgs]
  jobs = [(c, m, i) for c, m in zip(cfgs, fitted) for i in segs]
  costs = np.array(pool.map(_rollout, jobs, chunksize=4)).reshape(len(cfgs), len(segs))
  return costs.mean(1).tolist()


def main():
  pool = Pool(32, initializer=ev._init_worker)
  data = build_dataset()
  # check: the default breakpoints must reproduce the hand-tuned MPC's model
  ref = arx.load(ROOT / "results" / "sysid" / "arx_models_round1.json")
  mine = refit(arx.SPEED_EDGES, *data)
  dmax = max(np.abs(np.r_[a.a, a.b, a.c, a.d] - np.r_[b.a, b.b, b.c, b.d]).max() for a, b in zip(ref, mine))
  print(f"refit with default breakpoints vs arx_models_round1.json: max coefficient difference {dmax:.2e}", flush=True)
  START = start_point(data[2])
  print("start decodes to breakpoints", np.round(decode(START, data[2])["edges"], 3), flush=True)
  import cma                                    # its own RNG, created after the pool exists
  es = cma.CMAEvolutionStrategy(START, 1.0, dict(CMA_stds=STDS, popsize=12, maxiter=int(sys.argv[1]) if len(sys.argv) > 1 else 40,
                                                 seed=12345, verbose=-9))
  log, best = [], (np.inf, None)
  t0 = time.time()
  while not es.stop():
    X = es.ask()
    f = mean_costs(pool, [decode(x, data[2]) for x in X], data, OBJ_SEGS)
    es.tell(X, f)
    i = int(np.argmin(f))
    if f[i] < best[0]:
      best = (f[i], np.array(X[i]))
    log.append(dict(gen=es.countiter, best_gen=f[i], median_gen=float(np.median(f)), best_so_far=best[0]))
    print(f"gen {es.countiter:3d}  best {f[i]:.3f}  median {np.median(f):.3f}  best so far {best[0]:.3f}  "
          f"({time.time() - t0:.0f}s)", flush=True)
  cands = {"best_evaluated": best[1], "cma_mean": np.array(es.result.xfavorite), "hand_tuned_start": np.array(START)}
  check = dict(zip(cands, mean_costs(pool, [decode(x, data[2]) for x in cands.values()], data, CHECK_SEGS)))
  obj = dict(zip(cands, mean_costs(pool, [decode(x, data[2]) for x in cands.values()], data, OBJ_SEGS)))
  chosen = min(["best_evaluated", "cma_mean"], key=lambda k: check[k])
  print("objective set:", {k: round(v, 3) for k, v in obj.items()})
  print("check set:    ", {k: round(v, 3) for k, v in check.items()}, "-> chosen", chosen)
  cfg = decode(cands[chosen], data[2])
  arx.save(refit(cfg["edges"], *data), ROOT / "results" / "sysid" / "arx_models_cma.json", edges=cfg["edges"],
           meta=dict(note="ARX refit for the CMA-ES speed breakpoints (scripts/tune_mpc_cma.py)",
                     na=NA, nb=NB, nc=NC, fit_segs="15000-15299 (round 0 + round 1 data)"))
  out = dict(chosen=chosen, config=cfg, objective_segs="10500-10999", check_segs="11000-11499",
             objective_mean_total=obj, check_mean_total=check, refit_check_max_coef_diff=float(dmax),
             candidates={k: decode(v, data[2]) for k, v in cands.items()}, raw={k: v.tolist() for k, v in cands.items()}, log=log)
  json.dump(out, open(ROOT / "results" / "tuning" / "mpc_cma.json", "w"), indent=1)
  pool.close()


if __name__ == "__main__":
  main()
