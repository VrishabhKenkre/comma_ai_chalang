"""Iterative closed-loop re-identification ("identification for control").

The first ARX (scripts/sysid.py) was fitted on PID+FF-like steering. The MPC
steers more aggressively, and there the ARX is less accurate (see the
"previous model" column of results/sysid/SYSID.md), which forces a large
steer-rate penalty. Each round here:
  1. runs the current MPC (+ dither) on the sysID segments 15000-15399,
     half with steer-rate weight 0.5 and half with 1.0,
  2. refits the ARX (same order as round 0) on all data collected so far,
  3. reports fit error on held-out segments from every regime.
Test segments 0-4999 are never used.

  python scripts/sysid_closed_loop.py --rounds 2
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import sysid as S0  # noqa: E402  (round-0 helpers)
from ctrl import arx  # noqa: E402
from ctrl.arx_controller import ARXController  # noqa: E402
from ctrl.evaluate import seg_path  # noqa: E402
from ctrl.torch_model import TinyPhysicsTorch  # noqa: E402
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, CONTEXT_LENGTH  # noqa: E402

OUT = ROOT / "results" / "sysid"


class Dithered:
  """Adds a held-random dither (own Generator) to an ARXController once engaged.
  The applied steer is written back into the controller's u history, because
  the controller must know what it actually applied."""
  def __init__(self, inner, seed, amp=0.1):
    self.inner, self.rng, self.amp = inner, np.random.default_rng(seed), amp
    self.hold, self.dither, self.step = 0, 0.0, CONTEXT_LENGTH

  def update(self, *args, **kw):
    u = self.inner.update(*args, **kw)
    t, self.step = self.step, self.step + 1
    if t < CONTROL_START_IDX:
      return u
    if self.hold == 0:
      self.dither, self.hold = self.rng.uniform(-self.amp, self.amp), int(self.rng.integers(2, 8))
    self.hold -= 1
    u = float(np.clip(u + self.dither, -2, 2))
    self.inner.u[-1] = u
    return u


def _collect_one(args):
  i, models_path = args
  w_du = 0.5 if i % 2 == 0 else 1.0
  ctrl = Dithered(ARXController(solver="osqp", observer=True, w_du=w_du, models_path=models_path), seed=i)
  model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
  sim = TinyPhysicsSimulator(model, seg_path(i), controller=ctrl, debug=False)
  sim.rollout()
  st = np.array(sim.state_history)
  return np.column_stack([sim.action_history, st[:, 0], st[:, 1], st[:, 2], sim.current_lataccel_history])[:S0.SEG_LEN]


def collect(segs, models_path):
  with Pool(32) as pool:
    return np.stack(pool.map(_collect_one, [(i, models_path) for i in segs], chunksize=4))


def fit_models(datasets, model, na, nb, nc):
  t_all = np.arange(CONTROL_START_IDX, S0.SEG_LEN)
  X, Y, V = [], [], []
  for D in datasets:
    E = S0.expected(model, *S0.contexts(D, t_all))
    x, y, v = S0.arx_rows(D, E, t_all, na, nb, nc)
    X.append(x); Y.append(y); V.append(v)
  X, Y, V = np.vstack(X), np.concatenate(Y), np.concatenate(V)
  B = np.array([arx.speed_bin(v) for v in V])
  return [arx.fit(Y[B == b], X[B == b], na, nb, nc) for b in range(len(arx.SPEED_EDGES) - 1)]


def fit_error(models, D, model):
  """One-step and 20-step free-run RMSE vs the expected-value model on held-out data D."""
  t_all = np.arange(CONTROL_START_IDX, S0.SEG_LEN)
  m0 = models[0]
  E = S0.expected(model, *S0.contexts(D, t_all))
  X, Y, V = S0.arx_rows(D, E, t_all, m0.na, m0.nb, m0.nc)
  W = np.array([np.r_[m.a, m.b, m.c, m.d] for m in models])
  one = np.sqrt(np.mean((np.einsum("ij,ij->i", X, W[[arx.speed_bin(v) for v in V]]) - Y) ** 2))
  err = []
  for t0 in np.arange(120, S0.SEG_LEN - 20, 20):
    FR = S0.model_free_run(model, D, t0, 20)
    for s in range(len(D)):
      m = models[arx.speed_bin(D[s, t0, 2])]
      err.append(arx.free_run(m, D[s, :, 4], D[s, :, 0], D[s, :, 1], t0, 20) - FR[s])
  return float(one), float(np.sqrt(np.mean(np.square(err))))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--rounds", type=int, default=2)
  args = ap.parse_args()
  model = TinyPhysicsTorch().to(S0.DEV)
  cache = np.load(ROOT / "cache_sysid_data.npz")
  meta0 = json.load(open(OUT / "arx_models.json"))
  na, nb, nc = meta0["na"], meta0["nb"], meta0["nc"]
  fit_sets, hold_sets = [cache["D_fit"]], {"round0 (PID+FF)": cache["D_hold"]}
  models_path = OUT / "arx_models_round0.json"
  if not models_path.exists():
    models_path.write_text((OUT / "arx_models.json").read_text())
  report = []
  for r in range(1, args.rounds + 1):
    print(f"round {r}: collecting with MPC using {models_path.name} ...", flush=True)
    fit_sets.append(collect(S0.FIT_SEGS, str(models_path)))
    hold_sets[f"round{r} (MPC)"] = collect(S0.HOLDOUT_SEGS, str(models_path))
    models = fit_models(fit_sets, model, na, nb, nc)
    errs = {name: fit_error(models, D, model) for name, D in hold_sets.items()}
    prev = arx.load(models_path)
    errs_prev = {name: fit_error(prev, D, model) for name, D in hold_sets.items()}
    for name in hold_sets:
      print(f"  held-out {name:16s}: previous model one-step/free-run {errs_prev[name][0]:.4f}/{errs_prev[name][1]:.4f}"
            f"  ->  refit {errs[name][0]:.4f}/{errs[name][1]:.4f}", flush=True)
    models_path = OUT / f"arx_models_round{r}.json"
    arx.save(models, models_path, meta=dict(na=na, nb=nb, nc=nc, round=r, fit_segs="15000-15299",
                                             data="round0 PID+FF + rounds 1..r MPC closed loop"))
    report.append(dict(round=r, refit={k: dict(one_step=v[0], free_run=v[1]) for k, v in errs.items()},
                       previous={k: dict(one_step=v[0], free_run=v[1]) for k, v in errs_prev.items()}))
    np.savez_compressed(ROOT / f"cache_sysid_round{r}.npz", fit=fit_sets[-1], hold=hold_sets[f"round{r} (MPC)"])
  json.dump(report, open(OUT / "closed_loop_rounds.json", "w"), indent=1)


if __name__ == "__main__":
  main()
