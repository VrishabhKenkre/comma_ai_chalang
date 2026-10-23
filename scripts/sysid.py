"""System identification of TinyPhysics, binned by speed.

1. Collect closed-loop data: tuned PID+FF plus a small held-random dither on
   sysID segments 15000-15299 (fit) and 15300-15399 (held out). These are
   outside the test set 0-4999. The dither uses its own numpy Generator, so the
   simulator's global RNG is untouched.
2. Linearize: at operating points along those runs, central finite differences
   of the EXPECTED lataccel  E[y_t | context] = sum_i p_i * bin_i  with respect
   to each past steer u_{t-k}, past lataccel y_{t-k} and roll_{t-k}.
   Averaged per speed bin, this is the local linear model of the full 20-lag
   transformer.
3. Fit a small ARX per speed bin by least squares, target = E[y_t | context].
   Order = smallest na+nb+nc within 1% of the best held-out free-run RMSE.
4. Report fit error on held-out segments: one-step RMSE and 20-step free-run
   RMSE against the expected-value model run with the same inputs.

Outputs go to results/sysid/.
"""
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctrl import arx  # noqa: E402
from ctrl.torch_model import TinyPhysicsTorch  # noqa: E402
from ctrl.evaluate import seg_path  # noqa: E402
from controllers import pid_ff  # noqa: E402
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX  # noqa: E402

FIT_SEGS = range(15000, 15300)
HOLDOUT_SEGS = range(15300, 15400)
OUT = ROOT / "results" / "sysid"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_LAGS = 20
SEG_LEN = 596        # shortest segment in 15000-15399 has 598 rows


class DitherPIDFF(pid_ff.Controller):
  """PID+FF plus a random step dither (+-0.1 steer, held 2-7 steps) for excitation."""
  def __init__(self, seed):
    super().__init__()
    self.rng = np.random.default_rng(seed)
    self.hold, self.dither = 0, 0.0

  def update(self, *args, **kw):
    if self.hold == 0:
      self.dither, self.hold = self.rng.uniform(-0.1, 0.1), int(self.rng.integers(2, 8))
    self.hold -= 1
    return super().update(*args, **kw) + self.dither


def _collect_one(i):
  model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
  sim = TinyPhysicsSimulator(model, seg_path(i), controller=DitherPIDFF(seed=i), debug=False)
  sim.rollout()
  st = np.array(sim.state_history)
  return np.column_stack([sim.action_history, st[:, 0], st[:, 1], st[:, 2], sim.current_lataccel_history])


def collect(segs):
  """Returns array [n_segs, T, 5] with columns (u, roll, v, a, y); segments have
  576-602 rows, so all runs are truncated to T = SEG_LEN steps."""
  with Pool(32) as pool:
    runs = pool.map(_collect_one, segs, chunksize=4)
  return np.stack([r[:SEG_LEN] for r in runs])


def contexts(D, t):
  """Model inputs for predicting y_t: states rows t-19..t, past y_{t-20..t-1}."""
  idx = t[:, None] + np.arange(-N_LAGS + 1, 1)[None, :]
  return D[:, idx, :4], D[:, idx - 1, 4]   # [S, T, 20, 4], [S, T, 20]


@torch.no_grad()
def expected(model, states, past, chunk=65536):
  s = torch.tensor(states.reshape(-1, N_LAGS, 4), dtype=torch.float32)
  p = torch.tensor(past.reshape(-1, N_LAGS), dtype=torch.float32)
  out = [model.expected(s[i:i + chunk].to(DEV), p[i:i + chunk].to(DEV)).cpu()
         for i in range(0, len(s), chunk)]
  return torch.cat(out).numpy().reshape(states.shape[:-2])


def jacobians(model, states, past, du=0.05, dr=0.05, dy=0.03):
  """Central differences of E[y_t] w.r.t. u_{t-k}, roll_{t-k} (k=0..19) and y_{t-k} (k=1..20).

  Lataccel enters as a token, so the y perturbation is measured on decoded bin
  values (the actual change seen by the model), not the nominal dy.
  """
  states = states.reshape(-1, N_LAGS, 4)
  past = past.reshape(-1, N_LAGS)
  bins = model.bins.cpu().numpy()
  snap = lambda y: bins[np.digitize(np.clip(y, -5, 5), bins, right=True)]
  J = {"u": [], "roll": [], "y": []}
  for k in range(N_LAGS):
    pos = N_LAGS - 1 - k          # state row of time t-k
    for name, col, h in [("u", 0, du), ("roll", 1, dr)]:
      sp, sm = states.copy(), states.copy()
      sp[:, pos, col] += h
      sm[:, pos, col] -= h
      J[name].append((expected(model, sp, past) - expected(model, sm, past)) / (2 * h))
    pp, pm = past.copy(), past.copy()   # token of y_{t-1-k} sits at position 19-k
    pp[:, pos] = snap(past[:, pos] + dy)
    pm[:, pos] = snap(past[:, pos] - dy)
    J["y"].append((expected(model, states, pp) - expected(model, states, pm)) / (pp[:, pos] - pm[:, pos]))
  return {k: np.stack(v, axis=1) for k, v in J.items()}   # each [N, 20]


@torch.no_grad()
def model_free_run(model, D, t0, n):
  """Expected-value model run from true history before t0 with its own y fed back."""
  y = D[:, :, 4].copy()
  for t in range(t0, t0 + n):
    s, p = contexts(np.concatenate([D[:, :, :4], y[:, :, None]], axis=2), np.array([t]))
    y[:, t] = expected(model, s, p)[:, 0]
  return y[:, t0:t0 + n]


def arx_rows(D, E, t_idx, na, nb, nc):
  X, Y, V = [], [], []
  for s in range(len(D)):
    u, roll, v, y = D[s, :, 0], D[s, :, 1], D[s, :, 2], D[s, :, 4]
    X.append(arx.regressors(y, u, roll, t_idx, na, nb, nc))
    Y.append(E[s])
    V.append(v[t_idx])
  return np.vstack(X), np.concatenate(Y), np.concatenate(V)


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  model = TinyPhysicsTorch().to(DEV)
  print("collecting closed-loop data ...", flush=True)
  D_fit, D_hold = collect(FIT_SEGS), collect(HOLDOUT_SEGS)
  t_all = np.arange(CONTROL_START_IDX, SEG_LEN)

  # ---- 2. linearization at operating points (every 4th controlled step) ----
  t_op = np.arange(CONTROL_START_IDX, SEG_LEN, 4)
  s, p = contexts(D_fit, t_op)
  print(f"finite differences at {s.shape[0] * s.shape[1]} operating points ...", flush=True)
  J = jacobians(model, s, p)
  v_op = D_fit[:, t_op, 2].reshape(-1)
  bins_op = np.array([arx.speed_bin(v) for v in v_op])
  n_bins = len(arx.SPEED_EDGES) - 1
  lin = []
  for b in range(n_bins):
    sel = bins_op == b
    lin.append({k: J[k][sel].mean(0).tolist() for k in J} | {"n_points": int(sel.sum())})

  # ---- 3. ARX fits per speed bin, target = expected one-step lataccel ----
  E_fit = expected(model, *contexts(D_fit, t_all))
  E_hold = expected(model, *contexts(D_hold, t_all))
  # free-run windows on held-out data: 20 steps from every 20th step
  starts = np.arange(120, SEG_LEN - 20, 20)
  FR = {t0: model_free_run(model, D_hold, t0, 20) for t0 in starts}

  def evaluate_models(models, na, nb, nc):
    """One-step and free-run RMSE on held-out segs, models chosen by speed at each step."""
    X, Y, V = arx_rows(D_hold, E_hold, t_all, na, nb, nc)
    W = np.array([np.r_[m.a, m.b, m.c, m.d] for m in models])
    pred = np.einsum("ij,ij->i", X, W[[arx.speed_bin(v) for v in V]])
    one_step = np.sqrt(np.mean((pred - Y) ** 2))
    err = []
    for t0 in starts:
      for s_ in range(len(D_hold)):
        u, roll, y = D_hold[s_, :, 0], D_hold[s_, :, 1], D_hold[s_, :, 4]
        m = models[arx.speed_bin(D_hold[s_, t0, 2])]
        err.append(arx.free_run(m, y, u, roll, t0, 20) - FR[t0][s_])
    return one_step, np.sqrt(np.mean(np.square(err)))

  results = []
  for na in [1, 2, 3]:
    for nb in [2, 4, 6, 7, 8, 10]:
      for nc in [0, 1, 3, 6, 8]:
        X, Y, V = arx_rows(D_fit, E_fit, t_all, na, nb, nc)
        B = np.array([arx.speed_bin(v) for v in V])
        models = [arx.fit(Y[B == b], X[B == b], na, nb, nc) for b in range(n_bins)]
        one, fr = evaluate_models(models, na, nb, nc)
        results.append(dict(na=na, nb=nb, nc=nc, one_step_rmse=one, free_run_rmse=fr, models=models))
        print(f"ARX na={na} nb={nb} nc={nc}: one-step RMSE {one:.4f}  20-step free-run RMSE {fr:.4f}", flush=True)

  # ---- linearization ARX: bin-averaged Jacobians truncated to 8 lags, only bias fitted ----
  K = 8
  lin_models = []
  X, Y, V = arx_rows(D_fit, E_fit, t_all, K, K - 1, K - 1)
  B = np.array([arx.speed_bin(v) for v in V])
  for b in range(n_bins):
    a, bb, c = lin[b]["y"][:K], lin[b]["u"][:K], lin[b]["roll"][:K]
    resid = Y[B == b] - X[B == b][:, :-1] @ np.r_[a, bb, c]
    lin_models.append(arx.ARX(a=a, b=bb, c=c, d=float(resid.mean())))
  lin_one, lin_fr = evaluate_models(lin_models, K, K - 1, K - 1)
  print(f"Jacobian linearization (8 lags): one-step {lin_one:.4f}  free-run {lin_fr:.4f}")

  # references: the full model's own one-step error vs the sampled y (noise floor),
  # and a "hold last value" predictor
  y_hold = D_hold[:, t_all, 4]
  model_vs_sampled = np.sqrt(np.mean((E_hold - y_hold) ** 2))
  persist_one = np.sqrt(np.mean((D_hold[:, t_all - 1, 4] - E_hold) ** 2))
  persist_fr = np.sqrt(np.mean([(D_hold[:, t0 - 1, 4][:, None] - FR[t0]) ** 2 for t0 in starts]))

  # parsimony: smallest model (na + nb + nc) whose free-run RMSE is within 1% of the best
  best_fr = min(r["free_run_rmse"] for r in results)
  best = min((r for r in results if r["free_run_rmse"] <= 1.01 * best_fr),
             key=lambda r: (r["na"] + r["nb"] + r["nc"], r["free_run_rmse"]))
  arx.save(best["models"], OUT / "arx_models.json",
           meta=dict(na=best["na"], nb=best["nb"], nc=best["nc"], fit_segs="15000-15299",
                     one_step_rmse=best["one_step_rmse"], free_run_rmse=best["free_run_rmse"]))
  arx.save(lin_models, OUT / "jacobian_arx_models.json", meta=dict(lags=K))
  summary = dict(
    fit_segs="15000-15299", holdout_segs="15300-15399",
    grid=[{k: v for k, v in r.items() if k != "models"} for r in results],
    chosen={k: v for k, v in best.items() if k != "models"},
    jacobian_arx=dict(lags=K, one_step_rmse=lin_one, free_run_rmse=lin_fr),
    references=dict(full_model_vs_sampled_one_step=model_vs_sampled,
                    hold_last_value_one_step=persist_one, hold_last_value_free_run=persist_fr),
    linearization_by_speed_bin=lin,
    y_std_holdout=float(E_hold.std()),
  )
  json.dump(summary, open(OUT / "sysid_summary.json", "w"), indent=1)
  np.savez_compressed(ROOT / "cache_sysid_data.npz", D_fit=D_fit, D_hold=D_hold)
  print("chosen:", summary["chosen"], "\nreferences:", summary["references"])


if __name__ == "__main__":
  main()
