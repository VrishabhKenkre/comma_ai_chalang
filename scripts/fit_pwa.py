"""Step 2a: fit piecewise-affine models of TinyPhysics' expected lataccel and report fit error.

Data: the system-ID runs already cached by scripts/sysid.py and sysid_closed_loop.py
(segments 15000-15299 to fit, 15300-15399 held out; PID+FF-driven round 0 and
MPC-driven rounds 1-2). Targets: the network's expected lataccel E[y_t].

Models (all with the same regressors phi = [y_{t-1}, u_t..t-6, roll_t..t-6, a_t, 1]):
  global ARX      one affine model
  speed-binned    one per 5 m/s speed bin (like the MPC's ARX)
  PWA tree, M     model tree with M box regions over (v, a, roll, y_{t-1}, u_t)
  PWA tree (y,u)  model tree splitting only on y_{t-1} and u_t (the variables a controller
                  changes, so regions there make the MPC genuinely hybrid)
Error vs the network: one-step RMSE and 20-step free-run RMSE (model fed back its own
output, compared with the network's expected-value free run on the same inputs).

Writes results/pwa/pwa_fit.json, results/pwa/PWA.md and the chosen model's JSON.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import sysid as S0  # noqa: E402
from ctrl import arx  # noqa: E402
from ctrl.pwa import PWAModel, Leaf, fit_tree, features, _lsq, Z_NAMES  # noqa: E402
from ctrl.torch_model import TinyPhysicsTorch  # noqa: E402

OUT = ROOT / "results" / "pwa"
T0 = 100


def dataset(D, model):
  t = np.arange(T0, S0.SEG_LEN)
  E = S0.expected(model, *S0.contexts(D, t))
  Z, P, Y = [], [], []
  for s in range(len(D)):
    u, roll, v, a, y = (D[s, :, j] for j in range(5))
    z, phi = features(y, u, roll, v, a, t)
    Z.append(z); P.append(phi); Y.append(E[s])
  return np.vstack(Z), np.vstack(P), np.concatenate(Y)


def free_run_error(pwa, D, model, starts):
  """20-step free run of the PWA vs the network's expected-value free run."""
  err = []
  for t0 in starts:
    ref = S0.model_free_run(model, D, t0, 20)
    yhat = D[:, :, 4].copy()
    for t in range(t0, t0 + 20):
      z, phi = [], []
      for s in range(len(D)):
        zz, pp = features(yhat[s], D[s, :, 0], D[s, :, 1], D[s, :, 2], D[s, :, 3], np.array([t]))
        z.append(zz[0]); phi.append(pp[0])
      yhat[:, t] = pwa.predict(np.array(z), np.array(phi))
    err.append(yhat[:, t0:t0 + 20] - ref)
  return float(np.sqrt(np.mean(np.square(err))))


def speed_binned(Z, P, Y):
  leaves = []
  for b in range(len(arx.SPEED_EDGES) - 1):
    lo = [-np.inf] * len(Z_NAMES); hi = [np.inf] * len(Z_NAMES)
    lo[0], hi[0] = arx.SPEED_EDGES[b], arx.SPEED_EDGES[b + 1]
    sel = (Z[:, 0] >= lo[0]) & (Z[:, 0] < hi[0])
    w, _ = _lsq(P[sel], Y[sel])
    leaves.append(Leaf(lo=lo, hi=hi, theta=w.tolist()))
  leaves[0].lo[0] = -np.inf
  return PWAModel(leaves)


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  model = TinyPhysicsTorch().to(S0.DEV)
  c0 = np.load(ROOT / "cache_sysid_data.npz")
  rounds = [np.load(ROOT / f"cache_sysid_round{r}.npz") for r in (1, 2)]
  fit_sets = [c0["D_fit"]] + [c["fit"] for c in rounds]
  hold_sets = {"PID+FF regime": c0["D_hold"], "MPC regime": np.concatenate([c["hold"] for c in rounds])}
  Zf, Pf, Yf = (np.concatenate(x) for x in zip(*[dataset(D, model) for D in fit_sets]))
  holds = {k: dataset(D, model) for k, D in hold_sets.items()}
  print(f"fit points {len(Yf)}; held-out " + ", ".join(f"{k} {len(v[2])}" for k, v in holds.items()), flush=True)

  w, _ = _lsq(Pf, Yf)
  inf = [np.inf] * len(Z_NAMES)
  candidates = {"global ARX (1 region)": (PWAModel([Leaf(lo=[-np.inf] * 5, hi=inf, theta=w.tolist())]), []),
                "speed-binned ARX (8 regions)": (speed_binned(Zf, Pf, Yf), [])}
  for M in (2, 4, 8, 16):
    candidates[f"PWA tree, {M} regions"] = fit_tree(Zf, Pf, Yf, M)
  for M in (4, 8):
    candidates[f"PWA tree on (y_prev, u) only, {M} regions"] = fit_tree(Zf, Pf, Yf, M, split_features=[3, 4])

  FILE_NAMES = {"global ARX (1 region)": "global_arx.json", "speed-binned ARX (8 regions)": "speed_binned_8.json",
                **{f"PWA tree, {M} regions": f"pwa_all_{M}.json" for M in (2, 4, 8, 16)},
                **{f"PWA tree on (y_prev, u) only, {M} regions": f"pwa_yu_{M}.json" for M in (4, 8)}}
  starts = np.arange(120, S0.SEG_LEN - 20, 40)
  report = {}
  for name, (pwa, log) in candidates.items():
    row = {"splits": log}
    for hk, (Z, P, Y) in holds.items():
      row[hk] = dict(one_step_rmse=float(np.sqrt(np.mean((pwa.predict(Z, P) - Y) ** 2))),
                     free_run_rmse=free_run_error(pwa, hold_sets[hk], model, starts))
    report[name] = row
    print(f"{name:40s} " + "  ".join(f"{hk}: 1-step {v['one_step_rmse']:.4f} free-run {v['free_run_rmse']:.4f}"
                                      for hk, v in row.items() if hk != "splits"), flush=True)
    pwa.save(OUT / FILE_NAMES[name])
  ref = S0.expected(model, *S0.contexts(hold_sets["PID+FF regime"], np.arange(T0, S0.SEG_LEN)))
  sampled = hold_sets["PID+FF regime"][:, T0:S0.SEG_LEN, 4]
  report["_noise_floor_one_step"] = float(np.sqrt(np.mean((ref - sampled) ** 2)))
  json.dump(report, open(OUT / "pwa_fit.json", "w"), indent=1)


if __name__ == "__main__":
  main()
