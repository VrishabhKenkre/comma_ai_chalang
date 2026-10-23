"""Piecewise-affine (PWA) model of TinyPhysics' expected lataccel: a greedy model tree.

Regions are axis-aligned boxes (convex) in the partition features
  z_t = (v_ego_t, a_ego_t, roll_t, y_{t-1}, u_t)
and each region has its own affine ARX
  y_t = theta_m . phi_t,   phi_t = [y_{t-1}, u_t..u_{t-6}, roll_t..roll_{t-6}, a_t, 1]
(the same lags as the MPC's ARX, plus a_ego as a known exogenous input).

Growth is best-first: repeatedly split the leaf whose best split (feature x quantile
threshold) most reduces the summed least-squares error of the affine fits in its
children, until M leaves. Fitted by least squares (tiny ridge) to the network's
expected lataccel, like ctrl/arx.py.
"""
import json
from dataclasses import dataclass, field

import numpy as np

Z_NAMES = ["v_ego", "a_ego", "roll", "y_prev", "u"]
NB, NC = 6, 6
N_PHI = 1 + (NB + 1) + (NC + 1) + 2          # y_{t-1}, u lags, roll lags, a_t, 1


def features(y, u, roll, v, a, t):
  """z [n, 5] and phi [n, N_PHI] for time indices t (arrays over one segment)."""
  z = np.column_stack([v[t], a[t], roll[t], y[t - 1], u[t]])
  phi = np.column_stack([y[t - 1]] + [u[t - j] for j in range(NB + 1)] + [roll[t - l] for l in range(NC + 1)]
                        + [a[t], np.ones(len(t))])
  return z, phi


def _lsq(X, Y, ridge=1e-6):
  A = X.T @ X + ridge * np.eye(X.shape[1])
  w = np.linalg.solve(A, X.T @ Y)
  r = Y - X @ w
  return w, float(r @ r)


@dataclass
class Leaf:
  lo: list                 # box lower bounds per z feature (-inf if open)
  hi: list                 # box upper bounds
  theta: list = field(default_factory=list)


class PWAModel:
  def __init__(self, leaves):
    self.leaves = leaves
    self.lo = np.array([l.lo for l in leaves]); self.hi = np.array([l.hi for l in leaves])
    self.theta = np.array([l.theta for l in leaves])

  def region(self, z):
    """Index of the box containing each row of z (boxes tile the space; ties go to the first)."""
    z = np.atleast_2d(z)
    inside = np.all((z[:, None, :] >= self.lo[None]) & (z[:, None, :] < self.hi[None]), axis=2)
    return inside.argmax(1)

  def predict(self, z, phi):
    return np.einsum("ij,ij->i", phi, self.theta[self.region(z)])

  def save(self, path, meta=None):
    json.dump({"z_names": Z_NAMES, "leaves": [l.__dict__ for l in self.leaves], **(meta or {})},
              open(path, "w"), indent=1, default=float)

  @staticmethod
  def load(path):
    raw = json.load(open(path))
    return PWAModel([Leaf(**l) for l in raw["leaves"]])


def fit_tree(Z, PHI, Y, n_leaves, min_leaf=2000, quantiles=np.linspace(0.1, 0.9, 9), split_features=None):
  """Greedy best-first model tree. Returns (PWAModel, log of splits)."""
  feats = range(Z.shape[1]) if split_features is None else split_features
  inf = float("inf")
  root = dict(idx=np.arange(len(Y)), lo=[-inf] * Z.shape[1], hi=[inf] * Z.shape[1])
  root["w"], root["sse"] = _lsq(PHI, Y)

  def best_split(node):
    idx, best = node["idx"], None
    for f in feats:
      zf = Z[idx, f]
      for th in np.unique(np.quantile(zf, quantiles)):
        left = zf < th
        nl = int(left.sum())
        if nl < min_leaf or len(idx) - nl < min_leaf:
          continue
        wl, sl = _lsq(PHI[idx[left]], Y[idx[left]])
        wr, sr = _lsq(PHI[idx[~left]], Y[idx[~left]])
        gain = node["sse"] - sl - sr
        if best is None or gain > best[0]:
          best = (gain, f, float(th), left, wl, sl, wr, sr)
    return best

  leaves, log = [root], []
  root["split"] = best_split(root)
  while len(leaves) < n_leaves:
    cand = [(n["split"][0], i) for i, n in enumerate(leaves) if n["split"] is not None]
    if not cand:
      break
    _, i = max(cand)
    node = leaves.pop(i)
    gain, f, th, left, wl, sl, wr, sr = node["split"]
    L = dict(idx=node["idx"][left], lo=list(node["lo"]), hi=list(node["hi"]), w=wl, sse=sl)
    R = dict(idx=node["idx"][~left], lo=list(node["lo"]), hi=list(node["hi"]), w=wr, sse=sr)
    L["hi"][f] = th
    R["lo"][f] = th
    for n in (L, R):
      n["split"] = best_split(n)
      leaves.append(n)
    log.append(dict(feature=Z_NAMES[f], threshold=th, sse_reduction=gain))
  model = PWAModel([Leaf(lo=n["lo"], hi=n["hi"], theta=n["w"].tolist()) for n in leaves])
  return model, log
