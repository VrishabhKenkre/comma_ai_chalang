"""Small ARX model of the car, one per speed bin.

  y_t = sum_{i=1..na} a_i y_{t-i} + sum_{j=0..nb} b_j u_{t-j} + sum_{l=0..nc} c_l roll_{t-l} + d

y = lateral acceleration, u = steer command, roll = roll_lataccel. Index t is
the simulator's step index: u_t is applied at step t and the simulator returns
y_t in the same step (the controller sees y_t at step t+1 as current_lataccel).
"""
import json
from dataclasses import dataclass, asdict

import numpy as np

SPEED_EDGES = [0, 5, 10, 15, 20, 25, 30, 35, 100]   # m/s, 8 bins


def speed_bin(v, edges=SPEED_EDGES):
  return int(np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2))


@dataclass
class ARX:
  a: list   # na coefficients on y_{t-1..t-na}
  b: list   # nb+1 coefficients on u_{t..t-nb}
  c: list   # nc+1 coefficients on roll_{t..t-nc}
  d: float  # bias

  @property
  def na(self): return len(self.a)

  @property
  def nb(self): return len(self.b) - 1

  @property
  def nc(self): return len(self.c) - 1

  def dc_gain(self):
    """Steady-state dy/du and dy/droll."""
    s = 1.0 - sum(self.a)
    return sum(self.b) / s, sum(self.c) / s


def regressors(y, u, roll, t, na, nb, nc):
  """Row [y_{t-1..t-na}, u_{t..t-nb}, roll_{t..t-nc}, 1] for each index in t."""
  cols = [y[t - i] for i in range(1, na + 1)]
  cols += [u[t - j] for j in range(nb + 1)]
  cols += [roll[t - l] for l in range(nc + 1)]
  cols.append(np.ones(len(t)))
  return np.column_stack(cols)


def fit(Y, X, na, nb, nc, ridge=1e-6):
  """Least squares (tiny ridge for conditioning)."""
  w = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)
  return ARX(a=w[:na].tolist(), b=w[na:na + nb + 1].tolist(),
             c=w[na + nb + 1:na + nb + nc + 2].tolist(), d=float(w[-1]))


def free_run(m, y0, u, roll, t0, n):
  """Simulate the ARX from measured history before t0 for n steps (its own y fed back)."""
  y = np.array(y0, dtype=float).copy()
  for t in range(t0, t0 + n):
    y[t] = (sum(m.a[i - 1] * y[t - i] for i in range(1, m.na + 1))
            + sum(m.b[j] * u[t - j] for j in range(m.nb + 1))
            + sum(m.c[l] * roll[t - l] for l in range(m.nc + 1)) + m.d)
  return y[t0:t0 + n]


def save(models, path, meta=None, edges=SPEED_EDGES):
  json.dump({"speed_edges": list(edges), "models": [asdict(m) for m in models], **(meta or {})},
            open(path, "w"), indent=1)


def load(path):
  raw = json.load(open(path))
  assert raw["speed_edges"] == SPEED_EDGES
  return [ARX(**m) for m in raw["models"]]


def load_with_edges(path):
  """Models and the speed breakpoints they were fitted for (tuned breakpoints may differ)."""
  raw = json.load(open(path))
  return [ARX(**m) for m in raw["models"]], list(raw["speed_edges"])
