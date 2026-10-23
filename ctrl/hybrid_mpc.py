"""Hybrid MPC on a piecewise-affine model, solved with Graphs of Convex Sets (step 2b).

Model: ctrl/pwa.py. Region m is a box in z_k = (v_k, a_k, roll_k, y_{k-1}, u_k) with affine
dynamics y_k = theta_m . phi_k. Over the horizon v, a and roll are known from the plan,
so at step k only regions whose (v, a, roll) bounds contain the planned values are possible;
the remaining choice depends on y_{k-1} and u_k, which the controller decides.

Graph (Marcucci & Tedrake style shortest path in a graph of convex sets):
  vertex (k, m): variables w = (x_k [8], u_k) in a box:
      x_k = [y_{k-1}, y_{k-2}, u_{k-1}, ..., u_{k-6}]
      region box of m on (y_{k-1}, u_k), |u_k| <= 2, |u_k - u_{k-1}| <= du_max.
      Inside mode m the output is affine, y_k = theta_m . phi(x_k, u_k) + c_km (known exogenous
      terms + observer offset), so the stage cost
      0.5*(y_k - r_k)^2 + (y_k - y_{k-1})^2 + w_du*(u_k - u_{k-1})^2   (the score / 100)
      is a sum of squares in w.
  edges (k, m) -> (k+1, m'): x_{k+1} = shift(x_k, u_k, y_k) with y_k from mode m's dynamics
  source = the measured x_0; target after step H-1.
Solve: convex relaxation (lower bound on the hybrid optimum), then candidate mode sequences,
each solved as a convex QP with the sequence fixed; the best is applied. Candidates: Drake's
randomized rounding of the flows, "simulation rounding" (simulate the PWA model under the
relaxation's flow-weighted steer and take the modes it visits), and the previous step's
solution shifted by one step.
Relaxation gap = (rounded - relaxed) / rounded.

`solve_miqp` solves the same problem exactly as a convex-hull mixed-integer QP with SCIP
(cvxpy), for the cross-check on a few segments.

The observer is the output-offset Kalman filter of the linear MPC (ctrl/observer.py).
"""
import time

import numpy as np

from controllers import BaseController
from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH

from .observer import BiasRollKF
from .pwa import PWAModel, NB, NC

NX = 8
U_MAX, Y_MAX, HIST_MAX = 2.0, 5.0, 10.0
WARMUP_FF = (0.55, -0.6)
BIG = 50.0                                    # finite stand-in for open box sides


class HybridProblem:
  """Holds the data of one hybrid MPC problem and solves it (GCS or MIQP)."""

  def __init__(self, pwa, x0, refs, exo, roll_hist, offset_terms, w_du, du_max):
    """exo [H, 3] = (v, a, roll) for steps 0..H-1; roll_hist = roll_{-6..-1};
    offset_terms [H, M] = observer output offset already scaled per region."""
    self.pwa, self.x0, self.r, self.exo = pwa, x0, refs, exo
    self.H, self.M = len(refs), len(pwa.leaves)
    self.w_du, self.du_max = w_du, du_max
    rolls = np.r_[roll_hist, exo[:, 2]]                          # roll_{-6} .. roll_{H-1}
    self.modes, self.const = [], np.zeros((self.H, self.M))
    for k in range(self.H):
      v, a, roll = exo[k]
      ok = [m for m in range(self.M)
            if all(pwa.lo[m, j] <= val < pwa.hi[m, j] for j, val in ((0, v), (1, a), (2, roll)))]
      self.modes.append(ok)
      for m in ok:
        th = pwa.theta[m]
        lag_rolls = rolls[6 + k - np.arange(NC + 1)]               # roll_k .. roll_{k-6}
        self.const[k, m] = th[8:15] @ lag_rolls + th[15] * a + th[16] + offset_terms[k, m]

  def _y_map(self, k, m):
    """y_k = a . w + c for w = (x_k [8], u_k) inside mode m."""
    th = self.pwa.theta[m]
    a = np.zeros(NX + 1)
    a[0] = th[0]                                               # y_{k-1} = x[0]
    a[NX] = th[1]                                              # u_k
    for j in range(1, NB + 1):
      a[1 + j] = th[1 + j]                                     # u_{k-j} = x[1 + j]
    return a, self.const[k, m]

  def _box(self, k, m):
    """Inequalities A w <= b for vertex (k, m), w = (x [8], u)."""
    lo, hi = self.pwa.lo[m], self.pwa.hi[m]
    A, b = [], []
    def le(coef, val):
      A.append(coef); b.append(val)
    e = np.eye(NX + 1)
    le(e[0], min(hi[3], HIST_MAX)); le(-e[0], -max(lo[3], -HIST_MAX))    # region on y_{k-1}
    le(e[NX], min(hi[4], U_MAX)); le(-e[NX], -max(lo[4], -U_MAX))        # region on u_k, |u| <= 2
    le(e[NX] - e[2], self.du_max); le(e[2] - e[NX], self.du_max)         # rate vs u_{k-1} = x[2]
    # history components (y_{k-2}, past steers) only need to be bounded for GCS; they are fixed by
    # x0 or by earlier steps, and warmup steer guesses can exceed 2, so use a generous bound
    le(e[1], HIST_MAX); le(-e[1], HIST_MAX)
    for i in range(2, NX):
      le(e[i], HIST_MAX); le(-e[i], HIST_MAX)
    return np.array(A), np.array(b)

  def _stage_cost(self, k, m):
    """Stage cost as 0.5 w'Pw + q'w + c0 (a sum of squares, so >= 0 as GCS requires)."""
    a, c = self._y_map(k, m)
    e = np.eye(NX + 1)
    # residuals: sqrt(0.5)*(y - r), (y - y_prev), sqrt(w_du)*(u - u_prev); each = g . w + h
    res = [(np.sqrt(0.5) * a, np.sqrt(0.5) * (c - self.r[k])),
           (a - e[0], c),
           (np.sqrt(self.w_du) * (e[NX] - e[2]), 0.0)]
    P = sum(2 * np.outer(g, g) for g, _ in res)
    q = sum(2 * h * g for g, h in res)
    c0 = sum(h * h for _, h in res)
    return P, q, c0

  # ---------------------------------------------------------------- GCS
  def simulate_modes(self, u_seq):
    """Mode sequence the PWA model actually visits when u_seq is applied from x0."""
    x, modes = np.array(self.x0, float), []
    for k in range(self.H):
      u = float(np.clip(u_seq[k], x[2] - self.du_max, x[2] + self.du_max))
      v, a, roll = self.exo[k]
      m = int(self.pwa.region(np.array([[v, a, roll, x[0], u]]))[0])
      if m not in self.modes[k]:
        return None
      ak, c = self._y_map(k, m)
      y = ak @ np.r_[x, u] + c
      modes.append(m)
      x = np.r_[y, x[0], u, x[2:NX - 1]]
    return modes

  def solve_gcs(self, max_rounded_paths=5, seed=0, extra_mode_seqs=()):
    from pydrake.geometry.optimization import (GraphOfConvexSets, GraphOfConvexSetsOptions, HPolyhedron, Point)
    from pydrake.solvers import ClarabelSolver
    from pydrake.symbolic import Expression
    g = GraphOfConvexSets()
    src = g.AddVertex(Point(self.x0), "src")
    tgt = g.AddVertex(Point(np.zeros(1)), "tgt")
    V = {}
    for k in range(self.H):
      for m in self.modes[k]:
        A, b = self._box(k, m)
        if not HPolyhedron(A, b).IsBounded() or HPolyhedron(A, b).IsEmpty():
          continue
        P, q, c0 = self._stage_cost(k, m)
        v = g.AddVertex(HPolyhedron(A, b), f"{k},{m}")
        w = v.x()
        v.AddCost(0.5 * w.dot(P @ w) + q.dot(w) + c0)
        V[k, m] = v
    E = {}
    for m in self.modes[0]:
      if (0, m) not in V:
        continue
      e = g.AddEdge(src, V[0, m])
      E["src", m] = e
      for i in range(NX):
        e.AddConstraint(e.xv()[i] == e.xu()[i])
    for k in range(self.H - 1):
      for m in self.modes[k]:
        if (k, m) not in V:
          continue
        a, c = self._y_map(k, m)
        for m2 in self.modes[k + 1]:
          if (k + 1, m2) not in V:
            continue
          e = g.AddEdge(V[k, m], V[k + 1, m2])
          E[k, m, m2] = e
          xu, xv = e.xu(), e.xv()
          e.AddConstraint(xv[0] == a.dot(xu) + c)              # y_k from mode m
          e.AddConstraint(xv[1] == xu[0])                      # y_{k-1}
          e.AddConstraint(xv[2] == xu[NX])                     # u_k
          for j in range(3, NX):
            e.AddConstraint(xv[j] == xu[j - 1])                # u_{k-1} .. u_{k-5}
    for m in self.modes[self.H - 1]:
      if (self.H - 1, m) in V:
        E[m, "tgt"] = g.AddEdge(V[self.H - 1, m], tgt)

    def path_of(modes):
      if modes is None or any((k, m) not in V for k, m in enumerate(modes)):
        return None
      p = [E["src", modes[0]]] + [E[k, modes[k], modes[k + 1]] for k in range(self.H - 1)] + [E[modes[-1], "tgt"]]
      return p
    opts = GraphOfConvexSetsOptions()
    opts.convex_relaxation = True
    opts.max_rounded_paths = 0
    opts.solver = ClarabelSolver()
    t0 = time.perf_counter()
    relaxed = g.SolveShortestPath(src, tgt, opts)
    t_relax = time.perf_counter() - t0
    if not relaxed.is_success():
      return dict(failed="relaxation", status=str(relaxed.get_solution_result()))
    opts.max_rounded_paths = max_rounded_paths
    opts.rounding_seed = seed                                  # Drake's own RNG
    paths = [("random", p) for p in g.SamplePaths(src, tgt, relaxed, opts)]
    # rounding by forward simulation: flow-weighted steer of the relaxation, simulated on the PWA
    def incoming(k):                                           # edges whose head is at step k
      if k == 0:
        return [e for key, e in E.items() if key[0] == "src"]
      return [e for key, e in E.items() if len(key) == 3 and key[0] == k - 1]
    u_rel = np.zeros(self.H)
    for k in range(self.H):
      flows = [(relaxed.GetSolution(e.phi()), e) for e in incoming(k)]
      tot = sum(f for f, _ in flows)
      if tot > 1e-9:
        u_rel[k] = sum(f * relaxed.GetSolution(e.xv())[NX] for f, e in flows) / tot
    for name, modes in [("simulated", self.simulate_modes(u_rel))] + [("given", m) for m in extra_mode_seqs]:
      p = path_of(modes)
      if p is not None:
        paths.append((name, p))
    best = None
    for name, path in paths:
      res = g.SolveConvexRestriction(path, opts)
      if res.is_success() and (best is None or res.get_optimal_cost() < best[0]):
        best = (res.get_optimal_cost(), path, res, name)
    t_total = time.perf_counter() - t0
    if best is None:
      return dict(failed="rounding", n_paths=len(paths), relaxed=relaxed.get_optimal_cost())
    cost, path, res, how = best
    v0 = path[0].v()                                           # first edge: src -> (0, m)
    w0 = res.GetSolution(v0.x())
    u_plan = [float(res.GetSolution(e.v().x())[NX]) for e in path[:-1]]
    modes = [int(e.v().name().split(",")[1]) for e in path[:-1]]
    return dict(u0=float(w0[NX]), u_plan=u_plan, relaxed=relaxed.get_optimal_cost(), rounded=cost, modes=modes,
                t_relax=t_relax, t_total=t_total, n_paths=len(paths), rounding=how)

  # ---------------------------------------------------------------- exact MIQP (cross-check)
  def solve_fixed_modes(self, modes):
    """Convex QP with the mode sequence fixed (cvxpy + Clarabel). Returns (cost, u0)."""
    import cvxpy as cp
    H = self.H
    W = [cp.Variable(NX + 1) for _ in range(H)]
    cons, cost = [W[0][:NX] == self.x0], 0
    for k, m in enumerate(modes):
      A, b = self._box(k, m)
      a, c = self._y_map(k, m)
      cons.append(A @ W[k] <= b)
      yk = a @ W[k] + c
      cost += 0.5 * cp.square(yk - self.r[k]) + cp.square(yk - W[k][0]) + self.w_du * cp.square(W[k][NX] - W[k][2])
      if k + 1 < H:
        cons += [W[k + 1][0] == yk, W[k + 1][1] == W[k][0], W[k + 1][2] == W[k][NX], W[k + 1][3:NX] == W[k][2:NX - 1]]
    prob = cp.Problem(cp.Minimize(cost), cons)
    prob.solve(solver=cp.CLARABEL)
    return prob.value, (float(W[0].value[NX]) if W[0].value is not None else None)

  def solve_miqp(self, time_limit=60):
    import cvxpy as cp
    H = self.H
    W = [cp.Variable(NX + 1) for _ in range(H)]               # aggregated (x_k, u_k)
    cons, cost, modes = [W[0][:NX] == self.x0], 0, []
    for k in range(H):
      ms = self.modes[k]
      d = cp.Variable(len(ms), boolean=True)
      Wm = [cp.Variable(NX + 1) for _ in ms]                   # convex-hull copies
      Ym = []
      cons += [cp.sum(d) == 1, W[k] == sum(Wm)]
      cost_k = 0
      for i, m in enumerate(ms):
        A, b = self._box(k, m)
        cons.append(A @ Wm[i] <= b * d[i])
        a, c = self._y_map(k, m)
        Ym.append(a @ Wm[i] + c * d[i])
        P, q, c0 = self._stage_cost(k, m)
        # perspective of the mode's stage cost (exact for binary d)
        cost_k += cp.quad_over_lin(cp.hstack([np.sqrt(0.5) * (a @ Wm[i] + (c - self.r[k]) * d[i]),
                                              a @ Wm[i] - Wm[i][0] + c * d[i],
                                              np.sqrt(self.w_du) * (Wm[i][NX] - Wm[i][2])]), d[i])
      cost += cost_k
      modes.append((ms, d))
      if k + 1 < H:
        yk = sum(Ym)
        cons += [W[k + 1][0] == yk, W[k + 1][1] == W[k][0], W[k + 1][2] == W[k][NX],
                 W[k + 1][3:NX] == W[k][2:NX - 1]]
    prob = cp.Problem(cp.Minimize(cost), cons)
    t0 = time.perf_counter()
    prob.solve(solver=cp.SCIP, scip_params={"limits/time": time_limit})
    ok = W[0].value is not None
    return dict(u0=float(W[0].value[NX]) if ok else None, optimal=prob.value, status=prob.status,
                modes=[ms[int(np.argmax(d.value))] for ms, d in modes] if ok else None,
                t=time.perf_counter() - t0)


class HybridMPCController(BaseController):
  """Official-API controller: hybrid MPC on the PWA model, GCS relaxation + rounding."""

  def __init__(self, pwa_path, H=15, w_du=4.0, du_max=0.3, max_rounded_paths=5, kf=None, log=False):
    self.pwa = PWAModel.load(pwa_path)
    self.H, self.w_du, self.du_max, self.paths = H, w_du, du_max, max_rounded_paths
    self.kf = BiasRollKF(**{"q_d": 3e-3, "q_k": 1e-5, **(kf or {})})
    self.step = CONTEXT_LENGTH
    self.y, self.u, self.roll, self.v, self.a = [], [], [], [], []
    self.log = [] if log else None
    self.prev_modes = None
    self.prev_plan = None

  def _plan(self, now, future, n):
    seq = np.r_[now, future][:n]
    return np.r_[seq, np.full(n - len(seq), seq[-1])]

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    t = self.step
    self.step += 1
    self.y.append(current_lataccel); self.roll.append(state.roll_lataccel)
    self.v.append(state.v_ego); self.a.append(state.a_ego)
    H = self.H
    refs = self._plan(target_lataccel, future_plan.lataccel, H)
    exo = np.column_stack([self._plan(state.v_ego, future_plan.v_ego, H), self._plan(state.a_ego, future_plan.a_ego, H),
                           self._plan(state.roll_lataccel, future_plan.roll_lataccel, H)])
    if t < CONTROL_START_IDX:
      self.u.append(WARMUP_FF[0] * refs[2] + WARMUP_FF[1] * exo[2, 2])
      return self.u[-1]
    # observer: residual of y_{t-1} under the region active at t-1
    if t >= CONTROL_START_IDX + NB + 1:
      z_prev = np.array([[self.v[-2], self.a[-2], self.roll[-2], self.y[-2], self.u[-1]]])
      m_prev = self.pwa.region(z_prev)[0]
      phi = np.r_[self.y[-2], [self.u[-1 - j] for j in range(NB + 1)], [self.roll[-2 - l] for l in range(NC + 1)],
                  self.a[-2], 1.0]
      g = 1 - self.pwa.theta[m_prev][0]
      self.kf.update(self.y[-1] - phi @ self.pwa.theta[m_prev], self.roll[-2], g=g)
    offs = np.array([[(1 - self.pwa.theta[m][0]) * self.kf.offset(exo[k, 2]) for m in range(len(self.pwa.leaves))]
                     for k in range(H)])
    x0 = np.r_[self.y[-1], self.y[-2], self.u[-1:-7:-1]]
    prob = HybridProblem(self.pwa, x0, refs, exo, np.array(self.roll[-7:-1]), offs, self.w_du, self.du_max)
    shifted = [] if self.prev_modes is None else [self.prev_modes[1:] + self.prev_modes[-1:]]
    sol = prob.solve_gcs(self.paths, seed=t, extra_mode_seqs=shifted)
    failed = "failed" in sol
    self.prev_modes = None if failed else sol["modes"]
    if not failed:
      u, self.prev_plan = float(np.clip(sol["u0"], -U_MAX, U_MAX)), sol["u_plan"][1:]
    elif self.prev_plan:                                       # on failure: next step of the last plan
      u, self.prev_plan = float(np.clip(self.prev_plan[0], -U_MAX, U_MAX)), self.prev_plan[1:]
    else:                                                      # no plan yet: hold steer
      u = self.u[-1]
    if self.log is not None:
      self.log.append(dict(t=t, failed=sol.get("failed", False), n_modes=[len(m) for m in prob.modes],
                           **{k: sol[k] for k in ("relaxed", "rounded", "t_relax", "t_total", "n_paths", "rounding") if k in sol}))
    self.u.append(u)
    return u
