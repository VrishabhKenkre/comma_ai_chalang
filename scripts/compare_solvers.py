"""Compare OSQP and the hand-written ADMM on the exact same QPs from real runs.

A diagnostic controller solves every step with both solvers (same x0, affine
terms, references), applies the OSQP answer (so the trajectory is identical),
and records the first-move difference, the full-plan difference, objective
values, ADMM iterations and solve times. Run on dev segments.

  python scripts/compare_solvers.py
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctrl import arx  # noqa: E402
from ctrl.arx_controller import ARXController  # noqa: E402
from ctrl.evaluate import seg_path  # noqa: E402
from ctrl.mpc import MPCProblem  # noqa: E402
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator  # noqa: E402

SEGS = range(10000, 10020)
# (rho, du_max): tuned rate limit 0.3 never binds, so 0.03 is added to exercise the constraints
CONFIGS = [(0.3, 0.3), (1.0, 0.3), (3.0, 0.3), (1.0, 0.03), (3.0, 0.03)]


def qp_cost(p, delta, X, r):
  """Objective 0.5 x'Qx + q'x + 0.5 R d^2 summed over the horizon (same as both solvers)."""
  return float(sum(0.5 * x @ p.Q @ x - rk * x[0] for x, rk in zip(X, r)) + 0.5 * p.R * delta @ delta)


class Both(ARXController):
  def __init__(self, rho, du_max):
    super().__init__(solver="osqp", rho=rho, du_max=du_max)
    self.log = []

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # capture the QP the OSQP path builds by wrapping solve_osqp for this call
    p = self.problems[arx.speed_bin(state.v_ego)]
    orig = p.solve_osqp

    def wrapped(x0, f, r):
      t0 = time.perf_counter(); d_o, X_o = orig(x0, f, r); t_o = time.perf_counter() - t0
      t0 = time.perf_counter(); d_a, X_a, self.warm, it = p.solve_admm(x0, f, r, warm=self.warm)
      t_a = time.perf_counter() - t0
      self.log.append(dict(d0=abs(d_o[0] - d_a[0]), dmax=float(np.abs(d_o - d_a).max()),
                           obj_o=qp_cost(p, d_o, X_o, r), obj_a=qp_cost(p, d_a, X_a, r),
                           iters=it, t_osqp=t_o, t_admm=t_a,
                           active=bool(np.any(np.abs(d_o) > p.du_max - 1e-4) or np.any(np.abs(X_o[:, p.ny]) > 2 - 1e-4))))
      return d_o, X_o
    p.solve_osqp = wrapped
    try:
      return super().update(target_lataccel, current_lataccel, state, future_plan)
    finally:
      p.solve_osqp = orig


def run(args):
  cfg, i = args
  c = Both(*cfg)
  TinyPhysicsSimulator(TinyPhysicsModel("./models/tinyphysics.onnx", debug=False), seg_path(i), controller=c).rollout()
  return cfg, c.log


def main():
  # compile the numba kernel once before forking
  MPCProblem(arx.load("results/sysid/arx_models_round1.json")[5]).solve_admm(np.zeros(8), np.zeros((40, 8)), np.zeros(40))
  with Pool(32) as pool:
    out = pool.map(run, [(cfg, i) for cfg in CONFIGS for i in SEGS])
  summary = {}
  for cfg in CONFIGS:
    L = [row for c, log in out if c == cfg for row in log]
    d0 = np.array([x["d0"] for x in L]); it = np.array([x["iters"] for x in L])
    gap = np.array([x["obj_a"] - x["obj_o"] for x in L])
    key = f"rho={cfg[0]},du_max={cfg[1]}"
    summary[key] = dict(n_qps=len(L), first_move_absdiff_max=float(d0.max()), first_move_absdiff_mean=float(d0.mean()),
                        plan_absdiff_max=float(max(x["dmax"] for x in L)), abs_objective_gap_max=float(np.abs(gap).max()), objective_gap_min=float(gap.min()),
                        admm_iters_mean=float(it.mean()), admm_iters_p99=float(np.percentile(it, 99)), admm_iters_max=int(it.max()),
                        hit_iter_cap=int((it >= 500).sum()), frac_qps_with_active_constraint=float(np.mean([x["active"] for x in L])),
                        osqp_ms_mean=1e3 * float(np.mean([x["t_osqp"] for x in L])),
                        admm_ms_mean=1e3 * float(np.mean([x["t_admm"] for x in L])))
    s = summary[key]
    print(f"{key:18s} QPs={s['n_qps']}  |du0| max {s['first_move_absdiff_max']:.1e} mean {s['first_move_absdiff_mean']:.1e}  "
          f"plan max {s['plan_absdiff_max']:.1e}  |obj gap| max {s['abs_objective_gap_max']:.1e}  "
          f"iters mean {s['admm_iters_mean']:.1f} p99 {s['admm_iters_p99']:.0f} max {s['admm_iters_max']} (cap hits {s['hit_iter_cap']})  "
          f"active {100 * s['frac_qps_with_active_constraint']:.2f}%  osqp {s['osqp_ms_mean']:.2f} ms  admm {s['admm_ms_mean']:.3f} ms")
  json.dump(summary, open(ROOT / "results" / "solver_comparison.json", "w"), indent=1)


if __name__ == "__main__":
  main()
