"""Controller built on the identified ARX models: preview MPC (OSQP or ADMM) or LQR.

Information used each step is only what update() receives: target, current
lataccel (= y_{t-1}), state (roll, v_ego, a_ego at t) and the future plan.
The controller also keeps its own history of what it observed and returned.

Before step CONTROL_START_IDX the simulator replaces our output with the
logged steer, which the controller never sees. During that warmup we store a
static inverse-model guess of that steer (fitted offline on non-test data,
see results/sysid) so that the ARX state has sensible u history at engagement.
"""
import numpy as np

from controllers import BaseController
from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH

from . import arx
from .mpc import MPCProblem, U_MAX
from .observer import BiasRollKF

ARX_PATH = "results/sysid/arx_models_round1.json"   # after one round of closed-loop re-identification
WARMUP_FF = (0.55, -0.6)   # logged steer ~ 0.55*target(t+2) - 0.6*roll(t+2), fitted on segs 10000-10999


class ARXController(BaseController):
  # defaults chosen on dev segments 10000-10199 (results/tuning/mpc_sweep_10000-10199.json)
  def __init__(self, solver="osqp", observer=True, H=40, w_du=4.0, du_max=0.3, rho=1.0,
               kf=None, models_path=ARX_PATH):
    self.models = arx.load(models_path)
    self.solver, self.use_observer, self.H = solver, observer, H
    self.problems = [MPCProblem(m, H=H, w_du=w_du, du_max=du_max, rho=rho) for m in self.models]
    self.kf = BiasRollKF(**{"q_d": 3e-3, "q_k": 1e-5, **(kf or {})})
    self.step = CONTEXT_LENGTH
    self.y, self.u, self.roll, self.v = [], [], [], []     # y_{..t-1}, u_{..t-1}, roll_{..t}, v_{..t}
    self.warm = None
    self.last_iters = 0

  # ---- helpers ----
  def _plan(self, now, future, n):
    """[now, future_0, ...] padded with the last value to length n."""
    seq = np.r_[now, future][:n]
    return np.r_[seq, np.full(n - len(seq), seq[-1])]

  def _x0(self, p):
    ys = self.y[::-1][:p.ny]
    us = self.u[::-1][:p.nu]
    return np.r_[ys, np.zeros(p.ny - len(ys)), us, np.zeros(p.nu - len(us))]

  def _arx_predict_last(self, m):
    """ARX prediction of y_{t-1} from measured regressors (for the observer residual).
    History tails at step t: y[-1] = y_{t-1}, u[-1] = u_{t-1}, roll[-1] = roll_t."""
    return (sum(m.a[i - 1] * self.y[-1 - i] for i in range(1, m.na + 1))
            + sum(m.b[j] * self.u[-1 - j] for j in range(m.nb + 1))
            + sum(m.c[l] * self.roll[-2 - l] for l in range(m.nc + 1)) + m.d)

  # ---- main ----
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    t = self.step
    self.step += 1
    self.y.append(current_lataccel)
    self.roll.append(state.roll_lataccel)
    self.v.append(state.v_ego)
    rolls = self._plan(state.roll_lataccel, future_plan.roll_lataccel, self.H)
    refs = self._plan(target_lataccel, future_plan.lataccel, self.H)

    if t < CONTROL_START_IDX:
      # output is ignored by the simulator; remember a guess of the logged steer
      self.u.append(WARMUP_FF[0] * refs[min(2, self.H - 1)] + WARMUP_FF[1] * rolls[min(2, self.H - 1)])
      return self.u[-1]

    # observer: residual of y_{t-1} under the model of step t-1's speed bin,
    # once all regressors are our own applied steers
    m_prev = self.models[arx.speed_bin(self.v[-2])]
    if self.use_observer and t >= CONTROL_START_IDX + m_prev.nb + 1:
      self.kf.update(self.y[-1] - self._arx_predict_last(m_prev), self.roll[-2], g=1 - sum(m_prev.a))

    p = self.problems[arx.speed_bin(state.v_ego)]
    m = p.m
    # known exogenous part of each predicted y_{t+k}: roll terms + bias (+ observer)
    roll_hist = np.r_[self.roll[-1 - m.nc:-1] if m.nc else [], rolls]   # roll_{t-nc..t+H-1}
    f = np.zeros((self.H, p.n))
    for k in range(self.H):
      f[k, 0] = sum(m.c[l] * roll_hist[m.nc + k - l] for l in range(m.nc + 1)) + m.d
      if self.use_observer:   # output offset enters the ARX equation scaled by 1 - sum(a)
        f[k, 0] += (1 - sum(m.a)) * self.kf.offset(rolls[k])
    x0 = self._x0(p)

    if self.solver == "osqp":
      delta, _ = p.solve_osqp(x0, f, refs)
      d0 = delta[0]
    elif self.solver == "admm":
      delta, _, self.warm, self.last_iters = p.solve_admm(x0, f, refs, warm=self.warm)
      d0 = delta[0]
    elif self.solver == "lqr":
      # steady state for the current target with current roll/offset; no preview, no constraints
      r, w = refs[0], f[0, 0]
      u_ss = (r * (1 - sum(m.a)) - w) / sum(m.b)
      x_ss = np.r_[np.full(p.ny, r), np.full(p.nu, u_ss)]
      d0 = -p.lqr_K @ (x0 - x_ss)
    else:
      raise ValueError(self.solver)

    u = float(np.clip(self.u[-1] + d0, -U_MAX, U_MAX))   # simulator clips to [-2, 2] too
    self.u.append(u)
    return u
