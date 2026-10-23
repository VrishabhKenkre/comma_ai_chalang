"""PID + preview feedforward with gains tuned by CMA-ES: the strong honest baseline.

  u = ff_scale(v) * (sum_j w_j * mean(target over window j) + k_roll * mean(roll[t+2..t+4]))
      + kp*e + ki*sum(e) + kd*(e - e_prev),          e = target - current
  ff_scale(v) = 1 + k_v * (v_ego - 20) / 10

Windows over [target_t, plan_{t+1}, plan_{t+2}, ...]: now, t+1, t+2..4, t+5..9,
t+10..19 (the t+2..4 window is the one controllers/pid_ff.py uses, so the CMA-ES
start point is exactly that controller). All 10 parameters are global, tuned once by CMA-ES on dev segments
10000-10199 (scripts/tune_pid_ff_cma.py, results/tuning/pid_ff_cma.json). The
test set was not used.
"""
import numpy as np

from . import BaseController
from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH

WINDOWS = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20)]   # [start, end) offsets from "now"
NAMES = ["kp", "ki", "kd", "w0", "w1", "w2", "w3", "w4", "k_roll", "k_v"]
# set by scripts/tune_pid_ff_cma.py (CMA-ES mean; results/tuning/pid_ff_cma.json)
DEFAULTS = dict(kp=0.14245871491923007, ki=0.08621254762243447, kd=0.038699808266642065, w0=-0.133536958636313, w1=0.16417850296982114, w2=0.3011588657801114, w3=0.07831414475476076, w4=0.023688158265805823, k_roll=-0.4828942962712415, k_v=0.019560043167553755)


class Controller(BaseController):
  def __init__(self, **params):
    p = {**DEFAULTS, **params}
    self.kp, self.ki, self.kd = p["kp"], p["ki"], p["kd"]
    self.w = [p[f"w{j}"] for j in range(len(WINDOWS))]
    self.k_roll, self.k_v = p["k_roll"], p["k_v"]
    self.integral = 0.0
    self.prev_error = 0.0
    self.step = CONTEXT_LENGTH

  @staticmethod
  def _window_mean(seq, a, b):
    part = seq[a:b]
    return part.mean() if len(part) else seq[-1]          # plan runs out at the end of a segment

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    targets = np.r_[target_lataccel, future_plan.lataccel]
    rolls = np.r_[state.roll_lataccel, future_plan.roll_lataccel]
    ff = sum(w * self._window_mean(targets, a, b) for w, (a, b) in zip(self.w, WINDOWS))
    ff += self.k_roll * self._window_mean(rolls, 2, 5)
    ff *= 1.0 + self.k_v * (state.v_ego - 20.0) / 10.0

    error = target_lataccel - current_lataccel
    if self.step >= CONTROL_START_IDX:   # integrate only while engaged
      self.integral += error
    d_error = error - self.prev_error
    self.prev_error = error
    self.step += 1
    return ff + self.kp * error + self.ki * self.integral + self.kd * d_error
