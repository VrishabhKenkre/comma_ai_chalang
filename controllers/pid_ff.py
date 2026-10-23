"""PID + feedforward: the honest reference controller.

  u = u_ff + kp*e + ki*sum(e) + kd*(e - e_prev),   e = target - current

Feedforward is a static inverse model of the car, applied to a short preview
window of the plan (the car responds with ~3 steps of delay, so steering for
the target a few steps ahead is what matters):

  u_ff = k_r * mean(target[t+s : t+s+L]) + k_roll * mean(roll[t+s : t+s+L])

All gains are global (same for every segment), tuned once on dev segments
10000-10499 with scripts/tune_pid_ff.py. The test set (0-4999) was not used.
"""
import numpy as np

from . import BaseController
from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH

# tuned by scripts/tune_pid_ff.py (see results/tuning/pid_ff.json)
DEFAULTS = dict(kp=0.14058, ki=0.077216, kd=-0.0055522, k_r=0.41264, k_roll=-0.37319, s=2, L=3)


class Controller(BaseController):
  def __init__(self, **params):
    p = {**DEFAULTS, **params}
    self.kp, self.ki, self.kd = p["kp"], p["ki"], p["kd"]
    self.k_r, self.k_roll = p["k_r"], p["k_roll"]
    self.s, self.L = int(p["s"]), int(p["L"])
    self.integral = 0.0
    self.prev_error = 0.0
    self.step = CONTEXT_LENGTH  # update() is first called at step 20

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # window over [target_t, plan_{t+1}, plan_{t+2}, ...]; index 0 is "now"
    targets = np.r_[target_lataccel, future_plan.lataccel]
    rolls = np.r_[state.roll_lataccel, future_plan.roll_lataccel]
    win = slice(self.s, self.s + self.L)
    r_ff = targets[win].mean() if len(targets[win]) else target_lataccel
    roll_ff = rolls[win].mean() if len(rolls[win]) else state.roll_lataccel
    u_ff = self.k_r * r_ff + self.k_roll * roll_ff

    error = target_lataccel - current_lataccel
    if self.step >= CONTROL_START_IDX:   # integrate only while engaged
      self.integral += error
    d_error = error - self.prev_error
    self.prev_error = error
    self.step += 1
    return u_ff + self.kp * error + self.ki * self.integral + self.kd * d_error
