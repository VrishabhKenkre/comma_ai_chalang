"""Disturbance observer for offset-free MPC.

The ARX regressors (past y, past u, roll) are all measured, so the only
unknowns are slowly varying disturbances. We model them as an OUTPUT offset

  y_t = z_t + d_t + k_t * roll_t

where z follows the ARX model, d is a model-bias offset and k corrects the
roll -> lataccel gain (the part of the road-roll effect the ARX gets wrong).
Substituting into the ARX equation (which uses measured y as regressors), the
offset enters the one-step equation scaled by g = 1 - sum(a):

  residual_t = y_t - yhat_t = g * (d_t + k_t * roll_t) + v_t,   v_t ~ N(0, r)
  d_{t+1} = d_t + w_d,   k_{t+1} = k_t + w_k                     (random walks)

(Putting d directly in the equation instead would make its steady-state effect
d / g, about 20x larger with a ~ 0.95, which made the loop unstable.)

A 2-state Kalman filter with measurement row g * [1, roll_t] estimates (d, k);
the MPC holds them constant over the horizon, removing steady-state offset.
"""
import numpy as np


class BiasRollKF:
  def __init__(self, q_d=1e-4, q_k=1e-5, r=2e-3, p0=(0.01, 0.01)):
    self.Q = np.diag([q_d, q_k])
    self.r = r
    self.theta = np.zeros(2)          # [d, k]
    self.P = np.diag(p0)

  def update(self, residual, roll, g):
    self.P = self.P + self.Q                       # predict (random walk)
    h = g * np.array([1.0, roll])
    S = h @ self.P @ h + self.r
    K = self.P @ h / S
    self.theta = self.theta + K * (residual - h @ self.theta)
    self.P = self.P - np.outer(K, h @ self.P)

  def offset(self, roll):
    """Output offset d + k * roll for a step with known roll."""
    return self.theta[0] + self.theta[1] * roll
