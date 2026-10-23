"""Hybrid MPC on the 8-region PWA model (regions on lataccel and steer), GCS relaxation + rounding,
horizon 8 (research branch, step 2). Slow: seconds per step."""
from ctrl.hybrid_mpc import HybridMPCController


class Controller(HybridMPCController):
  def __init__(self):
    super().__init__("results/pwa/pwa_yu_8.json", H=8, w_du=4.0)
