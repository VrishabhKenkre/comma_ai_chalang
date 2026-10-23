"""Infinite-horizon LQR on the same ARX model and observer: no preview, no constraints (ablation)."""
from ctrl.arx_controller import ARXController


class Controller(ARXController):
  def __init__(self):
    super().__init__(solver="lqr", observer=True)
