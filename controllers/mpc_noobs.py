"""Preview MPC as in mpc.py but without the disturbance observer (ablation)."""
from ctrl.arx_controller import ARXController


class Controller(ARXController):
  def __init__(self):
    super().__init__(solver="osqp", observer=False)
