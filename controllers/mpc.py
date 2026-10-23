"""Preview MPC on the speed-scheduled ARX model, OSQP solver, with disturbance observer."""
from ctrl.arx_controller import ARXController


class Controller(ARXController):
  def __init__(self):
    super().__init__(solver="osqp", observer=True)
