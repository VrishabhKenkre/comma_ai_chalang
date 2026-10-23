"""Preview MPC as in mpc.py, solved with hand-written ADMM + cached Riccati (TinyMPC style)."""
from ctrl.arx_controller import ARXController


class Controller(ARXController):
  def __init__(self):
    super().__init__(solver="admm", observer=True)
