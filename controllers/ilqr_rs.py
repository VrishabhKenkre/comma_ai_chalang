"""Risk-sensitive iLQR: the final iLQR plus var_weight * sigma_k^2 per horizon step, where sigma_k
is the std of TinyPhysics' predicted lataccel distribution (exact over the bins). var_weight = 1.5
makes each step's planned cost the exact one-step expectation of the score (research branch, step 1)."""
from ctrl.ilqr import ILQRController
from controllers.ilqr import PARAMS as BASE

PARAMS = dict(BASE, var_weight=1.5)


class Controller(ILQRController):
  def __init__(self):
    super().__init__(**PARAMS)
