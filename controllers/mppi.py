"""MPPI with moment-matched expected cost (research branch, step 1). Own seeded torch RNG.
Smooth perturbations (offset + ramp over the horizon): with independent per-step noise the
averaged plan was jittery and the controller failed on dev (results/variance/VARIANCE.md)."""
from ctrl.batch_wrap import OneSegment
from ctrl.variance_ctrl import BatchMPPI

PARAMS = dict(var_weight=1.5, noise="ramp", n_samples=128, noise_std=0.1, lam=0.01, iters=2)


class Controller(OneSegment):
  def __init__(self):
    super().__init__(BatchMPPI(1, **PARAMS))
