"""Receding-horizon iLQR through the TinyPhysics model (expected-value dynamics).

Settings chosen on dev segments 10000-10199 (results/tuning/ilqr_gpu_sweep):
H=12 with 2 iterations per step scored 36.49 vs 36.74 for 1 iteration, within
noise, so we keep the cheaper one (timing in results/timing.json).

`shifts`: when the warm start is saturated (steer at +-2 or predicted lataccel at
the +-5 token limit) the model's gradient vanishes and iLQR can lock up at the
limit. We then also try the warm start shifted by these steer offsets. This was
added after the first test run showed two blow-ups; it was designed and checked
on non-test data only (stress set results/stress/stress_segs.txt and dev 10000-10199).
"""
from ctrl.ilqr import ILQRController

PARAMS = dict(H=12, iters=1, w_du=2.0, shifts=(-0.3, -0.1, 0.1, 0.3))


class Controller(ILQRController):
  def __init__(self):
    super().__init__(**PARAMS)
