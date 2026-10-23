"""Expected-cost lookahead (research branch, step 1): one held steer value chosen on a grid to minimize
the exact expected cost over the next 4 steps. A negative result; kept for timing and reproduction."""
from ctrl.batch_wrap import OneSegment
from ctrl.variance_ctrl import BatchLookahead


class Controller(OneSegment):
  def __init__(self):
    super().__init__(BatchLookahead(1, d=3, w_du=2.0))
