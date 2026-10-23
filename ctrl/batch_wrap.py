"""Official-API wrapper: run any batched controller (update_batch) with a batch of one segment."""
from controllers import BaseController


class OneSegment(BaseController):
  def __init__(self, batch_controller):
    self.core = batch_controller

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    return float(self.core.update_batch([target_lataccel], [current_lataccel], [state], [future_plan], [True])[0])
