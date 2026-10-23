"""Lockstep evaluation of many segments at once, reproducing tinyphysics exactly.

Why: the iLQR controller runs a 1M-parameter transformer ~100 times per step.
Run one segment per process, 32 processes thrash the CPU caches; batching all
segments through the controller on the GPU is much faster.

How it stays identical to the official simulator (tinyphysics.py):
  * data loading is the official TinyPhysicsSimulator.get_data, and the step
    logic below mirrors TinyPhysicsSimulator.step / control_step / sim_step /
    compute_cost line by line;
  * each segment gets its own np.random.RandomState seeded exactly like
    reset() seeds the global RNG (md5 of the path string), and samples with
    the same .choice call, so the random streams are the official ones;
  * the simulator model is the official TinyPhysicsModel session (1 thread);
    rows are batched, which we checked gives bit-identical logits to the
    batch-1 calls the official code makes (scripts/verify_batch_eval.py);
  * softmax is the official TinyPhysicsModel.softmax applied per row.
scripts/verify_batch_eval.py checks that per-segment costs match the official
runs exactly for controllers we already evaluated with run_eval.py.

Controllers: either a list of ordinary per-segment controllers (used for the
exactness check), or one batched controller with
  update_batch(targets[B], currents[B], states: list[State], plans: list[FuturePlan], active[B]) -> actions[B]
whose computations never mix rows (each segment only sees its own inputs).
"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5

import numpy as np

from tinyphysics import (CONTEXT_LENGTH, CONTROL_START_IDX, COST_END_IDX, DEL_T, FUTURE_PLAN_STEPS,
                         LAT_ACCEL_COST_MULTIPLIER, MAX_ACC_DELTA, STEER_RANGE, FuturePlan, State,
                         TinyPhysicsModel, TinyPhysicsSimulator)

from .evaluate import seg_path

CHUNK = 512          # rows per ONNX call (bit-identity checked up to 650)
N_THREADS = 16       # parallel ONNX sessions, each single-threaded like the official one


class BatchSim:
  def __init__(self, segs, model_path="./models/tinyphysics.onnx", deterministic=False):
    # deterministic=True replaces the random sample by the token nearest to the
    # expected lataccel. ANALYSIS ONLY (scripts/noise_floor.py); never used for scores.
    self.deterministic = deterministic
    self.segs = list(segs)
    self.paths = [seg_path(i) for i in self.segs]
    self.models = [TinyPhysicsModel(model_path, debug=False) for _ in range(N_THREADS)]
    self.tok = self.models[0].tokenizer
    data = [TinyPhysicsSimulator.get_data(None, p) for p in self.paths]
    self.cols = [{k: d[k].values for k in d.columns} for d in data]
    self.lengths = np.array([len(d) for d in data])
    self.B = len(self.segs)
    # reset(): histories from the first CONTEXT_LENGTH rows, RNG seeded from the path
    self.state_hist, self.action_hist, self.cur_hist, self.target_hist = [], [], [], []
    for c in self.cols:
      self.state_hist.append([State(c["roll_lataccel"][i], c["v_ego"][i], c["a_ego"][i]) for i in range(CONTEXT_LENGTH)])
      self.action_hist.append(c["steer_command"][:CONTEXT_LENGTH].tolist())
      self.cur_hist.append([c["target_lataccel"][i] for i in range(CONTEXT_LENGTH)])
      self.target_hist.append([c["target_lataccel"][i] for i in range(CONTEXT_LENGTH)])
    self.current = np.array([h[-1] for h in self.cur_hist])
    self.rngs = [np.random.RandomState(int(md5(p.encode()).hexdigest(), 16) % 10**4) for p in self.paths]
    self.pool = ThreadPoolExecutor(N_THREADS)

  def _state_target_plan(self, i, t):
    c = self.cols[i]
    s = slice(t + 1, t + FUTURE_PLAN_STEPS)
    return (State(c["roll_lataccel"][t], c["v_ego"][t], c["a_ego"][t]), c["target_lataccel"][t],
            FuturePlan(lataccel=c["target_lataccel"][s].tolist(), roll_lataccel=c["roll_lataccel"][s].tolist(),
                       v_ego=c["v_ego"][s].tolist(), a_ego=c["a_ego"][s].tolist()))

  def _logits(self, states, tokens):
    def run(k):
      lo = k * CHUNK
      return self.models[k % N_THREADS].ort_session.run(
        None, {"states": states[lo:lo + CHUNK], "tokens": tokens[lo:lo + CHUNK]})[0]
    n = (len(states) + CHUNK - 1) // CHUNK
    return np.concatenate(list(self.pool.map(run, range(n))))

  def run(self, controller):
    per_segment = isinstance(controller, list)
    for t in range(CONTEXT_LENGTH, int(self.lengths.max())):
      active = t < self.lengths
      idx = np.flatnonzero(active)
      stp = [self._state_target_plan(i, t) if active[i] else None for i in range(self.B)]
      for i in idx:                                              # step(): append state and target
        self.state_hist[i].append(stp[i][0])
        self.target_hist[i].append(stp[i][1])
      # control_step()
      if per_segment:
        actions = {i: controller[i].update(self.target_hist[i][t], self.current[i], self.state_hist[i][t],
                                           future_plan=stp[i][2]) for i in idx}
      else:
        targets = np.array([self.target_hist[i][t] if active[i] else 0.0 for i in range(self.B)])
        out = controller.update_batch(targets, self.current.copy(), [s[0] if s else None for s in stp],
                                      [s[2] if s else None for s in stp], active)
        actions = {i: out[i] for i in idx}
      for i in idx:
        a = self.cols[i]["steer_command"][t] if t < CONTROL_START_IDX else actions[i]
        self.action_hist[i].append(np.clip(a, STEER_RANGE[0], STEER_RANGE[1]))
      # sim_step(): official get_current_lataccel, rows batched
      states = np.stack([np.column_stack([self.action_hist[i][-CONTEXT_LENGTH:],
                                          [list(x) for x in self.state_hist[i][-CONTEXT_LENGTH:]]]) for i in idx])
      tokens = np.stack([self.tok.encode(self.cur_hist[i][-CONTEXT_LENGTH:]) for i in idx])
      res = self._logits(states.astype(np.float32), tokens.astype(np.int64))
      for row, i in enumerate(idx):
        probs = self.models[0].softmax(res[row:row + 1] / 0.8, axis=-1)
        if self.deterministic:
          pred = self.tok.decode(self.tok.encode(probs[0, -1] @ self.tok.bins))
        else:
          pred = self.tok.decode(self.rngs[i].choice(probs.shape[2], p=probs[0, -1]))
        pred = np.clip(pred, self.current[i] - MAX_ACC_DELTA, self.current[i] + MAX_ACC_DELTA)
        self.current[i] = pred if t >= CONTROL_START_IDX else self._state_target_plan(i, t)[1]
        self.cur_hist[i].append(self.current[i])
    return [self._cost(i) for i in range(self.B)]

  def _cost(self, i):
    # same as TinyPhysicsSimulator.compute_cost
    target = np.array(self.target_hist[i])[CONTROL_START_IDX:COST_END_IDX]
    pred = np.array(self.cur_hist[i])[CONTROL_START_IDX:COST_END_IDX]
    lat_accel_cost = np.mean((target - pred) ** 2) * 100
    jerk_cost = np.mean((np.diff(pred) / DEL_T) ** 2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    return {"seg": f"{self.segs[i]:05d}", "lataccel_cost": lat_accel_cost, "jerk_cost": jerk_cost,
            "total_cost": total_cost}
