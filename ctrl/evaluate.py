"""Parallel rollouts with controller *instances* (used for tuning and analysis).

Equivalent to tinyphysics.run_rollout, but takes a picklable factory so that
controllers can be built with parameters. The simulator object and its seeding
are the official ones; the data path string is built the same way as eval.py
("data/12345.csv") so the per-segment noise is identical to an official run.
"""
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

_model = None


def seg_path(i):
  return str(Path("data") / f"{i:05d}.csv")


def _init_worker():
  global _model
  os.environ["OMP_NUM_THREADS"] = "1"
  _model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)


def _run(args):
  make_controller, i, keep_history = args
  sim = TinyPhysicsSimulator(_model, seg_path(i), controller=make_controller(), debug=False)
  cost = sim.rollout()
  row = {"seg": f"{i:05d}", **cost}
  if keep_history:
    row["target"] = sim.target_lataccel_history
    row["current"] = sim.current_lataccel_history
    row["action"] = sim.action_history
  return row


def evaluate(make_controller, segs, workers=32, keep_history=False):
  jobs = [(make_controller, i, keep_history) for i in segs]
  with Pool(workers, initializer=_init_worker) as pool:
    rows = pool.map(_run, jobs, chunksize=max(1, len(jobs) // (4 * workers)))
  return pd.DataFrame(rows)
