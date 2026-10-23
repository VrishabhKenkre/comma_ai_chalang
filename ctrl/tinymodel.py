"""Expected-value wrapper around the TinyPhysics ONNX model (offline analysis only).

The simulator draws a token from softmax(logits / 0.8) and decodes it. Here we
use the mean of that distribution instead of a sample:

    E[lataccel | context] = sum_i p_i * bin_i

This is used for system identification and for the iLQR stretch goal. It never
touches numpy's global RNG.
"""
import numpy as np
import onnxruntime as ort

from tinyphysics import CONTEXT_LENGTH, LATACCEL_RANGE, VOCAB_SIZE

TEMPERATURE = 0.8
BINS = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)


def encode(y):
  """Same tokenization as tinyphysics.LataccelTokenizer.encode."""
  return np.digitize(np.clip(y, *LATACCEL_RANGE), BINS, right=True)


class ExpectedModel:
  def __init__(self, model_path="./models/tinyphysics.onnx", threads=1):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    with open(model_path, "rb") as f:
      self.sess = ort.InferenceSession(f.read(), opts, ["CPUExecutionProvider"])

  def probs(self, states, past_y):
    """states: [B, 20, 4] = (steer, roll_lataccel, v_ego, a_ego); past_y: [B, 20].
    Returns next-token probabilities [B, 1024] at the last position."""
    states = np.asarray(states, dtype=np.float32)
    tokens = encode(np.asarray(past_y)).astype(np.int64)
    assert states.shape[1] == CONTEXT_LENGTH and tokens.shape[1] == CONTEXT_LENGTH
    logits = self.sess.run(None, {"states": states, "tokens": tokens})[0][:, -1, :]
    z = logits / TEMPERATURE
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)

  def expected(self, states, past_y, batch=4096):
    out = []
    for i in range(0, len(states), batch):
      out.append(self.probs(states[i:i + batch], past_y[i:i + batch]) @ BINS)
    return np.concatenate(out)
