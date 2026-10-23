"""TinyPhysics transformer re-implemented in PyTorch, weights read from the ONNX file.

Architecture (read from the ONNX graph):
  x = concat(states @ W_s + b_s, tok_emb[tokens]) + pos_emb      # 64 + 64 = 128 dims
  4 x [ x += proj(causal_attn(LN(x)))  ;  x += mlp(LN(x)) ]       # 4 heads, GELU(tanh)
  logits = LN_f(x) @ W_head                                       # 1024 lataccel bins

Checked against onnxruntime in scripts/check_torch_model.py.

For gradients we also need a differentiable version of the token input
(past lataccel enters as a discrete token). `soft_tokens=True` replaces the
embedding lookup by linear interpolation between the two nearest bin
embeddings, which equals the lookup exactly at bin centers.
"""
import math

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from tinyphysics import LATACCEL_RANGE, VOCAB_SIZE

TEMPERATURE = 0.8


class TinyPhysicsTorch(torch.nn.Module):
  def __init__(self, onnx_path="./models/tinyphysics.onnx"):
    super().__init__()
    g = onnx.load(onnx_path).graph
    w = {t.name: torch.tensor(numpy_helper.to_array(t).copy()) for t in g.initializer}
    mm = [w[n] for n in sorted((n for n in w if n.startswith("onnx::MatMul_")),
                               key=lambda n: int(n.split("_")[-1]))]
    # MatMul order in the graph: state embedding, then per layer (qkv, proj, fc, fc_proj), then head
    self.W_s, self.b_s = mm[0], w["transformer.wt_embedding.bias"]
    self.tok_emb = w["transformer.wt2_embedding.weight"]
    self.pos_emb = w["transformer.wp_embedding.weight"]
    self.mask = w["transformer.h.0.attn.bias"].bool()[0, 0]          # [20, 20] lower-triangular
    self.layers = []
    for i in range(4):
      p = f"transformer.h.{i}."
      self.layers.append(dict(
        ln1=(w[p + "attn.layer_norm.weight"], w[p + "attn.layer_norm.bias"]),
        qkv=mm[1 + 4 * i], proj=mm[2 + 4 * i],
        ln2=(w[p + "mlp.layer_norm.weight"], w[p + "mlp.layer_norm.bias"]),
        fc=mm[3 + 4 * i], fc_proj=mm[4 + 4 * i]))
    self.ln_f = (w["transformer.layer_norm_f.weight"], w["transformer.layer_norm_f.bias"])
    self.W_head = mm[17]
    self.bins = torch.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE)
    self.n_head = 4

  def to(self, device):
    for k, v in list(vars(self).items()):
      if isinstance(v, torch.Tensor):
        setattr(self, k, v.to(device))
    for layer in self.layers:
      for k, v in layer.items():
        layer[k] = tuple(t.to(device) for t in v) if isinstance(v, tuple) else v.to(device)
    self.ln_f = tuple(t.to(device) for t in self.ln_f)
    return self

  @staticmethod
  def _ln(x, wb):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + 1e-5) * wb[0] + wb[1]

  def embed_tokens(self, past_y, soft, grad_halfwidth=3):
    if not soft:
      # same as LataccelTokenizer.encode: np.digitize(clip(y), bins, right=True)
      tok = torch.bucketize(past_y.contiguous().clamp(*LATACCEL_RANGE), self.bins, right=False)
      return self.tok_emb[tok]
    step = self.bins[1] - self.bins[0]
    pos = (past_y.clamp(*LATACCEL_RANGE) - LATACCEL_RANGE[0]) / step
    lo = pos.floor().clamp(0, VOCAB_SIZE - 2)
    frac = (pos - lo).unsqueeze(-1)
    lo = lo.long()
    emb = self.tok_emb[lo] * (1 - frac) + self.tok_emb[lo + 1] * frac
    if not past_y.requires_grad:
      return emb
    # Learned embeddings are not smooth from one bin to the next, so the slope of
    # the interpolation inside one bin is noisy. For gradients we use the slope of
    # the table over +-grad_halfwidth bins (forward value unchanged).
    m = grad_halfwidth
    i0 = (lo - m + 1).clamp(0, VOCAB_SIZE - 1)
    i1 = (lo + m).clamp(0, VOCAB_SIZE - 1)
    slope = (self.tok_emb[i1] - self.tok_emb[i0]) / ((i1 - i0).clamp(min=1).unsqueeze(-1) * step)
    return emb.detach() + (past_y - past_y.detach()).unsqueeze(-1) * slope.detach()

  def logits(self, states, past_y, soft_tokens=False):
    """states [B, T, 4], past_y [B, T] (float lataccel values) -> logits [B, T, 1024]."""
    B, T, _ = states.shape
    x = torch.cat([states @ self.W_s + self.b_s, self.embed_tokens(past_y, soft_tokens)], -1)
    x = x + self.pos_emb[:T]
    mask = self.mask[:T, :T]
    hd = x.shape[-1] // self.n_head
    for L in self.layers:
      q, k, v = (self._ln(x, L["ln1"]) @ L["qkv"]).split(x.shape[-1], dim=-1)
      q, k, v = (t.view(B, T, self.n_head, hd).transpose(1, 2) for t in (q, k, v))
      att = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
      att = att.masked_fill(~mask, float("-inf")).softmax(-1)
      x = x + (att @ v).transpose(1, 2).reshape(B, T, -1) @ L["proj"]
      h = self._ln(x, L["ln2"]) @ L["fc"]
      h = 0.5 * h * (1 + torch.tanh(math.sqrt(2 / math.pi) * (h + 0.044715 * h ** 3)))
      x = x + h @ L["fc_proj"]
    return self._ln(x, self.ln_f) @ self.W_head

  def expected(self, states, past_y, soft_tokens=False):
    """E[lataccel_t | context] at the last position, temperature 0.8 like the simulator."""
    p = (self.logits(states, past_y, soft_tokens)[:, -1] / TEMPERATURE).softmax(-1)
    return p @ self.bins

  def mean_std(self, states, past_y, soft_tokens=False):
    """Mean and standard deviation of the predicted lataccel distribution (exact, over the bins)."""
    p = (self.logits(states, past_y, soft_tokens)[:, -1] / TEMPERATURE).softmax(-1)
    mean = p @ self.bins
    var = (p * (self.bins - mean[:, None]) ** 2).sum(-1)
    return mean, torch.sqrt(var + 1e-12)
