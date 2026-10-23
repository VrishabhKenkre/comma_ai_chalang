"""Check the PyTorch port against onnxruntime on real contexts from the data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctrl.tinymodel import ExpectedModel, BINS  # noqa: E402
from ctrl.torch_model import TinyPhysicsTorch  # noqa: E402

torch.set_default_dtype(torch.float32)

# real contexts: 20-step windows from the warmup part of 50 non-test segments
states, past = [], []
for i in range(15000, 15050):
  df = pd.read_csv(ROOT / "data" / f"{i:05d}.csv")
  s = np.column_stack([-df.steerCommand, np.sin(df.roll) * 9.81, df.vEgo, df.aEgo])
  y = df.targetLateralAcceleration.values
  for t in range(20, 99, 3):
    states.append(s[t - 19:t + 1]); past.append(y[t - 20:t])
states, past = np.array(states, np.float32), np.array(past)
# snap past lataccel to bin centres so soft and hard tokens should agree exactly
past_snapped = BINS[np.digitize(np.clip(past, -5, 5), BINS, right=True)]

onnx_model = ExpectedModel(threads=8)
p_onnx = onnx_model.probs(states, past)
e_onnx = p_onnx @ BINS

for device in ["cpu", "cuda"]:
  m = TinyPhysicsTorch().to(device)
  with torch.no_grad():
    S = torch.tensor(states, device=device)
    P = torch.tensor(past, dtype=torch.float32, device=device)
    logit = m.logits(S, P)[:, -1].cpu().numpy()
    p_t = torch.softmax(torch.tensor(logit) / 0.8, -1).numpy()
    e_hard = m.expected(S, P).cpu().numpy()
    e_soft = m.expected(S, torch.tensor(past_snapped, dtype=torch.float32, device=device),
                        soft_tokens=True).cpu().numpy()
  print(f"{device}: n={len(states)}  max|dp|={np.abs(p_t - p_onnx).max():.2e}  "
        f"max|dE| hard={np.abs(e_hard - e_onnx).max():.2e}  soft@bin-centres={np.abs(e_soft - e_onnx).max():.2e}  "
        f"argmax agree={np.mean(p_t.argmax(1) == p_onnx.argmax(1)):.4f}")
