"""Step 1a: does the steer action change the WIDTH of TinyPhysics' predicted distribution?

Operating points: every 4th controlled step (100..496) of the final iLQR's closed-loop
runs on dev segments 10000-10099 (lockstep evaluator, official sampling).
At each point the past is held fixed and the steer from step t on is set to
u_t + delta (held for k+1 steps, k = 0..4); intermediate lataccels are the model's
expected values. We record, for the distribution of y_{t+k}:
  std  = sqrt(sum p_i (bin_i - mean)^2),   entropy = -sum p_i log p_i,   mean
Only model probabilities are used, never a sample (no RNG).

Outputs results/variance/variance_vs_action.json and .png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from controllers.ilqr import PARAMS  # noqa: E402
from ctrl.batch_eval import BatchSim  # noqa: E402
from ctrl.ilqr import BatchILQR  # noqa: E402
from ctrl.torch_model import TinyPhysicsTorch, TEMPERATURE  # noqa: E402

SEGS = range(10000, 10100)
DELTAS = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
KS = [0, 1, 2, 3, 4]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = ROOT / "results" / "variance"


def operating_points():
  sim = BatchSim(SEGS)
  sim.run(BatchILQR(len(SEGS), device=DEV, **PARAMS))
  S, Y, T = [], [], []
  for i in range(len(SEGS)):
    u = np.array(sim.action_hist[i]); y = np.array(sim.cur_hist[i])
    ex = np.array([list(s) for s in sim.state_hist[i]])          # roll, v, a
    for t in range(100, 497, 4):
      # rows t-19 .. t+4 (actions, exo), past y t-20 .. t-1
      S.append(np.column_stack([u[t - 19:t + 5], ex[t - 19:t + 5]]))
      Y.append(y[t - 20:t])
      T.append(t)
  return np.array(S, np.float32), np.array(Y, np.float32)


@torch.no_grad()
def distribution_stats(model, S, Y):
  """For every delta and k: std, entropy and mean of p(y_{t+k}). S: [N, 24, 4], Y: [N, 20]."""
  N = len(S)
  S = torch.tensor(S, device=DEV); Y = torch.tensor(Y, device=DEV)
  bins = model.bins
  out = {k: {"std": [], "ent": [], "mean": []} for k in KS}
  for d in DELTAS:
    s = S.clone()
    s[:, 19:, 0] = s[:, 19:20, 0] + d                  # u_t + delta held for steps t .. t+4
    y = Y.clone()
    for k in KS:
      logits = model.logits(s[:, k:k + 20], y[:, -20:], soft_tokens=True)[:, -1]
      p = (logits / TEMPERATURE).softmax(-1)
      mean = p @ bins
      std = torch.sqrt((p * (bins - mean[:, None]) ** 2).sum(-1))
      ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
      out[k]["std"].append(std.cpu().numpy()); out[k]["ent"].append(ent.cpu().numpy())
      out[k]["mean"].append(mean.cpu().numpy())
      y = torch.cat([y, mean[:, None]], 1)              # roll the mean forward
  return {k: {m: np.stack(v[m], 1) for m in v} for k, v in out.items()}   # each [N, len(DELTAS)]


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  S, Y = operating_points()
  model = TinyPhysicsTorch().to(DEV)
  st = distribution_stats(model, S, Y)
  i0 = DELTAS.index(0.0)
  summary = dict(n_points=len(S), segs="10000-10099 (iLQR closed loop)", deltas=DELTAS, by_k={})
  for k in KS:
    std, ent, mean = st[k]["std"], st[k]["ent"], st[k]["mean"]
    var0 = std[:, i0] ** 2
    row = dict(std_at_delta0_mean=float(std[:, i0].mean()), entropy_at_delta0_mean=float(ent[:, i0].mean()))
    for lim in (0.05, 0.1, 0.3, 0.5):
      cols = [j for j, d in enumerate(DELTAS) if abs(d) <= lim + 1e-9]
      vmin = (std[:, cols] ** 2).min(1)
      row[f"|delta|<={lim}"] = dict(
        # best achievable variance reduction by choosing delta (ignoring its effect on the mean)
        mean_rel_var_reduction=float(np.mean(1 - vmin / var0)),
        median_rel_var_reduction=float(np.median(1 - vmin / var0)),
        # how far the same deltas move the mean, for comparison
        mean_abs_mean_shift=float(np.abs(mean[:, cols] - mean[:, [i0]]).max(1).mean()),
        mean_abs_std_change=float(np.abs(std[:, cols] - std[:, [i0]]).max(1).mean()))
    row["mean_std_by_delta"] = [float(x) for x in std.mean(0)]
    row["mean_entropy_by_delta"] = [float(x) for x in ent.mean(0)]
    summary["by_k"][k] = row
    r1, r3 = row["|delta|<=0.1"], row["|delta|<=0.3"]
    print(f"k={k}: std(0) {row['std_at_delta0_mean']:.4f}  entropy(0) {row['entropy_at_delta0_mean']:.3f} | "
          f"|d|<=0.1: var reduction mean {100 * r1['mean_rel_var_reduction']:.1f}% (median "
          f"{100 * r1['median_rel_var_reduction']:.1f}%), std change {r1['mean_abs_std_change']:.4f}, "
          f"mean shift {r1['mean_abs_mean_shift']:.3f} | |d|<=0.3: var reduction {100 * r3['mean_rel_var_reduction']:.1f}%, "
          f"mean shift {r3['mean_abs_mean_shift']:.3f}", flush=True)
  json.dump(summary, open(OUT / "variance_vs_action.json", "w"), indent=1)

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, axs = plt.subplots(1, 2, figsize=(12, 4))
  for k in KS:
    axs[0].plot(DELTAS, summary["by_k"][k]["mean_std_by_delta"], "o-", label=f"y(t+{k})")
    axs[1].plot(DELTAS, summary["by_k"][k]["mean_entropy_by_delta"], "o-", label=f"y(t+{k})")
  axs[0].set(xlabel="steer change delta (held from step t)", ylabel="mean std of p(y) [m/s^2]",
             title="Spread of the predicted lataccel vs steer change")
  axs[1].set(xlabel="steer change delta", ylabel="mean entropy [nats]", title="Entropy vs steer change")
  axs[0].legend()
  fig.tight_layout()
  fig.savefig(OUT / "variance_vs_action.png", dpi=110)


if __name__ == "__main__":
  main()
