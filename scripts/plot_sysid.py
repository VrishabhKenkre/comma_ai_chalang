"""Plots and a markdown report for results/sysid/sysid_summary.json."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctrl import arx  # noqa: E402

OUT = ROOT / "results" / "sysid"


def main():
  S = json.load(open(OUT / "sysid_summary.json"))
  lin = S["linearization_by_speed_bin"]
  edges = arx.SPEED_EDGES
  labels = [f"{edges[i]}-{edges[i + 1] if edges[i + 1] < 100 else ''} m/s" for i in range(len(edges) - 1)]
  cmap = plt.get_cmap("viridis")

  fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
  for b, L in enumerate(lin):
    if L["n_points"] < 200:
      continue
    c = cmap(b / (len(lin) - 1))
    axs[0].plot(range(20), L["u"], "o-", ms=3, color=c, label=f"{labels[b]} (n={L['n_points']})")
    axs[1].plot(range(1, 21), L["y"], "o-", ms=3, color=c)
    axs[2].plot(range(20), L["roll"], "o-", ms=3, color=c)
  axs[0].set(title="dE[y_t]/du_{t-k}  (steer)", xlabel="lag k [steps of 0.1 s]")
  axs[1].set(title="dE[y_t]/dy_{t-k}  (past lataccel)", xlabel="lag k")
  axs[2].set(title="dE[y_t]/droll_{t-k}", xlabel="lag k")
  axs[0].legend(fontsize=7)
  for a in axs:
    a.axhline(0, color="k", lw=0.5)
  fig.suptitle("Finite-difference linearization of the expected TinyPhysics output, averaged per speed bin")
  fig.tight_layout()
  fig.savefig(OUT / "linearization_lags.png", dpi=120)

  # markdown report
  ch = S["chosen"]
  models = json.load(open(OUT / "arx_models.json"))["models"]
  lines = ["# System identification", "",
           f"Fit segments {S['fit_segs']}, held-out segments {S['holdout_segs']} (test set 00000-04999 not used).", "",
           "## Fit error on held-out segments", "",
           "RMSE in m/s^2. One-step = predict E[y_t] from the true past. Free-run = 20 steps (2 s) "
           "with the model's own outputs fed back, compared with the expected-value TinyPhysics run on the same inputs. "
           f"Std of E[y] on held-out data: {S['y_std_holdout']:.3f}.", "",
           "| model | one-step RMSE | 20-step free-run RMSE |", "|---|---:|---:|"]
  for g in sorted(S["grid"], key=lambda g: g["free_run_rmse"])[:8]:
    tag = " **(chosen)**" if (g["na"], g["nb"], g["nc"]) == (ch["na"], ch["nb"], ch["nc"]) else ""
    lines.append(f"| ARX na={g['na']} nb={g['nb']} nc={g['nc']}{tag} | {g['one_step_rmse']:.4f} | {g['free_run_rmse']:.4f} |")
  ja = S["jacobian_arx"]
  ref = S["references"]
  lines += [f"| Jacobian linearization, {ja['lags']} lags (only bias fitted) | {ja['one_step_rmse']:.4f} | {ja['free_run_rmse']:.4f} |",
            f"| hold last value (reference) | {ref['hold_last_value_one_step']:.4f} | {ref['hold_last_value_free_run']:.4f} |", "",
            f"Noise floor: RMSE between the full model's E[y_t] and the sampled y_t is "
            f"{ref['full_model_vs_sampled_one_step']:.4f} (the simulator's own sampling noise).", "",
            "## Chosen ARX per speed bin", "",
            "| speed bin | n points | a (y lags) | b (u lags 0..nb) | c (roll lags) | d | dc gain u | dc gain roll |",
            "|---|---:|---|---|---|---:|---:|---:|"]
  for b, m in enumerate(models):
    mm = arx.ARX(**m)
    gu, gr = mm.dc_gain()
    f = lambda v: "[" + ", ".join(f"{x:.3f}" for x in v) + "]"
    lines.append(f"| {labels[b]} | {lin[b]['n_points']} | {f(mm.a)} | {f(mm.b)} | {f(mm.c)} | {mm.d:.3f} | {gu:.3f} | {gr:.3f} |")
  lines += ["", "![linearization](linearization_lags.png)", ""]

  rounds_file = OUT / "closed_loop_rounds.json"
  if rounds_file.exists():
    lines += ["## Closed-loop re-identification (scripts/sysid_closed_loop.py)", "",
              "The ARX above was fitted on PID+FF-like steering. Each round runs the MPC (+ dither) on the sysID "
              "segments and refits the same ARX order on all data so far. One-step / 20-step free-run RMSE on "
              "held-out segments of each regime, before and after the refit:", "",
              "| round | held-out data | previous model | refit model |", "|---|---|---|---|"]
    for R in json.load(open(rounds_file)):
      for name, v in R["refit"].items():
        p = R["previous"][name]
        lines.append(f"| {R['round']} | {name} | {p['one_step']:.4f} / {p['free_run']:.4f} | {v['one_step']:.4f} / {v['free_run']:.4f} |")
    sweep = json.load(open(ROOT / "results" / "tuning" / "mpc_sweep_10000-10199.json"))
    grid0 = [r for r in sweep if "q_k" not in r]          # first grid: rounds 0 and 1 on identical settings
    best = {k: min(r["total_cost"] for r in grid0 if r["round"] == k) for k in (0, 1)}
    wins = sum(1 for a in grid0 for b in grid0 if a["round"] == 1 and b["round"] == 0
               and (a["w_du"], a["q_d"], a["H"]) == (b["w_du"], b["q_d"], b["H"]) and a["total_cost"] < b["total_cost"])
    lines += ["", f"The MPC uses the round-1 models (`arx_models_round1.json`). In the dev grid "
              f"(segments 10000-10199, results/tuning/mpc_sweep_10000-10199.json) the best round-1 setting scored "
              f"{best[1]:.2f} vs {best[0]:.2f} for the best round-0 setting; round 1 was better on {wins} of "
              f"{len(grid0) // 2} identical settings. The gain is small: the best settings steer smoothly, where "
              "the round-0 model was already accurate.", ""]
  (OUT / "SYSID.md").write_text("\n".join(lines))
  print("\n".join(lines))


if __name__ == "__main__":
  main()
