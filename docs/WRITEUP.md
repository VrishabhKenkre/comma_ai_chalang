# Controls Challenge v2 with optimal control, honest track

**Result.** Receding-horizon iLQR through the TinyPhysics model scores **40.55** on the
official 5000 segments in the unmodified `eval.py` (PID baseline 110.25). A
risk-sensitive variant that prices the model's predicted variance scores **40.22** in `eval.py`
(40.23 in our lockstep evaluator). A linear
preview MPC on an identified model scores **57.44**, and the strongest PID+FF I could tune
scores 60.41. All controllers are causal and only use what `update()` receives. They
never touch the RNG or seed, never read the data file, and use the same gains on every
segment. Tuning used held-out dev segments only. Repo: `<link>`.

| controller (5000 test segments) | lataccel | jerk | total | ms/step |
|---|---:|---:|---:|---:|
| PID (baseline) | 1.695 | 25.49 | 110.25 | 0.001 |
| PID + preview feedforward, gains tuned once | 0.868 | 20.24 | 63.64 | 0.014 |
| PID + richer preview feedforward, 10 gains tuned once by CMA-ES | 0.813 | 19.75 | 60.41 | 0.022 |
| LQR on identified ARX + Kalman observer (no preview) | 1.504 | 24.64 | 99.85 | 0.11 |
| Preview MPC on ARX, no observer | 1.634 | 17.80 | 99.49 | 0.40 |
| Preview MPC on ARX + observer, OSQP | 0.734 | 20.72 | 57.44 | 0.43 |
| same, hand-written ADMM (cached Riccati, TinyMPC style) | 0.734 | 20.69 | 57.40 | 0.24 |
| same MPC, 13 parameters tuned by CMA-ES on dev | 0.737 | 19.81 | 56.68 | 0.15 |
| iLQR through TinyPhysics (expected value), H = 12 (eval.py) | 0.437 | 18.70 | 40.55 | 64 |
| same, risk-sensitive (+ exact predicted variance in the cost; eval.py) | 0.433 | 18.56 | **40.22** | 70 |

**What I did.**
1. *System ID.* I ported the ONNX model to PyTorch (matches to 4e-6). Finite differences
   of the expected lataccel `sum p_i bin_i` show `E[y_t] ~ 0.95 y_{t-1}` plus a band-pass
   steer kernel with 2-4 steps of delay. I fitted a 1-7-7 ARX per 5 m/s speed bin. Its
   held-out one-step RMSE is 0.044 m/s², the same as the simulator's own sampling noise
   (0.046).
2. *Offset-free preview MPC.* The cost is the score itself, `50 e² + 100 Δy²`, over a
   40-step plan, with steer and steer-rate limits. A 2-state Kalman filter estimates an output
   offset and a roll-gain error. ADMM with cached finite-horizon Riccati gains matches OSQP
   to 1e-5 on about 10k real QPs at about 3x the speed (0.23 vs 0.65 ms).
3. *iLQR through the simulator model.* The state is the 39-dim context and the Jacobians
   come from one batched autograd pass. Past lataccel enters as a token, and its
   interpolated-embedding gradient is noise, so the backward pass uses the embedding slope
   over ±3 bins. When the plan saturates at the ±5 token limit the gradient vanishes, so a
   few shifted plans are tried there.

**Findings worth knowing.**
- Most of what remains is the simulator's sampling noise. With the random token draw
  replaced by the mean token (analysis only), iLQR drops from 36.8 to 12.7 on dev, so 66% of
  its cost is noise. The figure is 31% for PID+FF. Preview matters most: the car has about
  3 steps of delay.
- The ARX is accurate only for smooth steering. The MPC needs a large steer-rate weight, and
  the observer has to be an *output* offset. Inside an ARX with a ≈ 0.95 a bias acts about
  20x, and in my first version it destabilised the loop.
- The first iLQR test run scored 58.19. Two of the 5000 segments locked at full steer
  (one at 84,143). I reproduced the failure on non-test high-lateral-acceleration segments,
  fixed it there, and re-ran the test once. Only 7 test segments changed. Both numbers are
  in the repo.

**Caveats.** The iLQR has a perfect *mean* model because it uses the simulator's own network.
That is model-in-the-loop control, comparable to the leaderboard's MPC entries (33.6-44.3),
not to pure feedback controllers. On a car the model would be learned and the gap would
be closer to the linear-MPC result. The simulator runs at 10 Hz. The iLQR takes 64 ms per
step on one desktop CPU thread (max 66 ms, risk-sensitive 70 / 73 ms), including the
engagement step: its first plan is warmed during the warmup steps, whose output the simulator
ignores, instead of with 10 iterations at engagement. Numbers come from the unmodified `tinyphysics.py`,
either one process per segment or a lockstep batch evaluator that reproduces official
per-segment costs exactly.

## Where the remaining cost comes from, and what optimization can do about it

**The split.** I ran the iLQR on dev segments with the simulator's random token draw
replaced by the token nearest the mean (an analysis-only plant). Its cost drops from 36.8 to
12.7. So about two thirds of what is left is the simulator's sampling noise, and one third is
the controller.

**Can steering reduce the noise? Only a little.** The width of the predicted distribution
does depend on the steer, but in one direction: any step in steer widens it. Two steps later
it is about 1.4x wider after a 0.1 step and about 4x wider after a 0.3 step. Smooth steering
is therefore near the minimum already. Putting the exact predicted variance into the iLQR cost (weight 1.5, fixed by the score, not tuned) gains
0.33 +- 0.03 on test (40.56 -> 40.23, paired, lockstep evaluator). That is real but under 1%.

**Exact expectations do not beat structure.** An exact expected-cost one-step lookahead is a
disaster (11,412 on dev), because the next lataccel barely depends on the current steer (3
steps of delay). Looking over the delay with one held steer value is still poor (326). MPPI
with independent per-step noise failed (1,649). With smooth perturbations it works (39.1),
but gradient-based iLQR stays ahead by about 2 points at a fraction of the compute.
Inside MPPI, pricing the variance is worth 1.8 +- 0.2 points (40.9 without it), much more
than inside iLQR, presumably because sampled plans contain more variance-inducing steer
changes.

**Better tuning barely helps the linear MPC.** CMA-ES over 13 MPC and observer parameters,
including the speed breakpoints, gains 0.75 +- 0.40 on test (57.44 -> 56.68). It found very
different settings that score almost the same, so the objective is flat. The linear model, not
the tuning, is the limit.

**A better model of the nonlinearity did not help the MPC.** A piecewise-affine model with 8
regions on lataccel and steer fits the network better than one linear model (20-step error
0.108 vs 0.124 m/s^2). I solved hybrid MPC on it with a Graphs-of-Convex-Sets relaxation. The
relaxation is loose: its bound is 41% below the exact optimum, checked against SCIP on 200
problems. Rounding nevertheless recovers the optimum on most problems (median 0%, mean
4.7% above). In closed loop it is still worse than the linear MPC on 27 of 32 segments, and it
blows up on a few by parking the steer on a region boundary. It is also 6x too slow for 10 Hz.
The gap between the linear MPC (57.4) and the iLQR (40.6) points to model fidelity (the iLQR
plans on the exact mean model), and a discontinuous PWA model does not close it.

**What is left.** With an exact mean model, the iLQR's noise-free cost on these segments is
12.7. That is what a better controller could still remove. The rest, about 24, is sampling
noise, which steering can only nudge. When the model is identified instead of exact, model
error takes over: the linear MPC is 17 points behind the iLQR on test. On a real car, where
no exact model exists, that suggests model quality would matter more than the choice of
solver.

<!-- ABLATION:START -->
| method | dev segments | dev lataccel | dev jerk | dev total | iLQR, same segs | linear MPC, same segs | test total (run file) | test total (eval.py) | ms/step | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PID (upstream baseline) | 10000-10499 (500) | 1.697 | 25.66 | 110.52 | 38.89 | 54.31 | 110.25 | 110.25 | 0.0007 |  |
| PID + feedforward, Nelder-Mead | 10000-10499 (500) | 0.820 | 20.05 | 61.05 | 38.89 | 54.31 | 63.64 |  | 0.015 |  |
| PID + feedforward, CMA-ES (strong baseline) | 10000-10499 (500) | 0.774 | 19.42 | 58.12 | 38.89 | 54.31 | 60.41 | 60.41 | 0.022 |  |
| LQR on ARX + observer (no preview) | 10000-10499 (500) | 1.419 | 23.94 | 94.88 | 38.89 | 54.31 | 99.85 |  | 0.108 |  |
| MPC on ARX, no observer | 10000-10499 (500) | 1.573 | 17.50 | 96.15 | 38.89 | 54.31 | 99.49 |  | 0.402 |  |
| MPC on ARX + observer, OSQP | 10000-10499 (500) | 0.686 | 20.00 | 54.31 | 38.89 | 54.31 | 57.44 | 57.44 | 0.430 |  |
| MPC on ARX + observer, ADMM | 10000-10499 (500) | 0.686 | 20.00 | 54.30 | 38.89 | 54.31 | 57.40 |  | 0.244 |  |
| MPC on ARX + observer, CMA-ES tuned (13 params) | 10000-10499 (500) | 0.684 | 18.82 | 53.03 | 38.89 | 54.31 | 56.68 | 56.68 | 0.148 | tuned on dev 10500-10999 |
| iLQR through TinyPhysics, first version | 10000-10499 (500) | 0.415 | 18.13 | 38.88 | 38.89 | 54.31 | 58.19 |  |  | blow-ups, see README |
| iLQR, 10 iterations at engagement (superseded) | 10000-10499 (500) | 0.415 | 18.14 | 38.92 | 38.89 | 54.31 | 40.59 | 40.54 | 64.4 | engagement step 404 ms |
| iLQR through TinyPhysics (final) | 10000-10499 (500) | 0.415 | 18.13 | 38.89 | 38.89 | 54.31 | 40.56 | 40.55 | 64.1 |  |
| Risk-sensitive iLQR, 10 iterations at engagement (superseded) | 10000-10499 (500) | 0.412 | 18.03 | 38.65 | 38.89 | 54.31 | 40.23 |  | 71.9 | engagement step 473 ms |
| Risk-sensitive iLQR (+1.5 sigma^2, exact) | 10000-10499 (500) | 0.412 | 18.04 | 38.64 | 38.89 | 54.31 | 40.23 | 40.22 | 69.6 |  |
| Expected-cost lookahead, one step (d=0) | 10000-10199 (200) | 226.999 | 61.99 | 11411.94 | 36.80 | 48.91 |  |  |  | negative result |
| Expected-cost lookahead over the delay (d=3) | 10000-10199 (200) | 6.038 | 24.13 | 326.05 | 36.80 | 48.91 |  |  | 68.4 | negative result |
| MPPI, independent per-step perturbations | 10000-10199 (200) | 31.680 | 64.82 | 1648.79 | 36.80 | 48.91 |  |  |  | negative result |
| MPPI, smooth (ramp) perturbations, + variance | 10000-10199 (200) | 0.427 | 17.75 | 39.09 | 36.80 | 48.91 |  |  | 1538 |  |
| MPPI, smooth perturbations, no variance term | 10000-10199 (200) | 0.453 | 18.23 | 40.89 | 36.80 | 48.91 |  |  |  |  |
| Linear MPC, horizon 8 (control for the hybrid MPC) | 10000-10031 (32) | 0.605 | 19.22 | 49.47 | 42.86 | 49.57 |  |  |  |  |
| Hybrid-MPC code with the linear MPC's ARX (check) | 10000-10031 (32) | 0.605 | 19.17 | 49.41 | 42.86 | 49.57 |  |  |  | must match the row above |
| Hybrid-MPC code, speed-binned refit (1 mode/step) | 10000-10031 (32) | 0.626 | 19.04 | 50.36 | 42.86 | 49.57 |  |  |  |  |
| Hybrid MPC on 8-region PWA, GCS relaxation + rounding | 10000-10031 (32) | 5.957 | 20.45 | 318.29 | 42.86 | 49.57 |  |  | 597 | no test run (too slow) |
<!-- ABLATION:END -->

Dev numbers are on the listed dev segments. The iLQR and linear-MPC columns are those two
controllers on the same segments. Test is the official 5000 segments, run once per final
method. Every number comes from a saved run (`scripts/make_ablation.py`).

