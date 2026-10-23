# Controls Challenge v2 with optimal control, honest track

**Result.** Receding-horizon iLQR through the TinyPhysics model scores **40.59** on the
official 5000 segments (PID baseline 110.25). A linear preview MPC on an identified model
scores **57.44**. Both controllers are causal and only use what `update()` receives. They
never touch the RNG or seed, never read the data file, and use the same gains on every
segment. Tuning used held-out dev segments only. Repo: `<link>`.

| controller (5000 test segments) | lataccel | jerk | total | ms/step |
|---|---:|---:|---:|---:|
| PID (baseline) | 1.695 | 25.49 | 110.25 | 0.001 |
| PID + preview feedforward, gains tuned once | 0.868 | 20.24 | 63.64 | 0.014 |
| LQR on identified ARX + Kalman observer (no preview) | 1.504 | 24.64 | 99.85 | 0.11 |
| Preview MPC on ARX, no observer | 1.634 | 17.80 | 99.49 | 0.40 |
| Preview MPC on ARX + observer, OSQP | 0.734 | 20.72 | 57.44 | 0.43 |
| same, hand-written ADMM (cached Riccati, TinyMPC style) | 0.734 | 20.69 | 57.40 | 0.24 |
| iLQR through TinyPhysics (expected value), H = 12 | 0.438 | 18.71 | **40.59** | 65 |

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
  the observer has to be an *output* offset (inside an ARX with a ≈ 0.95 it acts about 20x
  and destabilises the loop).
- The first iLQR test run scored 58.19. Two of the 5000 segments locked at full steer
  (one at 84,143). I reproduced the failure on non-test high-lateral-acceleration segments,
  fixed it there, and re-ran the test once. Only 7 test segments changed. Both numbers are
  in the repo.

**Caveats.** The iLQR has a perfect *mean* model because it uses the simulator's own network.
That is model-in-the-loop control, comparable to the leaderboard's MPC entries (33.6-44.3),
not to pure feedback controllers. On a car the model would be learned and the gap would
be closer to the linear-MPC result. The iLQR runs at 65 ms per step on one desktop CPU
thread. Numbers come from the unmodified `tinyphysics.py`, either one process per segment
or a lockstep batch evaluator that reproduces official per-segment costs exactly.
