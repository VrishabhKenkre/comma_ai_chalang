# comma Controls Challenge v2 with optimal control: one-page summary

Causal steering controllers that use only what `update()` receives. They never touch the RNG or seed, never read the data file and never tune per segment. All tuning used dev segments; each final method was run once on the official 5000 test segments. `tinyphysics.py` and `eval.py` are unmodified. Every number below comes from a saved run, and the table says which evaluator produced it.

| controller | test total cost | evaluator | ms/step (1 CPU thread) |
|---|---:|---|---:|
| PID (upstream baseline) | 110.25 | eval.py | 0.0007 |
| PID + preview feedforward, 10 gains tuned once by CMA-ES | 60.41 | eval.py | 0.022 |
| Linear MPC on an identified ARX + Kalman offset observer, hand-tuned | 57.44 | eval.py | 0.43 |
| same MPC, 13 parameters tuned by CMA-ES | 56.68 | eval.py | 0.15 |
| iLQR through the TinyPhysics network (expected value), receding horizon | **40.55** | eval.py | 64 |
| same iLQR, risk-sensitive (exact predicted variance in the cost) | **40.23** | lockstep | 70 |
| *published, not reproduced:* comma blog, CMA-ES-tuned feedback controller | 48.0 | blog.comma.ai/rlcontrols | |
| *published, not reproduced:* best PID+FF on the leaderboard | 50.608 | comma.ai/leaderboard | |

## What the results say

1. **The linear model is the limit, not the tuning.** CMA-ES over 13 MPC and observer parameters (cost weights, horizon, observer noise, speed breakpoints) only moves the linear MPC from 57.44 to 56.68. It finds very different settings that score almost the same, so the objective is flat. Planning through the learned network instead reaches 40.55 (risk-sensitive 40.23).
2. **Preview matters most.** The car responds to steer with about 3 steps (0.3 s) of delay. Feedforward on the planned target takes PID from 110 to the low 60s, and removing preview from the MPC costs about 42 points.
3. **Most of what remains is the simulator's noise.** With the random token draw replaced by the mean token (analysis only, dev segments), the iLQR's cost falls by 66%. Steering can widen that noise (any steer step does) but barely narrow it. Pricing the exact predicted variance in the iLQR cost gives the small, consistent gain in the table.
4. **Negative results.** A hybrid MPC on an 8-region piecewise-affine model, solved with a Graphs-of-Convex-Sets relaxation, has a loose relaxation (41% below the exact MIQP optimum). Its rounding is near-optimal (mean 4.7% above). In closed loop it is worse than the linear MPC (median dev segment 53.5 vs 50.7, with blow-ups) and runs at 597 ms per step. An exact expected-cost one-step lookahead fails because of the delay, and MPPI only works with smooth perturbations.
5. **Real time.** The simulator runs at 10 Hz. The iLQR takes 64 ms per step (max 66 ms, engagement step 64 ms) on one desktop thread, after warming its plan during the warmup steps instead of at engagement. The linear MPC takes under 0.5 ms.

## Evaluators and caveats

- `eval.py`: the challenge's own script, one process per segment. `run_eval.py`: our per-segment runner around the same `tinyphysics.run_rollout`; it reproduces eval.py exactly for PID and MPC. `lockstep`: a batched GPU evaluator that reproduces official per-segment costs bit for bit for PID, PID+FF and MPC on 1,500 dev segments. For the iLQR it matches only statistically (lockstep test: 40.56 iLQR, 40.23 risk-sensitive).
- The iLQR plans with the simulator's own network, so it has a perfect *mean* model. That puts it with the leaderboard's model-based MPC entries (33.6-44.3), not with feedback controllers. On a car the model would be learned and imperfect.
- One design decision was prompted by a test run: the iLQR's first test run (58.19) had two saturation blow-ups. The fix was designed and validated on non-test data and is disclosed in the README.
- Timings are on a desktop CPU (Ryzen 9 9950X), not a comma device.

Details: README.md, results/RESULTS.md, results/ABLATION.md, docs/WRITEUP.md.
