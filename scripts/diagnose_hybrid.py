"""Diagnose the hybrid MPC blow-up on dev segment 10009: per-step trajectory and failed solves
(saved to results/hybrid/diag_seg10009.json). Run from the repo root."""
import sys, logging, json, numpy as np
sys.path.insert(0, '.')
logging.getLogger('drake').setLevel(logging.WARNING)
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from ctrl.evaluate import seg_path
from ctrl.hybrid_mpc import HybridMPCController
c = HybridMPCController('results/pwa/pwa_yu_8.json', H=8, log=True)
sim = TinyPhysicsSimulator(TinyPhysicsModel('./models/tinyphysics.onnx', debug=False), seg_path(10009), controller=c)
cost = sim.rollout()
y = np.array(sim.current_lataccel_history); r = np.array(sim.target_lataccel_history); u = np.array(sim.action_history)
fails = [l['t'] for l in c.log if l['failed']]
err = np.abs(y - r)
first_bad = int(np.argmax(err[100:] > 1.0)) + 100 if (err[100:] > 1.0).any() else None
json.dump(dict(cost=cost, failed_steps=fails, first_err_gt_1=first_bad, y=y.tolist(), r=r.tolist(), u=u.tolist(),
               kf=None), open('results/hybrid/diag_seg10009.json', 'w'))
print('cost', {k: round(v, 1) for k, v in cost.items()}, 'failed steps', fails[:20], 'first |err|>1 at', first_bad)
if first_bad:
    for t in range(first_bad - 6, first_bad + 4):
        print(t, f"r={r[t]:+.2f} y={y[t]:+.2f} u={u[t]:+.3f}")
