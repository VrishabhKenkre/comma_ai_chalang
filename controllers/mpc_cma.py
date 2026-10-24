"""Preview MPC + disturbance observer with parameters tuned by CMA-ES on dev segments
(scripts/tune_mpc_cma.py; objective dev 10500-10999, check dev 11000-11499; results/tuning/mpc_cma.json).
Same controller as controllers/mpc.py; only weights, horizon, observer noise and speed breakpoints differ.
The ARX models are refitted for the tuned breakpoints (results/sysid/arx_models_cma.json)."""
from ctrl.arx_controller import ARXController

PARAMS = dict(w_track=5.078922315615144, w_du=34.11224237008714, H=8,
              kf=dict(q_d=0.0011971644741332224, q_k=2.7668992970397618e-05, r=0.0006499359563785198))


class Controller(ARXController):
  def __init__(self):
    super().__init__(solver="osqp", observer=True, models_path="results/sysid/arx_models_cma.json", **PARAMS)
