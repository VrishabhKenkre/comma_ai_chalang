"""Preview MPC on the ARX model: problem construction, OSQP and ADMM solvers, LQR.

State at the current step t (n = ny + nu):
  x = [y_{t-1}, ..., y_{t-ny}, u_{t-1}, ..., u_{t-nu}]
Input is the steer *rate*  delta_k = u_{t+k} - u_{t+k-1}, so
  x_{k+1} = A x_k + B delta_k + f_k
where row 0 is the ARX equation (y_{t+k}), row ny is u_{t+k} = u_{t+k-1} + delta_k,
the other rows shift the histories, and f_k[0] holds the known exogenous part
(roll terms, ARX bias, observer offset).

Cost (the challenge score per step, divided by 100):
  sum_{k=0}^{H-1}  0.5*(y_{t+k} - r_{t+k})^2 + (y_{t+k} - y_{t+k-1})^2 + w_du * delta_k^2
y_{t+k} and y_{t+k-1} are x_{k+1}[0] and x_{k+1}[1], so this is a cost on states only.
Constraints: |u_{t+k}| <= 2 (state bound on x_{k+1}[ny]),  |delta_k| <= du_max.
"""
import numba
import numpy as np
import osqp
import scipy.linalg
import scipy.sparse as sp

U_MAX = 2.0


def state_space(m):
  """A, B for one ARX model (ny >= 2 so that y_{t+k-1} is in the next state)."""
  ny, nu = max(m.na, 2), max(m.nb, 1)
  n = ny + nu
  a = np.r_[m.a, np.zeros(ny - m.na)]
  A = np.zeros((n, n))
  B = np.zeros(n)
  A[0, :ny] = a
  A[0, ny] += m.b[0]                       # b_0 * u_{t+k} = b_0 * (u_{t+k-1} + delta)
  for j in range(1, m.nb + 1):
    A[0, ny + j - 1] += m.b[j]
  B[0] = m.b[0]
  for i in range(1, ny):
    A[i, i - 1] = 1.0
  A[ny, ny] = 1.0
  B[ny] = 1.0
  for j in range(1, nu):
    A[ny + j, ny + j - 1] = 1.0
  return A, B, ny, nu


def stage_cost(n, w_track=1.0, w_jerk=1.0):
  """0.5 x'Qx for w_track*0.5*(y-r)^2 + w_jerk*(y - y_prev)^2; the linear term -w_track*r on x[0]
  is added per step. w_track = w_jerk = 1 is the challenge score (/100)."""
  Q = np.zeros((n, n))
  Q[0, 0], Q[0, 1], Q[1, 0], Q[1, 1] = w_track + 2.0 * w_jerk, -2.0 * w_jerk, -2.0 * w_jerk, 2.0 * w_jerk
  return Q


class MPCProblem:
  """Everything that depends only on the speed bin's model: built once, reused every step."""

  def __init__(self, m, H=40, w_du=0.1, du_max=0.3, rho=1.0, w_track=1.0, w_jerk=1.0):
    self.m, self.H, self.du_max, self.w_track = m, H, du_max, w_track
    self.A, self.B, self.ny, self.nu = state_space(m)
    n = self.n = len(self.B)
    self.Q = stage_cost(n, w_track, w_jerk)
    self.R = 2.0 * w_du
    self._setup_osqp()
    self._setup_admm(rho)
    self.lqr_K = self._lqr_gain()

  # ---------- OSQP: sparse formulation, z = [x_1..x_H, delta_0..delta_{H-1}] ----------
  def _setup_osqp(self):
    H, n, A, B = self.H, self.n, self.A, self.B
    P = sp.block_diag([sp.kron(sp.eye(H), sp.csc_matrix(self.Q)), self.R * sp.eye(H)], format="csc")
    Ax = sp.kron(sp.eye(H), sp.eye(n)) - sp.kron(sp.eye(H, k=-1), sp.csc_matrix(A))
    Bu = sp.kron(sp.eye(H), sp.csc_matrix(B.reshape(-1, 1)))
    Aeq = sp.hstack([Ax, -Bu])
    sel_u = sp.hstack([sp.kron(sp.eye(H), sp.csc_matrix(np.eye(n)[[self.ny]])), sp.csc_matrix((H, H))])
    sel_d = sp.hstack([sp.csc_matrix((H, H * n)), sp.eye(H)])
    self.osqp_A = sp.vstack([Aeq, sel_u, sel_d], format="csc")
    self.osqp_l_bounds = np.r_[-U_MAX * np.ones(H), -self.du_max * np.ones(H)]
    self.osqp_u_bounds = -self.osqp_l_bounds
    self.solver = osqp.OSQP()
    self.solver.setup(P, np.zeros(H * n + H), self.osqp_A,
                      np.r_[np.zeros(H * n), self.osqp_l_bounds], np.r_[np.zeros(H * n), self.osqp_u_bounds],
                      verbose=False, eps_abs=1e-6, eps_rel=1e-6, polish=True, warm_starting=True)

  def solve_osqp(self, x0, f, r):
    """x0 [n], f [H, n] affine terms, r [H] references -> delta sequence [H]."""
    H, n = self.H, self.n
    q = np.zeros(H * n + H)
    q[0:H * n:n] = -self.w_track * r                # from 0.5*w*(y - r)^2 = 0.5*w*y^2 - w*r*y + const
    beq = f.copy()
    beq[0] += self.A @ x0
    beq = beq.reshape(-1)
    self.solver.update(q=q, l=np.r_[beq, self.osqp_l_bounds], u=np.r_[beq, self.osqp_u_bounds])
    res = self.solver.solve()
    return res.x[H * n:], res.x[:H * n].reshape(H, n)

  # ---------- ADMM with cached finite-horizon Riccati (TinyMPC style) ----------
  def _setup_admm(self, rho):
    """Cache the Riccati recursion of the rho-augmented LQR subproblem.

    Only the constrained state component (u, index ny) gets the rho penalty.
    For fixed (A, B, Q, R, rho, H) the gains K_k do not depend on the
    reference or constraints, so they are computed once per speed bin.
    """
    H, n, A, B = self.H, self.n, self.A, self.B
    self.rho = rho
    self.rho_x = np.zeros(n)
    self.rho_x[self.ny] = rho
    Qt = self.Q + np.diag(self.rho_x)
    Rt = self.R + rho
    P = Qt.copy()                       # terminal: x_H has the same stage cost
    K = np.zeros((H, n)); Minv = np.zeros(H); Acl_T = np.zeros((H, n, n)); Pn = np.zeros((H, n, n))
    for k in range(H - 1, -1, -1):
      Pn[k] = P                         # P_{k+1}
      M = Rt + B @ P @ B
      K[k] = (B @ P @ A) / M
      Minv[k] = 1.0 / M
      Acl = A - np.outer(B, K[k])
      Acl_T[k] = Acl.T
      P = Qt + A.T @ P @ Acl
    self.adm = dict(K=K, Minv=Minv, Acl_T=Acl_T, Pn=Pn, Rt=Rt)

  def solve_admm(self, x0, f, r, warm=None, max_iter=500, tol=1e-4, alpha=1.6):
    """Returns (delta [H], x [H, n], warm, iterations). `warm` is the previous call's warm."""
    H, n = self.H, self.n
    q = np.zeros((H + 1, n))
    q[1:, 0] = -self.w_track * r        # q[k] is the linear cost on x_k, k = 1..H
    if warm is None or warm[0].shape != (H + 1, n):
      z = np.zeros((H + 1, n)); y = np.zeros((H + 1, n)); w = np.zeros(H); g = np.zeros(H)
    else:                               # shift last solution by one step
      z, y, w, g = (np.concatenate([a[1:], a[-1:]]) for a in warm)
    x, d, z, y, w, g, it = _admm(self.A, self.B, self.adm["K"], self.adm["Minv"], self.adm["Acl_T"],
                                 self.adm["Pn"], self.adm["Rt"], self.rho_x, self.rho, x0, f, q,
                                 self.ny, U_MAX, self.du_max, z, y, w, g, max_iter, tol, alpha)
    return d, x[1:], (z, y, w, g), it

  # ---------- infinite-horizon LQR (ablation: no preview, no constraints) ----------
  def _lqr_gain(self):
    # stage cost on x_k with Q (the one-step shift does not matter at infinite horizon)
    P = scipy.linalg.solve_discrete_are(self.A, self.B.reshape(-1, 1), self.Q + 1e-9 * np.eye(self.n),
                                        np.array([[self.R]]))
    return (self.B @ P @ self.A) / (self.R + self.B @ P @ self.B)


@numba.njit(cache=True)
def _admm(A, B, K, Minv, Acl_T, Pn, Rt, rho_x, rho, x0, f, q, ny, u_max, du_max,
          z, y, w, g, max_iter, tol, alpha):
  H, n = K.shape
  x = np.zeros((H + 1, n))
  d = np.zeros(H)
  dff = np.zeros(H)
  for it in range(max_iter):
    # --- primal: LQR with linear terms, using cached gains ---
    p = q[H] + rho_x * (y[H] - z[H])
    for k in range(H - 1, -1, -1):
      rt = rho * (g[k] - w[k])
      Pf = Pn[k] @ f[k] + p
      dff[k] = Minv[k] * (B @ Pf + rt)
      if k > 0:
        qt = q[k] + rho_x * (y[k] - z[k])
        p = qt + Acl_T[k] @ (Pf - Pn[k] @ B * dff[k]) + K[k] * (Rt * dff[k] - rt)
    x[0] = x0
    for k in range(H):
      d[k] = -(K[k] @ x[k]) - dff[k]
      x[k + 1] = A @ x[k] + B * d[k] + f[k]
    # --- slack: project the over-relaxed iterate onto the box constraints ---
    z_old = z.copy(); w_old = w.copy()
    xr = alpha * x[:, ny] + (1 - alpha) * z_old[:, ny]
    dr = alpha * d + (1 - alpha) * w_old
    for k in range(1, H + 1):
      z[k] = x[k] + y[k]
      z[k, ny] = min(max(xr[k] + y[k, ny], -u_max), u_max)
    for k in range(H):
      w[k] = min(max(dr[k] + g[k], -du_max), du_max)
    # --- dual update and residuals (only the constrained component matters) ---
    prim = 0.0; dual = 0.0
    for k in range(1, H + 1):
      y[k, ny] += xr[k] - z[k, ny]
      prim = max(prim, abs(x[k, ny] - z[k, ny]))
      dual = max(dual, rho * abs(z[k, ny] - z_old[k, ny]))
    for k in range(H):
      g[k] += dr[k] - w[k]
      prim = max(prim, abs(d[k] - w[k]))
      dual = max(dual, rho * abs(w[k] - w_old[k]))
    if prim < tol and dual < tol:
      break
  return x, d, z, y, w, g, it + 1
