"""Receding-horizon iLQR through the TinyPhysics transformer (expected-value dynamics).

Dynamics: y_t = F(context_t) = E[lataccel_t | last 20 (u, roll, v, a) rows, last 20 y]
computed with the PyTorch port of the simulator model (ctrl/torch_model.py),
soft tokens so that F is differentiable w.r.t. past lataccel.

iLQR state at horizon step k (39 dims):
  x_k = [y_{t+k-1}, ..., y_{t+k-20}, u_{t+k-1}, ..., u_{t+k-19}],   input u_{t+k}
  x_{k+1} = [F(x_k, u_{t+k}), y_{t+k-1..t+k-19}, u_{t+k}, u_{t+k-1..t+k-18}]
Cost on x_{k+1} (challenge score / 100, plus a steer-rate term):
  0.5*(y_{t+k} - r_{t+k})^2 + (y_{t+k} - y_{t+k-1})^2 + w_du*(u_{t+k} - u_{t+k-1})^2
Roll, v_ego and a_ego over the horizon come from future_plan (known exogenous inputs).

Each control step: shift the previous plan, then Gauss-Newton iLQR iterations
(Jacobians by autograd in one batched pass, Riccati backward pass, forward
line search over several step sizes at once). The model's sampling noise is not
modelled (we plan on the mean); the controller replans every step from the
measured lataccel.

BatchILQR runs B segments side by side (for ctrl/batch_eval.py on a GPU). Every
operation is row-wise: a segment's actions depend only on its own inputs, and
the line search picks the step size per segment. `Controller` below is the
same code with B = 1, usable in the official eval.py.
"""
import numpy as np
import torch

from controllers import BaseController
from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH

from .torch_model import TinyPhysicsTorch

N = CONTEXT_LENGTH            # 20
NX = 2 * N - 1                # 39
IU = N                        # index of u_{t+k-1} in x
WARMUP_FF = (0.55, -0.6)      # guess of the (unseen) logged steer during warmup, see arx_controller.py
U_MAX = 2.0
T_MAX = 700                   # longest segment is 602 steps


class BatchILQR:
  def __init__(self, B, H=12, iters=1, first_iters=10, w_du=1.0, mu=1e-6,
               alphas=(1.0, 0.5, 0.25, 0.1, 0.0), shifts=(), var_weight=0.0, pre_steps=0,
               device="cpu", lin_chunk=512):
    if device == "cpu":
      torch.set_num_threads(1)
    self.dev = torch.device(device)
    self.model = TinyPhysicsTorch().to(self.dev)
    self.B, self.H, self.iters, self.first_iters, self.w_du, self.mu = B, H, iters, first_iters, w_du, mu
    # pre_steps > 0: warm up the plan during the last pre_steps warmup calls (one iteration each;
    # the simulator ignores the output then), so the engagement step needs only `iters` iterations
    # and stays inside the 100 ms control period. pre_steps = 0: first_iters iterations at engagement.
    self.pre_steps = pre_steps
    self.lin_chunk = lin_chunk
    # Risk-sensitive variant: add var_weight * sigma_k^2 per horizon step, sigma_k = std of the
    # model's predicted distribution (exact, over the bins). In our cost units the exact one-step
    # expectation of 0.5*(y - r)^2 + (y - y_prev)^2 is the mean cost + 1.5 * sigma^2, so
    # var_weight = 1.5 prices the noise the way the score does. 0 = plain iLQR on the mean.
    self.var_weight = var_weight
    self.alphas = torch.tensor(alphas, dtype=torch.float32, device=self.dev)
    # Zeroth-order candidates: the warm start shifted by these constant steer offsets.
    # When the model saturates (lataccel pinned at the +-5 token limit, steer at +-2)
    # its gradient vanishes and iLQR cannot leave the plateau; trying shifted plans can.
    # They are only considered for segments whose warm start is saturated, so normal
    # driving is unaffected.
    self.shifts = torch.tensor((0.0,) + tuple(shifts), dtype=torch.float32, device=self.dev)
    self.t = CONTEXT_LENGTH
    z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=self.dev)
    self.y, self.u, self.exo = z(B, T_MAX), z(B, T_MAX), z(B, T_MAX, 3)   # indexed by step
    self.U = None
    ar = lambda *a: torch.arange(*a, device=self.dev)
    self.ctx_idx = ar(N)[None, :] + ar(H)[:, None]                       # [H, 20] window positions
    self.ystate_idx = ar(N - 1, -1, -1)[None, :] + ar(H + 1)[:, None]    # y_{t+k-1..t+k-20}
    self.ustate_idx = ar(N - 1, 0, -1)[None, :] + ar(H + 1)[:, None]     # u_{t+k-1..t+k-19}
    # constant stage cost on x_{k+1}: 0.5 x'Qx - r*x[0]
    Q = torch.zeros(NX, NX, dtype=torch.float64, device=self.dev)
    Q[0, 0], Q[0, 1], Q[1, 0], Q[1, 1] = 3.0, -2.0, -2.0, 2.0
    Q[IU, IU] += 2 * w_du; Q[IU + 1, IU + 1] += 2 * w_du
    Q[IU, IU + 1] -= 2 * w_du; Q[IU + 1, IU] -= 2 * w_du
    self.Q = Q
    A = torch.zeros(NX, NX, dtype=torch.float64, device=self.dev)
    A[torch.arange(1, N), torch.arange(0, N - 1)] = 1.0
    A[torch.arange(IU + 1, NX), torch.arange(IU, NX - 1)] = 1.0
    self.A_shift = A

  # ------------------------------------------------------------------ pieces
  def _rollout(self, U, exo, y_hist, u_hist, feedback=None):
    """Roll the expected-value model forward from the measured history.
    U [R, H] with R = rows; exo [R, 20+H, 3]; y_hist, u_hist [R, 20].
    With feedback=(Ubar, Xbar, kappa, K, alpha) (all per row) the input is
    recomputed each step as u = ubar + alpha*kappa + K (x - xbar)."""
    R, H = U.shape
    u_full = torch.cat([u_hist, U], 1)                                # times t-20 .. t+H-1
    y_full = torch.cat([y_hist, torch.zeros(R, H, device=self.dev)], 1)
    sig = torch.zeros(R, H, device=self.dev)
    for k in range(H):
      if feedback is not None:
        Ubar, Xbar, kappa, K, alpha = feedback
        x = torch.cat([y_full[:, self.ystate_idx[k]], u_full[:, self.ustate_idx[k]]], 1)
        u_full[:, N + k] = (Ubar[:, k] + alpha * kappa[:, k]
                            + ((x - Xbar[:, k]) * K[:, k]).sum(1)).clamp(-U_MAX, U_MAX)
      states = torch.cat([u_full[:, k + 1:k + N + 1, None], exo[:, k + 1:k + N + 1]], 2)
      if self.var_weight:
        y_full[:, N + k], sig[:, k] = self.model.mean_std(states, y_full[:, k:k + N], soft_tokens=True)
      else:
        y_full[:, N + k] = self.model.expected(states, y_full[:, k:k + N], soft_tokens=True)
    return y_full[:, N:], u_full[:, N:], sig

  def _cost(self, Y, U, r, y_prev, u_prev, sig):
    yp = torch.cat([y_prev[:, None], Y[:, :-1]], 1)
    up = torch.cat([u_prev[:, None], U[:, :-1]], 1)
    J = (0.5 * (Y - r) ** 2 + (Y - yp) ** 2 + self.w_du * (U - up) ** 2).sum(1)
    return J + self.var_weight * (sig ** 2).sum(1) if self.var_weight else J

  def _linearize(self, Y, U, exo, y_hist, u_hist):
    """Nominal states X [B, H+1, 39], dF/dx_k [B, H, 39], dF/du_k [B, H]."""
    u_full = torch.cat([u_hist, U], 1)                                # [B, 20 + H]
    y_full = torch.cat([y_hist, Y], 1)
    X = torch.cat([y_full[:, self.ystate_idx], u_full[:, self.ustate_idx]], 2)
    Fx, Fu, Sx, Su = [], [], [], []
    to_x = lambda gy, gu: torch.cat([gy.flip(2), gu[:, :, :N - 1].flip(2)], 2)   # same ordering as X
    for lo in range(0, self.B, self.lin_chunk):
      sl = slice(lo, lo + self.lin_chunk)
      with torch.enable_grad():
        u_ctx = u_full[sl][:, self.ctx_idx + 1].detach().requires_grad_(True)   # [b, H, 20]
        y_ctx = y_full[sl][:, self.ctx_idx].detach().requires_grad_(True)
        b = u_ctx.shape[0]
        states = torch.cat([u_ctx[..., None], exo[sl][:, self.ctx_idx + 1]], 3).reshape(b * self.H, N, 4)
        if self.var_weight:
          mean, std = self.model.mean_std(states, y_ctx.reshape(b * self.H, N), soft_tokens=True)
          gu, gy = torch.autograd.grad(mean.sum(), (u_ctx, y_ctx), retain_graph=True)
          su, sy = torch.autograd.grad(std.sum(), (u_ctx, y_ctx))
          Sx.append(to_x(sy, su)); Su.append(su[:, :, N - 1])
        else:
          self.model.expected(states, y_ctx.reshape(b * self.H, N), soft_tokens=True).sum().backward()
          gu, gy = u_ctx.grad, y_ctx.grad                                # [b, H, 20]
      Fx.append(to_x(gy, gu))
      Fu.append(gu[:, :, N - 1])
    S = (torch.cat(Sx).double(), torch.cat(Su).double()) if self.var_weight else None
    return X.double(), torch.cat(Fx).double(), torch.cat(Fu).double(), S

  def _backward(self, X, Fx, Fu, r, S=None, sig=None):
    """Batched Gauss-Newton Riccati recursion -> kappa [B, H], K [B, H, 39].
    With S = (dsigma/dx, dsigma/du) the stage cost also has var_weight * sigma_k(x_k, u_k)^2,
    linearized as a least-squares residual (gradient 2c*sigma*grad, Hessian 2c*grad*grad')."""
    B, H, Q = X.shape[0], self.H, self.Q
    r = r.double()
    kappa = torch.zeros(B, H, dtype=torch.float64, device=self.dev)
    K = torch.zeros(B, H, NX, dtype=torch.float64, device=self.dev)
    Vx = X[:, H] @ Q
    Vx[:, 0] -= r[:, H - 1]
    Vxx = Q.expand(B, NX, NX).clone()
    Bv = torch.zeros(B, NX, dtype=torch.float64, device=self.dev)
    Bv[:, IU] = 1.0
    for k in range(H - 1, -1, -1):
      A = self.A_shift.expand(B, NX, NX).clone()
      A[:, 0] = Fx[:, k]
      Bv[:, 0] = Fu[:, k]
      Qx = torch.einsum("bij,bi->bj", A, Vx)
      Qu = (Bv * Vx).sum(1)
      VB = torch.einsum("bij,bj->bi", Vxx, Bv)
      Quu = (Bv * VB).sum(1) + self.mu
      Qux = torch.einsum("bi,bij->bj", VB, A)
      if S is not None:
        c, sk = 2.0 * self.var_weight, sig[:, k].double()
        sx, su = S[0][:, k], S[1][:, k]
        Qx = Qx + (c * sk)[:, None] * sx
        Qu = Qu + c * sk * su
        Quu = Quu + c * su * su
        Qux = Qux + (c * su)[:, None] * sx
      kappa[:, k] = -Qu / Quu
      K[:, k] = -Qux / Quu[:, None]
      Kk, kk = K[:, k], kappa[:, k]
      Vx = Qx + Kk * (Quu * kk)[:, None] + Kk * Qu[:, None] + Qux * kk[:, None]
      Vxx = (A.transpose(1, 2) @ Vxx @ A + Quu[:, None, None] * Kk[:, :, None] * Kk[:, None, :]
             + Kk[:, :, None] * Qux[:, None, :] + Qux[:, :, None] * Kk[:, None, :])
      if S is not None:
        Vxx = Vxx + c * sx[:, :, None] * sx[:, None, :]
      Vxx = 0.5 * (Vxx + Vxx.transpose(1, 2))
      if k > 0:                                    # stage cost on x_k
        Vx = Vx + X[:, k] @ Q
        Vx[:, 0] -= r[:, k - 1]
        Vxx = Vxx + Q
    return kappa, K

  # ------------------------------------------------------------------ main
  @torch.no_grad()
  def update_batch(self, targets, currents, states, plans, active):
    """One control step for all B segments (inactive rows get dummy inputs)."""
    t, H, B = self.t, self.H, self.B
    self.t += 1
    refs, rolls, fut = np.zeros((B, H)), np.zeros((B, H)), np.zeros((B, H - 1, 3))
    exo_now = np.zeros((B, 3))
    for i in range(B):
      if not active[i]:
        continue
      s, p = states[i], plans[i]
      exo_now[i] = (s.roll_lataccel, s.v_ego, s.a_ego)
      r = np.r_[targets[i], p.lataccel][:H]
      refs[i] = np.r_[r, np.full(H - len(r), r[-1])]
      ro = np.r_[s.roll_lataccel, p.roll_lataccel][:H]
      rolls[i] = np.r_[ro, np.full(H - len(ro), ro[-1])]
      f = np.column_stack([p.roll_lataccel, p.v_ego, p.a_ego])[:H - 1]
      if len(f) < H - 1:     # pad the plan at the end of a segment with its last row
        f = np.vstack([f, np.repeat(exo_now[i][None] if len(f) == 0 else f[-1:], H - 1 - len(f), 0)])
      fut[i] = f
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=self.dev)
    self.y[:, t - 1] = T(currents)
    self.exo[:, t] = T(exo_now)
    if t < CONTROL_START_IDX:
      # output ignored by the simulator; remember a guess of the logged steer
      self.u[:, t] = T(WARMUP_FF[0] * refs[:, 2] + WARMUP_FF[1] * rolls[:, 2])
      if self.pre_steps and t >= CONTROL_START_IDX - self.pre_steps:
        self._plan(t, refs, rolls, fut, T, self.iters)              # warm up the plan only
      return self.u[:, t].cpu().numpy().astype(np.float64)
    n_iter = self.first_iters if (t == CONTROL_START_IDX and not self.pre_steps) else self.iters
    U = self._plan(t, refs, rolls, fut, T, n_iter)
    self.u[:, t] = U[:, 0]
    return U[:, 0].cpu().numpy().astype(np.float64)

  def _plan(self, t, refs, rolls, fut, T, n_iter):
    """Warm start (or feedforward guess), saturation search, then n_iter iLQR iterations. Sets self.U."""
    H, B = self.H, self.B
    exo = torch.cat([self.exo[:, t - N:t + 1], T(fut)], 1)           # times t-20 .. t+H-1
    y_hist, u_hist, r = self.y[:, t - N:t], self.u[:, t - N:t], T(refs)
    if self.U is None:          # first planning step: static feedforward guess
      ext = lambda a: np.concatenate([a[:, 2:], np.repeat(a[:, -1:], 2, 1)], 1)
      self.U = T(WARMUP_FF[0] * ext(refs) + WARMUP_FF[1] * ext(rolls))
    else:                       # warm start: shift the previous plan by one step
      self.U = torch.cat([self.U[:, 1:], self.U[:, -1:]], 1)
    # nominal = best of the warm start and its shifted copies (one batched rollout)
    nS = len(self.shifts)
    repS = lambda a: a.repeat_interleave(nS, 0)
    Ucand = (repS(self.U) + self.shifts.repeat(B)[:, None]).clamp(-U_MAX, U_MAX)
    Ys, Us, Ss = self._rollout(Ucand, repS(exo), repS(y_hist), repS(u_hist))
    Js = self._cost(Ys, Us, repS(r), repS(y_hist[:, -1]), repS(u_hist[:, -1]), Ss).view(B, nS)
    Y0, U0 = Ys.view(B, nS, H)[:, 0], Us.view(B, nS, H)[:, 0]
    saturated = (U0.abs().amax(1) >= U_MAX - 1e-3) | (Y0.abs().amax(1) >= 4.9)
    Js[:, 1:] = torch.where(saturated[:, None], Js[:, 1:], torch.full_like(Js[:, 1:], float("inf")))
    rows = torch.arange(B, device=self.dev) * nS + Js.argmin(1)
    Y, U, Sg, J = Ys[rows], Us[rows], Ss[rows], Js.min(1).values
    nA = len(self.alphas)
    rep = lambda a: a.repeat_interleave(nA, 0)                        # row = b * nA + alpha
    for _ in range(n_iter):
      X, Fx, Fu, S = self._linearize(Y, U, exo, y_hist, u_hist)
      kappa, K = self._backward(X, Fx, Fu, r, S, Sg)
      fb = (rep(U), rep(X.float()), rep(kappa.float()), rep(K.float()), self.alphas.repeat(B))
      Yc, Uc, Sc = self._rollout(rep(U).clone(), rep(exo), rep(y_hist), rep(u_hist), feedback=fb)
      Jc = self._cost(Yc, Uc, rep(r), rep(y_hist[:, -1]), rep(u_hist[:, -1]), Sc).view(B, nA)
      best = Jc.argmin(1)
      rows = torch.arange(B, device=self.dev) * nA + best
      better = Jc[torch.arange(B, device=self.dev), best] < J          # per-segment acceptance
      Y = torch.where(better[:, None], Yc[rows], Y)
      U = torch.where(better[:, None], Uc[rows], U)
      Sg = torch.where(better[:, None], Sc[rows], Sg)
      J = torch.where(better, Jc[torch.arange(B, device=self.dev), best], J)
    self.U = U
    return U


class ILQRController(BaseController):
  """Official-API controller: the same iLQR with a batch of one segment."""
  def __init__(self, **kw):
    self.core = BatchILQR(1, **kw)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    return float(self.core.update_batch([target_lataccel], [current_lataccel], [state], [future_plan], [True])[0])
