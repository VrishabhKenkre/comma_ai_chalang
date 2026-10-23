"""Variance-aware controllers through the TinyPhysics model (step 1 of the research branch).

Both use the model's full predicted distribution over the 1024 lataccel bins, never a
simulator sample. In our cost units (score / 100 per step) a step costs
  c(y) = 0.5*(y - r)^2 + (y - y_prev)^2
and, given the context, its exact expectation is  sum_i p_i * c(bin_i).

BatchLookahead  (i)  choose u_t on a grid around u_{t-1}, holding it for d+1 steps, to
                     minimize sum_{k=0..d} E[c(y_{t+k})] + w_du*(u_t - u_{t-1})^2. The
                     expectation is exact at every step; the y_{t+k-1} that step k conditions
                     on is the previous step's mean. d = 0 is the pure one-step lookahead.
BatchMPPI       (ii) sampling MPC: perturb the warm-start plan with Gaussian noise (our own
                     torch.Generator, seeded; the simulator's RNG is never touched), roll out on
                     the mean, score each rollout by the mean cost + var_weight * sum sigma_k^2
                     (moment matching) + steer-rate cost, and average with weights exp(-J/lam).

Both run B segments side by side (ctrl/batch_eval.py); every operation is row-wise.
"""
import numpy as np
import torch

from tinyphysics import CONTROL_START_IDX, CONTEXT_LENGTH

from .torch_model import TinyPhysicsTorch, TEMPERATURE

N = CONTEXT_LENGTH
WARMUP_FF = (0.55, -0.6)
U_MAX = 2.0
T_MAX = 700


class _Base:
  """History bookkeeping shared with ctrl/ilqr.py's BatchILQR (same conventions)."""

  def __init__(self, B, H, device):
    if device == "cpu":
      torch.set_num_threads(1)
    self.dev = torch.device(device)
    self.model = TinyPhysicsTorch().to(self.dev)
    self.B, self.H, self.t = B, H, CONTEXT_LENGTH
    z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=self.dev)
    self.y, self.u, self.exo = z(B, T_MAX), z(B, T_MAX), z(B, T_MAX, 3)

  def _ingest(self, targets, currents, states, plans, active):
    t, H, B = self.t, self.H, self.B
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
      if len(f) < H - 1:
        f = np.vstack([f, np.repeat(exo_now[i][None] if len(f) == 0 else f[-1:], H - 1 - len(f), 0)])
      fut[i] = f
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=self.dev)
    self.y[:, t - 1] = T(currents)
    self.exo[:, t] = T(exo_now)
    return refs, rolls, T(fut), T

  def _probs(self, states, past_y):
    return (self.model.logits(states, past_y, soft_tokens=True)[:, -1] / TEMPERATURE).softmax(-1)

  def _rollout(self, U, exo, y_hist, u_hist, r, y_prev, exact=True):
    """Roll the mean forward for plans U [R, H]. Returns per-step expected cost [R, H]
    (exact over the bins if exact, else on the mean), means [R, H] and std [R, H]."""
    R, H = U.shape
    u_full = torch.cat([u_hist, U], 1)
    y_full = torch.cat([y_hist, torch.zeros(R, H, device=self.dev)], 1)
    cost = torch.zeros(R, H, device=self.dev)
    sig = torch.zeros(R, H, device=self.dev)
    bins = self.model.bins
    prev = y_prev
    for k in range(H):
      states = torch.cat([u_full[:, k + 1:k + N + 1, None], exo[:, k + 1:k + N + 1]], 2)
      p = self._probs(states, y_full[:, k:k + N])
      mean = p @ bins
      if exact:   # sum over bins: E[0.5 (y - r)^2 + (y - prev)^2]
        cost[:, k] = (p * (0.5 * (bins - r[:, k:k + 1]) ** 2 + (bins - prev[:, None]) ** 2)).sum(-1)
      else:
        cost[:, k] = 0.5 * (mean - r[:, k]) ** 2 + (mean - prev) ** 2
      sig[:, k] = torch.sqrt((p * (bins - mean[:, None]) ** 2).sum(-1) + 1e-12)
      y_full[:, N + k] = mean
      prev = mean
    return cost, y_full[:, N:], sig


class BatchLookahead(_Base):
  def __init__(self, B, d=0, w_du=2.0, grid=np.linspace(-0.2, 0.2, 41), device="cpu"):
    super().__init__(B, max(d + 1, 3), device)
    self.d, self.w_du = d, w_du
    self.grid = torch.tensor(grid, dtype=torch.float32, device=self.dev)

  @torch.no_grad()
  def update_batch(self, targets, currents, states, plans, active):
    refs, rolls, fut, T = self._ingest(targets, currents, states, plans, active)
    t, B, G, D = self.t, self.B, len(self.grid), self.d + 1
    self.t += 1
    if t < CONTROL_START_IDX:
      self.u[:, t] = T(WARMUP_FF[0] * refs[:, 2] + WARMUP_FF[1] * rolls[:, 2])
      return self.u[:, t].cpu().numpy().astype(np.float64)
    exo = torch.cat([self.exo[:, t - N:t + 1], fut], 1)[:, :N + D]
    y_hist, u_hist, r = self.y[:, t - N:t], self.u[:, t - N:t], T(refs)[:, :D]
    rep = lambda a: a.repeat_interleave(G, 0)
    u_cand = (rep(u_hist[:, -1]) + self.grid.repeat(B)).clamp(-U_MAX, U_MAX)          # [B*G]
    cost, _, _ = self._rollout(u_cand[:, None].expand(-1, D).contiguous(), rep(exo), rep(y_hist),
                               rep(u_hist), rep(r), rep(y_hist[:, -1]), exact=True)
    J = cost.sum(1) + self.w_du * (u_cand - rep(u_hist[:, -1])) ** 2
    best = J.view(B, G).argmin(1)
    u = u_cand.view(B, G)[torch.arange(B, device=self.dev), best]
    self.u[:, t] = u
    return u.cpu().numpy().astype(np.float64)


class BatchMPPI(_Base):
  def __init__(self, B, H=12, n_samples=64, noise_std=0.05, lam=0.01, iters=1, w_du=2.0,
               var_weight=1.5, noise="iid", seed=0, device="cpu"):
    """noise = "iid" (independent per horizon step) or "ramp" (offset + linear ramp over the
    horizon, two numbers per sample: smoother candidate plans)."""
    super().__init__(B, H, device)
    self.n, self.noise_std, self.lam, self.iters, self.noise = n_samples, noise_std, lam, iters, noise
    self.w_du, self.var_weight = w_du, var_weight
    self.gen = torch.Generator(device=self.dev).manual_seed(seed)      # our own RNG
    self.U = None

  def _score(self, U, exo, y_hist, u_hist, r):
    cost, _, sig = self._rollout(U, exo, y_hist, u_hist, r, y_hist[:, -1], exact=False)
    up = torch.cat([u_hist[:, -1:], U[:, :-1]], 1)
    return cost.sum(1) + self.var_weight * (sig ** 2).sum(1) + self.w_du * ((U - up) ** 2).sum(1)

  @torch.no_grad()
  def update_batch(self, targets, currents, states, plans, active):
    refs, rolls, fut, T = self._ingest(targets, currents, states, plans, active)
    t, B, H, n = self.t, self.B, self.H, self.n
    self.t += 1
    if t < CONTROL_START_IDX:
      self.u[:, t] = T(WARMUP_FF[0] * refs[:, 2] + WARMUP_FF[1] * rolls[:, 2])
      return self.u[:, t].cpu().numpy().astype(np.float64)
    exo = torch.cat([self.exo[:, t - N:t + 1], fut], 1)
    y_hist, u_hist, r = self.y[:, t - N:t], self.u[:, t - N:t], T(refs)
    if self.U is None:
      ext = lambda a: np.concatenate([a[:, 2:], np.repeat(a[:, -1:], 2, 1)], 1)
      self.U = T(WARMUP_FF[0] * ext(refs) + WARMUP_FF[1] * ext(rolls))
    else:
      self.U = torch.cat([self.U[:, 1:], self.U[:, -1:]], 1)
    rep = lambda a: a.repeat_interleave(n, 0)
    for _ in range(self.iters):
      if self.noise == "ramp":
        ab = torch.randn(B, n, 2, generator=self.gen, device=self.dev) * self.noise_std
        ramp = torch.linspace(0, 1, H, device=self.dev)
        eps = ab[:, :, :1] + ab[:, :, 1:] * ramp
      else:
        eps = torch.randn(B, n, H, generator=self.gen, device=self.dev) * self.noise_std
      eps[:, 0] = 0.0                                                   # keep the warm start itself
      Uc = (self.U[:, None] + eps).clamp(-U_MAX, U_MAX).reshape(B * n, H)
      J = self._score(Uc, rep(exo), rep(y_hist), rep(u_hist), rep(r)).view(B, n)
      w = torch.softmax(-(J - J.min(1, keepdim=True).values) / self.lam, 1)
      self.U = (w[:, :, None] * Uc.view(B, n, H)).sum(1)
    self.u[:, t] = self.U[:, 0]
    return self.U[:, 0].cpu().numpy().astype(np.float64)
