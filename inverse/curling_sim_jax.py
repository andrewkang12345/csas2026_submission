#!/usr/bin/env python3

# curling_sim_jax.py
# Simplified curling simulator (JAX) with stable contacts and differentiable rollout.
# Variable-N pre-placed stones supported. Hard loss is identity-agnostic (via permutations).
# IMPORTANT: Permutations are computed on the HOST and can be passed into the loss.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, NamedTuple, Optional
import itertools
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray

# ----------------------------
# Parameters & State
# ----------------------------

@dataclass(frozen=True)
class CurlingParams:
    # Geometry & mass
    radius: float = 0.145
    mass: float = 19.96
    # Integration (balanced for speed + stability)
    dt: float = 0.02
    substeps: int = 2
    max_steps: int = 1500       # hard upper bound (seconds ~= dt * max_steps)
    v_stop: float = 0.03
    v_cap: float = 6.0          # clamp linear speed to avoid blow-ups
    # Ice/friction/drag
    a_linear: float = 0.11
    c_quadratic: float = 0.0
    # Curl lateral term
    k_curl: float = 0.10
    gamma_spin: float = 0.15
    curl_speed_cap: float = 2.5
    # Smooth contact (penalty) parameters
    k_penalty: float = 2.5e4
    c_damp: float = 220.0
    contact_fmax: float = 2.0e4
    # Layout
    hog_to_tee: float = 6.401
    # Optional walls (off by default)
    enable_walls: bool = False
    wall_half_extent_x: float = 2.5
    wall_half_extent_y: float = 2.5
    wall_k: float = 1e5
    wall_c: float = 300.0
    # Sleep deadband for idle stones (m/s)
    sleep_v_thresh: float = 1e-5

class SimState(NamedTuple):
    pos: Array   # (N,2)
    vel: Array   # (N,2)
    omega: Array # (N,)
    t: Array     # ()
    done: Array  # ()

# ----------------------------
# Physics
# ----------------------------

def _clip_speed(v: Array, vcap: float) -> Array:
    sp = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v * jnp.minimum(1.0, vcap / (sp + 1e-8))

def _ice_forces(vel: Array, omega: Array, p: CurlingParams) -> Tuple[Array, Array]:
    speed = jnp.linalg.norm(vel, axis=-1, keepdims=True)
    vhat  = vel / (speed + 1e-8)

    a_lin = -p.a_linear * vhat
    a_quad = -p.c_quadratic * speed * vel

    # curl, capped by speed
    perp_vhat = jnp.stack([-vhat[..., 1], vhat[..., 0]], axis=-1)
    s_cap = p.curl_speed_cap
    s_eff = s_cap * jnp.tanh(speed / s_cap)
    a_lat = p.k_curl * omega[..., None] * perp_vhat * s_eff

    acc = a_lin + a_quad + a_lat
    f   = p.mass * acc
    tau = jnp.zeros_like(omega)
    return f, tau

def _pairwise_contact_forces(pos: Array, vel: Array, p: CurlingParams) -> Tuple[Array, Array]:
    """
    Strictly-zero forces when stones do not overlap.
    Damping applies only while overlapping.
    """
    N = pos.shape[0]
    d = pos[jnp.newaxis, :, :] - pos[:, jnp.newaxis, :]      # (N,N,2) = pos_j - pos_i
    dist = jnp.linalg.norm(d, axis=-1) + 1e-9                # (N,N)
    n = d / dist[..., None]                                  # unit(i->j)
    mask = 1.0 - jnp.eye(N)

    r_sum = 2.0 * p.radius
    raw_overlap = r_sum - dist                               # (N,N)
    overlap = jnp.maximum(raw_overlap, 0.0) * mask           # (N,N)

    rel_v = vel[jnp.newaxis, :, :] - vel[:, jnp.newaxis, :]  # (N,N,2)
    v_n = jnp.sum(rel_v * n, axis=-1)                        # (N,N)

    contact_gate = (raw_overlap > 0.0).astype(vel.dtype) * mask
    v_n = v_n * contact_gate

    f_mag = p.k_penalty * overlap - p.c_damp * v_n
    f_mag = jnp.clip(f_mag, 0.0, p.contact_fmax) * contact_gate

    f_ij = f_mag[..., None] * n                               # (N,N,2)
    f_on_i = -jnp.sum(f_ij, axis=1)                           # (N,2)

    torque = jnp.zeros((N,))
    return f_on_i, torque

def _wall_forces(pos: Array, vel: Array, p: CurlingParams) -> Array:
    if not p.enable_walls:
        return jnp.zeros_like(pos)
    fx = jnp.zeros(pos.shape[0])
    fy = jnp.zeros(pos.shape[0])
    # x walls
    overlap_left  = p.radius - (pos[:, 0] + p.wall_half_extent_x)
    overlap_right = p.radius - (p.wall_half_extent_x - pos[:, 0])
    f_left  = jnp.maximum(overlap_left,  0.0) * p.wall_k
    f_right = jnp.maximum(overlap_right, 0.0) * p.wall_k
    fx = fx + f_left - f_right
    # y walls
    overlap_bottom = p.radius - (pos[:, 1] + p.wall_half_extent_y)
    overlap_top    = p.radius - (p.wall_half_extent_y - pos[:, 1])
    f_bottom = jnp.maximum(overlap_bottom, 0.0) * p.wall_k
    f_top    = jnp.maximum(overlap_top,    0.0) * p.wall_k
    fy = fy + f_bottom - f_top
    return jnp.stack([fx, fy], axis=-1)

# ----------------------------
# Integrator
# ----------------------------

def _micro_step(p: CurlingParams, state: SimState, micro_dt: float) -> SimState:
    pos, vel, omega, t, done = state

    f_ice, _  = _ice_forces(vel, omega, p)
    f_pair, _ = _pairwise_contact_forces(pos, vel, p)
    f_wall    = _wall_forces(pos, vel, p)
    f_total   = f_ice + f_pair + f_wall

    vel_new   = _clip_speed(vel + (f_total / p.mass) * micro_dt, p.v_cap)
    pos_new   = pos + vel_new * micro_dt
    omega_new = omega + (-p.gamma_spin * omega) * micro_dt
    t_new     = t + micro_dt

    # Sleep: zero tiny velocities only if NOT in contact
    d_new = pos_new[jnp.newaxis, :, :] - pos_new[:, jnp.newaxis, :]
    dist_new = jnp.linalg.norm(d_new, axis=-1)                       # (N,N)
    in_contact = jnp.any((2.0 * p.radius - dist_new) > 0.0, axis=1)  # (N,)
    vmag_s = jnp.linalg.norm(vel_new, axis=-1)                       # (N,)
    sleep_mask = ((vmag_s < p.sleep_v_thresh) & (~in_contact))[:, None]  # (N,1)
    vel_new = jnp.where(sleep_mask, jnp.zeros_like(vel_new), vel_new)    # (N,2)

    # Finite & stop conditions
    finite = (
        jnp.all(jnp.isfinite(pos_new)) &
        jnp.all(jnp.isfinite(vel_new)) &
        jnp.all(jnp.isfinite(omega_new))
    )
    pos   = jnp.where(finite, pos_new, pos)
    vel   = jnp.where(finite, vel_new, vel)
    omega = jnp.where(finite, omega_new, omega)
    done  = jnp.logical_or(done, jnp.logical_not(finite))

    speed = jnp.linalg.norm(vel, axis=-1)
    all_slow = jnp.all(speed < p.v_stop)
    done = jnp.logical_or(done, all_slow)

    return SimState(pos=pos, vel=vel, omega=omega, t=t_new, done=done)

def step(p: CurlingParams, s: SimState) -> SimState:
    """Semi-implicit Euler with substeps."""
    micro_dt = p.dt / p.substeps

    def body(carry, _):
        s = carry
        s_next = _micro_step(p, s, micro_dt)
        # If already done, keep previous state
        s = jax.tree_util.tree_map(lambda a, b: jnp.where(s.done, a, b), s, s_next)
        return s, None

    s_final, _ = lax.scan(body, s, None, length=p.substeps)
    return s_final

# ----------------------------
# Rollouts
# ----------------------------

def simulate_fixed(p: CurlingParams, s0: SimState) -> SimState:
    """Differentiable fixed-length rollout with early-stop mask."""
    def _body(carry, _):
        s = carry
        s_next = step(p, s)
        s = jax.tree_util.tree_map(lambda a, b: jnp.where(s_next.done, a, b), s, s_next)
        return s, None
    s_final, _ = lax.scan(_body, s0, None, length=p.max_steps)
    return s_final

def simulate_until_stop(p: CurlingParams, s0: SimState) -> Tuple[SimState, Array]:
    """Dynamic early-stop rollout using lax.while_loop (forward-only)."""
    def cond(carry):
        s, i = carry
        return jnp.logical_and(jnp.logical_not(s.done), i < p.max_steps)
    def body(carry):
        s, i = carry
        return (step(p, s), i + 1)
    (s_final, steps) = lax.while_loop(cond, body, (s0, jnp.array(0, dtype=jnp.int32)))
    return s_final, steps

def rollout_positions_until_stop(p: CurlingParams, s0: SimState) -> Tuple[Array, Array]:
    """
    Dynamic rollout that records positions until stop.
    Returns (traj[:T, N, 2], T) with T <= max_steps and includes t=0.
    """
    N = s0.pos.shape[0]
    traj0 = jnp.zeros((p.max_steps + 1, N, 2), dtype=s0.pos.dtype).at[0].set(s0.pos)

    def cond(carry):
        s, i, traj = carry
        return jnp.logical_and(jnp.logical_not(s.done), i < p.max_steps)

    def body(carry):
        s, i, traj = carry
        s_next = step(p, s)
        traj = traj.at[i + 1].set(s_next.pos)
        return (s_next, i + 1, traj)

    (s_final, steps, traj) = lax.while_loop(cond, body, (s0, jnp.array(0, jnp.int32), traj0))
    return traj[: steps + 1], steps + 1

# ----------------------------
# Helpers & Loss
# ----------------------------

def make_initial_state(
    p: CurlingParams,
    prev_positions_button: Array,      # (N_prev,2)
    shooter_from_hog_angle: float,
    speed: float,
    omega_release: float,
    shooter_centerline_y: float = 0.0
) -> SimState:
    N_prev = prev_positions_button.shape[0]
    shooter_pos = jnp.array([-p.hog_to_tee, shooter_centerline_y])
    v = jnp.array([jnp.cos(shooter_from_hog_angle), jnp.sin(shooter_from_hog_angle)]) * speed
    pos = jnp.concatenate([prev_positions_button, shooter_pos[None, :]], axis=0)       # (N_prev+1,2)
    vel = jnp.concatenate([jnp.zeros_like(prev_positions_button), v[None, :]], axis=0) # (N_prev+1,2)
    omega = jnp.concatenate([jnp.zeros((N_prev,)), jnp.array([omega_release])], axis=0)
    return SimState(pos=pos, vel=vel, omega=omega, t=jnp.array(0.0), done=jnp.array(False))

def make_permutation_indices(n: int) -> jnp.ndarray:
    """
    Build permutation indices on the HOST (no JIT). Use this outside of jitted code
    and then pass the resulting JAX array into jitted functions.
    """
    if n > 7:
        raise ValueError(f"Permutation-based matching too large (n={n}). "
                         f"Reduce N or implement a non-factorial matcher.")
    perm_np = np.array(list(itertools.permutations(range(n))), dtype=np.int32)
    return jnp.asarray(perm_np)

def hard_final_loss_from_x(
    p: CurlingParams,
    prev_positions_button: Array,
    target_positions_button: Array,  # (N_total,2)
    x: Array,                         # [speed, angle, spin, y0]
    *,
    perms: Optional[Array] = None     # (P,N_total) indices; pass from host for JIT use
) -> Array:
    """
    Hard (GIF-accurate) loss:
    - simulate dynamically to stop
    - take final positions (N_total,2)
    - compute sum of squared distances under best permutation (ID-agnostic)

    If `perms` is None, we compute them on the host here (OK outside JIT).
    """
    speed, theta, omega, y0 = x
    s0 = make_initial_state(p, prev_positions_button, theta, speed, omega, y0)
    sf, _ = simulate_until_stop(p, s0)                # SimState final
    pred_final = sf.pos                                # (N_total,2)
    N = pred_final.shape[0]

    if perms is None:
        perms = make_permutation_indices(int(N))       # HOST-built, fine outside JIT

    pred_perm = jnp.take(pred_final, perms, axis=0)    # (P,N,2)
    diffs = pred_perm - target_positions_button[None, :, :]
    d2 = jnp.sum(diffs**2, axis=(1, 2))                # (P,)
    return jnp.min(d2)

# --- Public API ---

def simulate_from_params(
    p: CurlingParams,
    prev_positions_button: Array,
    x: Array,
    *,
    dynamic: bool = False
) -> Array:
    """
    If dynamic=False: return FINAL positions only, shape (N_total,2), using fixed rollout.
    If dynamic=True:  return the whole trajectory, shape (T,N_total,2) with T <= max_steps+1.
    """
    speed, theta, omega, y0 = x
    s0 = make_initial_state(p, prev_positions_button, theta, speed, omega, y0)
    if dynamic:
        traj, _ = rollout_positions_until_stop(p, s0)
        return traj
    else:
        sf = simulate_fixed(p, s0)
        return sf.pos
