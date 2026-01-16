#!/usr/bin/env python3

# curling_inverse.py
# CEM inverse solver with a HARD loss that supports stones going off-sheet.
# The loss does rectangular matching:
# - Only predicted stones that end IN-BOUNDS are eligible to match to targets.
# - Targets are the IN-BOUNDS stones from your CSV next-state.
# - Any *unmatched target* gets a fixed miss penalty.
# - Any *unmatched in-bounds prediction* (i.e., stone that should have been removed but wasn't)
#   also gets a fixed penalty.
#
# This keeps the solver faithful when stones are knocked out (disappear).

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import logging
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, lax

from curling_sim_jax import (
    CurlingParams,
    simulate_from_params,
)

log = logging.getLogger("curling")

# ----------------------------
# Board extents (in *sim meters*)
# These match the CSV→meters mapping used in your viz scripts:
#   x_m = (y_csv - 800) * 0.003048         # along-sheet (button at 0)
#   y_m = (x_csv - 750) * 0.003048         # lateral (centerline at 0)
#
# CSV y ∈ [0,3000]  -> x_m ∈ [-2.4384, +6.7056]
# CSV x ∈ [0,1500]  -> y_m ∈ [-2.2860, +2.2860]
# ----------------------------
MIN_X = -2.4384
MAX_X =  6.7056
MIN_Y = -2.2860
MAX_Y =  2.2860

# Upper bounds (m^2). Effective penalty per stone becomes <= these caps.
MISS_TARGET_MAX    = 4.0
EXTRA_PRED_MAX     = 4.0
# Soft “how close to dead?” length scale (meters)
DEATH_MARGIN_SCALE = 0.20
BIG                 = 1e6  # internal masking for greedy assignment


# ----------------------------
# Bounds & helpers
# ----------------------------

@dataclass(frozen=True)
class SolveBounds:
    speed_min: float = 0.1
    speed_max: float = 3.0
    angle_min: float = -0.35
    angle_max: float = 0.35
    spin_min: float = -3.0
    spin_max: float = 3.0
    y0_min: float = -0.6
    y0_max: float = 0.6

def _make_bounds_arrays(b: SolveBounds):
    lo = jnp.array([b.speed_min, b.angle_min, b.spin_min, b.y0_min])
    hi = jnp.array([b.speed_max, b.angle_max, b.spin_max, b.y0_max])
    span = hi - lo
    return lo, hi, span

def _x_phys_from_x01(x01: jnp.ndarray, lo: jnp.ndarray, span: jnp.ndarray) -> jnp.ndarray:
    return lo + jnp.clip(x01, 0.0, 1.0) * span

def _x01_from_x_phys(x: jnp.ndarray, lo: jnp.ndarray, span: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip((x - lo) / (span + 1e-8), 0.0, 1.0)

def _safe_metric(v: jnp.ndarray) -> jnp.ndarray:
    return jnp.nan_to_num(v, nan=1e9, posinf=1e9, neginf=1e9)

# ----------------------------
# HARD loss that allows stones to disappear (off-sheet)
# ----------------------------

def _in_bounds_mask(pos: jnp.ndarray) -> jnp.ndarray:
    """Boolean mask for stones that end within the board rectangle."""
    x = pos[:, 0]
    y = pos[:, 1]
    return (x > MIN_X) & (x < MAX_X) & (y > MIN_Y) & (y < MAX_Y)

def _rect_greedy_cost(pred_final: jnp.ndarray, target_xy: jnp.ndarray) -> jnp.ndarray:
    """
    Greedy rectangular assignment cost between IN-BOUNDS predictions and targets,
    with distance-aware penalties for unmatched items based on proximity to the boundary.
    """
    Np = pred_final.shape[0]
    Nt = target_xy.shape[0]

    # Which predictions are eligible to match (must be in-bounds)
    inb = _in_bounds_mask(pred_final)
    N_in = jnp.sum(inb.astype(jnp.int32))

    # Early-out: no targets: pay a soft penalty for each in-bounds predicted stone (unmatched)
    if Nt == 0:
        # Weight by how far inside those predictions are (near edge ⇒ smaller penalty)
        margins_pred = _signed_margin(pred_final)
        w_pred = _soft_weight_from_margin(margins_pred) * inb.astype(pred_final.dtype)
        return EXTRA_PRED_MAX * jnp.sum(w_pred)

    # Pairwise squared distances for (eligible rows) x (targets)
    d2 = jnp.sum((pred_final[:, None, :] - target_xy[None, :, :]) ** 2, axis=2)  # (Np, Nt)
    C = d2 + (~inb)[:, None] * BIG  # mask out-of-bounds prediction rows

    # Greedy select K = min(N_in, Nt) matches; track which rows/cols were used
    K = jnp.minimum(N_in, jnp.int32(Nt))
    used_r = jnp.zeros((Np,), dtype=jnp.bool_)
    used_c = jnp.zeros((Nt,), dtype=jnp.bool_)
    total  = jnp.array(0.0, dtype=pred_final.dtype)

    def body(s, carry):
        C_curr, total_curr, used_r_curr, used_c_curr = carry
        # argmin (global)
        idx_flat = jnp.argmin(C_curr)
        i = idx_flat // Nt
        j = idx_flat %  Nt
        val = C_curr[i, j]

        def _do_update(_):
            total_new = total_curr + val
            used_r_new = used_r_curr.at[i].set(True)
            used_c_new = used_c_curr.at[j].set(True)
            # Invalidate selected row/col
            C1 = C_curr.at[i, :].set(BIG)
            C1 = C1.at[:, j].set(BIG)
            return (C1, total_new, used_r_new, used_c_new)

        def _skip(_):
            return (C_curr, total_curr, used_r_curr, used_c_curr)

        # Only actually select while s < K (after that, loop is a no-op)
        return lax.cond(s < K, _do_update, _skip, operand=None)

    C0 = C
    carry0 = (C0, total, used_r, used_c)
    C_fin, matched_sum, used_r_fin, used_c_fin = lax.fori_loop(0, Nt, body, carry0)

    # Compute soft penalties for the unmatched ones
    # Targets are *by construction* in-bounds. Weight by how far inside each target is.
    tgt_margins = _signed_margin(target_xy)                    # (Nt,)
    tgt_w       = _soft_weight_from_margin(tgt_margins)        # (Nt,)
    tgt_unmatched_mask = (~used_c_fin).astype(pred_final.dtype)
    miss_target_pen = MISS_TARGET_MAX * jnp.sum(tgt_w * tgt_unmatched_mask)

    # In-bounds predicted stones left unmatched
    pred_margins = _signed_margin(pred_final)                  # (Np,)
    pred_w       = _soft_weight_from_margin(pred_margins)      # (Np,)
    pred_unmatched_mask = (inb & (~used_r_fin)).astype(pred_final.dtype)
    extra_pred_pen = EXTRA_PRED_MAX * jnp.sum(pred_w * pred_unmatched_mask)

    return matched_sum + miss_target_pen + extra_pred_pen

def _hard_loss_allow_deaths(p: CurlingParams,
                            prev_positions_button: jnp.ndarray,
                            target_positions_button: jnp.ndarray,
                            x: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-friendly hard loss:
    - Roll out to final positions.
    - Compute greedy rectangular cost between IN-BOUNDS predictions and targets,
      with penalties for missing targets and extra in-bounds predictions.
    """
    # Final positions (N_prev+1, 2); uses early-stop mask internally
    pred_final = simulate_from_params(p, prev_positions_button, x, dynamic=False)  # (Np,2)
    return _rect_greedy_cost(pred_final, target_positions_button)

def _make_batched_hard_loss(p, prev, target):
    # Keep it JIT + vmap for CEM speed
    loss_one = lambda x_phys: _hard_loss_allow_deaths(p, prev, target, x_phys)
    return jax.jit(jax.vmap(loss_one))

# Signed margin (meters) to the rectangle edges.
# Positive = in-bounds distance to nearest edge; Negative = overshoot outside.
def _signed_margin(pos: jnp.ndarray) -> jnp.ndarray:
    x = pos[:, 0]; y = pos[:, 1]
    m = jnp.minimum(jnp.minimum(x - MIN_X, MAX_X - x),
                    jnp.minimum(y - MIN_Y, MAX_Y - y))
    return m

# Monotone [0,1) weight: small near edge (margin≈0) → small penalty, grows with margin
def _soft_weight_from_margin(margin: jnp.ndarray, tau: float = DEATH_MARGIN_SCALE) -> jnp.ndarray:
    margin_pos = jnp.maximum(margin, 0.0)  # we only “reward closeness” for items inside
    return 1.0 - jnp.exp(-margin_pos / tau)

# ----------------------------
# Sampling utilities
# ----------------------------

def _sample_diag(key, mean, sigma, n):
    d = mean.shape[0]
    if n <= 0:
        return jnp.empty((0, d))
    eps = random.normal(key, (n, d))
    samp = mean[None, :] + eps * sigma[None, :]
    return jnp.clip(samp, 0.0, 1.0)

def _sample_full(key, mean, cov, n):
    d = mean.shape[0]
    if n <= 0:
        return jnp.empty((0, d))
    cov_np = np.asarray(cov)
    try:
        L = np.linalg.cholesky(cov_np + 1e-10 * np.eye(d))
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.maximum(np.diag(cov_np), 1e-10)))
    eps = random.normal(key, (n, d))
    samp = mean[None, :] + jnp.asarray(eps @ L.T)
    return jnp.clip(samp, 0.0, 1.0)

# ----------------------------
# CEM Solver
# ----------------------------

def solve_inverse(
    p: CurlingParams,
    prev_positions_button: jnp.ndarray,    # (N_prev,2)
    target_positions_button: jnp.ndarray,  # (N_tgt,2)  <-- may be <= N_prev+1
    bounds: SolveBounds = SolveBounds(),
    *,
    pop_size: int = 96,
    generations: int = 30,
    elite_frac: float = 0.2,
    sigma_init: float = 0.20,
    sigma_floor: float = 0.01,
    ema_alpha: float = 0.7,
    use_full_cov: bool = False,
    mix_with_best_frac: float = 0.35,
    jitter_anchor: float = 0.002,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    init_x: Optional[jnp.ndarray] = None,
    loss_threshold: Optional[float] = 0.1,   # hard-loss early stop
    log_topk: int = 3,
) -> Tuple[jnp.ndarray, float]:
    """
    Returns (x_best_phys, best_hard_loss).
    Now robust to stones disappearing off-sheet in the target.
    """
    log.info(
        "START CEM: pop_size=%d, generations=%d, elite_frac=%.2f, sigma_init=%.3f, sigma_floor=%.3f, "
        "ema_alpha=%.2f, full_cov=%s, mix_with_best_frac=%.2f, jitter_anchor=%.4f, loss_threshold=%s",
        pop_size, generations, elite_frac, sigma_init, sigma_floor, ema_alpha,
        str(use_full_cov), mix_with_best_frac, jitter_anchor,
        ("None" if loss_threshold is None else f"{loss_threshold}")
    )

    lo, hi, span = _make_bounds_arrays(bounds)
    d = int(lo.shape[0])
    assert pop_size >= 4 and 0.0 < elite_frac < 1.0

    # Initialize mean/sigma in x01
    if init_x is not None:
        mean = _x01_from_x_phys(init_x, lo, span)
    else:
        mean = jnp.ones((d,)) * 0.5
    if use_full_cov:
        cov = jnp.eye(d) * (sigma_init ** 2)
    else:
        sigma = jnp.ones((d,)) * sigma_init

    # Batched hard loss evaluator (JIT)
    batched_hard = _make_batched_hard_loss(p, prev_positions_button, target_positions_button)

    # Keep track of the globally best solution
    best_x01 = mean
    best_phys = _x_phys_from_x01(best_x01, lo, span)
    best_hard = float(batched_hard(best_phys[None, :])[0])

    log.info("INIT | mean_x01=%s | best_hard=%.6f", list(map(float, mean)), best_hard)

    elites_k = max(1, int(round(pop_size * elite_frac)))
    key_loop = key

    for gen in range(generations):
        # Split population between model mean and best anchor
        n_anchor = int(round(pop_size * mix_with_best_frac))
        n_model  = pop_size - n_anchor - 1  # +1 for exact anchor
        n_model  = max(0, n_model)
        n_anchor = max(0, n_anchor)

        key_loop, k_model, k_anchor = random.split(key_loop, 3)
        if use_full_cov:
            cand_model  = _sample_full(k_model, mean, cov, n_model)
            anchor_sigma = jnp.ones((d,)) * max(sigma_floor, jitter_anchor)
            cand_anchor = _sample_diag(k_anchor, best_x01, anchor_sigma, n_anchor)
        else:
            cand_model  = _sample_diag(k_model, mean, sigma, n_model)
            anchor_sigma = jnp.maximum(sigma * 0.5, jnp.ones((d,)) * jitter_anchor)
            cand_anchor = _sample_diag(k_anchor, best_x01, anchor_sigma, n_anchor)

        anchor_exact = best_x01[None, :]
        X01 = jnp.concatenate([cand_model, cand_anchor, anchor_exact], axis=0)  # (pop_size, d)
        X_phys = _x_phys_from_x01(X01, lo, span)

        # Evaluate hard loss
        losses = _safe_metric(batched_hard(X_phys))  # (pop_size,)
        losses_np = np.asarray(losses)
        order = np.argsort(losses_np)
        topk = order[:min(log_topk, pop_size)]

        log.info(
            "GEN %03d | best=%.6f | top%d: %s",
            gen,
            float(np.min(losses_np)),
            len(topk),
            ", ".join([f"{i}:{losses_np[i]:.6f}" for i in topk])
        )

        if float(np.min(losses_np)) < best_hard:
            i0 = int(order[0])
            best_hard = float(losses_np[i0])
            best_x01  = X01[i0]
            best_phys = X_phys[i0]
            log.info("  NEW BEST | gen=%d | hard=%.6f | x=%s", gen, best_hard, list(map(float, best_phys)))

        if (loss_threshold is not None) and (best_hard <= float(loss_threshold)):
            log.info("EARLY-STOP at gen=%d | best_hard=%.6f <= %.6f", gen, best_hard, float(loss_threshold))
            break

        # Select elites and refit distribution
        elites_idx = order[:elites_k]
        E = X01[elites_idx, :]                            # (K,d)
        elite_mean = jnp.mean(E, axis=0)

        if use_full_cov:
            E0 = E - elite_mean[None, :]
            cov_elite = (E0.T @ E0) / max(1, E.shape[0] - 1)
            cov = (1.0 - ema_alpha) * cov + ema_alpha * cov_elite
            diag = jnp.maximum(jnp.diag(cov), (sigma_floor ** 2))
            cov = cov - jnp.diag(jnp.diag(cov)) + jnp.diag(diag)
        else:
            elite_std = jnp.std(E, axis=0)
            mean = (1.0 - ema_alpha) * mean + ema_alpha * elite_mean
            sigma = (1.0 - ema_alpha) * sigma + ema_alpha * elite_std
            sigma = jnp.maximum(sigma, jnp.ones_like(sigma) * sigma_floor)

        # Gentle tug toward anchor
        mean = 0.9 * mean + 0.1 * best_x01

    log.info("FINISH CEM | best_hard=%.6f | x_best=%s", best_hard, list(map(float, best_phys)))
    return best_phys, best_hard

# ----------------------------
# Small wrapper for convenience
# ----------------------------

def solve_and_simulate(
    p: CurlingParams,
    prev_positions_button: jnp.ndarray,
    target_positions_button: jnp.ndarray,
    **kwargs
):
    x_phys, hard_loss = solve_inverse(p, prev_positions_button, target_positions_button, **kwargs)
    pred_final = simulate_from_params(p, prev_positions_button, x_phys, dynamic=False)
    return x_phys, hard_loss, pred_final
