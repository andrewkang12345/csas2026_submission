# demo_infer.py
# Creates a scenario where at least one stone goes off-sheet ("dead").
# Only IN-BOUNDS stones from the fabricated next state are used as targets
# in the inverse solve (consistent with curling_inverse.py rectangular loss).

import logging
from logging.handlers import RotatingFileHandler
from dataclasses import replace as dataclass_replace
import numpy as np
import jax
import jax.numpy as jnp

from curling_sim_jax import (
    CurlingParams,
    simulate_from_params,
)
from curling_inverse import (
    solve_and_simulate,
    SolveBounds,
    MIN_X, MAX_X, MIN_Y, MAX_Y,  # use identical bounds for trimming/annotation
)
from preprocess import separate_overlaps  # optional sanitizer

# Optional GIF
try:
    from viz_sim import render_gif
    HAVE_VIZ = True
except Exception:
    HAVE_VIZ = False

def setup_logger():
    logger = logging.getLogger("curling")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = RotatingFileHandler("run.log", maxBytes=4_000_000, backupCount=3)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

def _in_bounds_mask_np(pos_xy: np.ndarray) -> np.ndarray:
    x = pos_xy[:, 0]
    y = pos_xy[:, 1]
    return (x > MIN_X) & (x < MAX_X) & (y > MIN_Y) & (y < MAX_Y)

def _fabricate_target_with_deaths(p: CurlingParams,
                                  prev: jnp.ndarray,
                                  log: logging.Logger) -> tuple[jnp.ndarray, np.ndarray, np.ndarray]:
    """
    Try a few 'true' throws; return (true_x, target_final_inbounds, target_final_full)
    ensuring at least one stone ends off-sheet if possible.
    """
    # Aggressive throws first (constant decel model: higher speed goes much farther).
    candidates = [
        jnp.array([3.00,  0.00, 0.00, 0.00], dtype=jnp.float32),  # straight, very fast
        jnp.array([2.80,  0.10, 0.00, 0.00], dtype=jnp.float32),  # slight right
        jnp.array([2.80, -0.10, 0.00, 0.00], dtype=jnp.float32),  # slight left
        jnp.array([2.50,  0.00, 0.00, 0.00], dtype=jnp.float32),
        jnp.array([2.30,  0.20, 0.00, 0.00], dtype=jnp.float32),
    ]

    for true_x in candidates:
        traj = simulate_from_params(p, prev, true_x, dynamic=True)  # (T, N, 2)
        final_full = np.asarray(traj)[-1]  # (N,2) includes off-sheet stones
        inb_mask = _in_bounds_mask_np(final_full)
        target_inb = final_full[inb_mask]
        num_off = int(final_full.shape[0] - target_inb.shape[0])

        log.info("fabricate | true=%s | in_bounds=%d | off_sheet=%d",
                 list(map(float, true_x)), int(target_inb.shape[0]), num_off)

        # We want a scenario with at least one dead stone
        if num_off >= 1:
            return true_x, target_inb, final_full

    # Fallback: if none went off (unlikely with the above), just use the first
    true_x = candidates[0]
    traj = simulate_from_params(p, prev, true_x, dynamic=True)
    final_full = np.asarray(traj)[-1]
    inb_mask = _in_bounds_mask_np(final_full)
    target_inb = final_full[inb_mask]
    log.warning("fabricate | fallback used; off-sheet may be 0 (in_bounds=%d, total=%d)",
                int(target_inb.shape[0]), int(final_full.shape[0]))
    return true_x, target_inb, final_full

def main():
    log = setup_logger()
    log.info("=== demo_infer (with deaths) start ===")

    # Physics params (same as your refine stage)
    p_refine = CurlingParams(dt=0.02, substeps=2, k_penalty=2.5e4, c_damp=220.0, k_curl=0.10)

    # ========= Pre-placed stones (edit freely) =========
    prev_list = [
        [ 0.20,  0.10],
        [ 0.05, -0.25],
        [ 0.15, 0.25],
        [-0.30,  0.05],
        [ 0.30,  0.05],
    ]
    prev_np = np.array(prev_list, dtype=np.float32) if len(prev_list) > 0 else np.zeros((0,2), dtype=np.float32)
    # ===================================================

    # --- Sanitize overlaps (optional but nice for robustness) ---
    if prev_np.shape[0] > 1:
        clean_prev, stats = separate_overlaps(prev_np, radius=p_refine.radius, min_clearance=1e-3)
        if stats["moved"]:
            log.info(
                "SANITIZE: overlaps resolved | iters=%d | max_pen=%.6f | total_disp=%.6f",
                stats["iters"], stats["max_penetration"], stats["total_displacement"]
            )
        else:
            log.info("SANITIZE: no overlaps detected")
        prev = jnp.array(clean_prev, dtype=jnp.float32)
    else:
        prev = jnp.array(prev_np, dtype=jnp.float32)

    # ----- Fabricate a target that includes at least one off-sheet stone -----
    true_x, target_final_inbounds, target_final_full = _fabricate_target_with_deaths(p_refine, prev, log)

    # For visibility in logs:
    num_total = int(target_final_full.shape[0])
    num_inb   = int(target_final_inbounds.shape[0])
    num_off   = num_total - num_inb
    log.info("target fabricated | total_stones=%d | in_bounds=%d | off_sheet=%d",
             num_total, num_inb, num_off)

    # ----- Stage A: coarse CEM (broader exploration) -----
    p_coarse = dataclass_replace(p_refine, dt=0.03, substeps=1, max_steps=900, k_penalty=2.0e4)

    x0, hard0, _ = solve_and_simulate(
        p_coarse,
        prev,
        jnp.asarray(target_final_inbounds, dtype=jnp.float32),
        bounds=SolveBounds(),
        pop_size=800,
        generations=25,
        elite_frac=0.20,
        sigma_init=0.25,
        sigma_floor=0.01,
        ema_alpha=0.7,
        use_full_cov=False,
        mix_with_best_frac=0.35,
        jitter_anchor=0.006,
        key=jax.random.PRNGKey(0),
        init_x=None,
        loss_threshold=0.5,
        log_topk=3,
    )
    log.info("[coarse] x0=%s | hard_loss=%.6f", list(map(float, x0)), hard0)

    # ----- Stage B: refine CEM (narrower) -----
    x, hard, pred_final = solve_and_simulate(
        p_refine,
        prev,
        jnp.asarray(target_final_inbounds, dtype=jnp.float32),
        bounds=SolveBounds(),
        pop_size=400,
        generations=80,
        elite_frac=0.30,
        sigma_init=0.10,
        sigma_floor=0.005,
        ema_alpha=0.75,
        use_full_cov=False,
        mix_with_best_frac=0.40,
        jitter_anchor=0.0015,
        key=jax.random.PRNGKey(1),
        init_x=x0,
        loss_threshold=0.10,
        log_topk=3,
    )

    log.info("[refine] x=%s | true=%s | hard_loss=%.6f",
             list(map(float, x)), list(map(float, true_x)), hard)

    # (Optional) quick rectangular-loss recompute for sanity (pure numpy, for logging)
    # Uses a simple greedy that mirrors curling_inverse's rectangular loss shape.
    def _rect_cost_np(pred_xy: np.ndarray, tgt_xy: np.ndarray) -> float:
        if tgt_xy.shape[0] == 0 and pred_xy.shape[0] == 0:
            return 0.0
        # Only predicted IN-BOUNDS can be matched against targets
        inb_mask = _in_bounds_mask_np(pred_xy)
        pred_inb = pred_xy[inb_mask]
        Np, Nt = pred_inb.shape[0], tgt_xy.shape[0]
        if Nt == 0:
            # penalize any in-bounds predictions that remain
            return float(4.0 * Np)  # matches EXTRA_PRED_COST default
        C = ((pred_inb[:, None, :] - tgt_xy[None, :, :]) ** 2).sum(axis=2)
        used_r, used_c = set(), set()
        total = 0.0
        for _ in range(min(Np, Nt)):
            i, j = divmod(C.argmin(), C.shape[1])
            total += float(C[i, j])
            used_r.add(i); used_c.add(j)
            C[i, :] = 1e9
            C[:, j] = 1e9
        miss_t = Nt - len(used_c)
        miss_p = Np - len(used_r)
        return total + 4.0 * miss_t + 4.0 * miss_p

    pred_final_np = np.asarray(pred_final)
    loss_check = _rect_cost_np(pred_final_np, target_final_inbounds)
    log.info("[refine] rectangular_loss_recompute=%.6f", loss_check)

    # Optional GIF render: target = IN-BOUNDS ONLY
    if HAVE_VIZ:
        gif = render_gif(
            p_refine,
            prev_positions_button=prev,
            x_params=x,
            target_positions_button=jnp.asarray(target_final_inbounds, dtype=jnp.float32),
            out_path="throw_with_deaths.gif",
            stride=5,
            dpi=120,
            annotate=True,
            draw_house=True,
            show_assignment=True,
        )
        log.info("GIF saved to: %s", gif)

    log.info("=== demo_infer end ===")

if __name__ == "__main__":
    main()