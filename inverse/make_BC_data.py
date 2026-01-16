# make_bc_dataset_parallel.py
# Parallel dataset builder for Stones.csv
#
# - Spawns one worker per visible GPU.
# - Processes the dataset in CHUNKS of 5000 shots (configurable).
# - Each worker writes periodic per-GPU CSV parts (default every 500 rows).
# - After each 5000-shot chunk completes, the parent merges parts into
#   a single chunk CSV: <out_prefix>.chunkXXXX.csv.
#
# Usage:
#   python make_bc_dataset_parallel.py \
#       --csv /path/to/Stones.csv \
#       --out-prefix stones_with_estimates \
#       [--chunk-size 5000] [--flush-every 500] [--limit N] [--seed 0]
#
# Requirements (same repo):
#   - curling_sim_jax.py
#   - curling_inverse.py
#
# Notes:
# - We intentionally import JAX and curling_* inside each worker process
#   AFTER pinning CUDA_VISIBLE_DEVICES, to ensure one GPU per worker.
# - We avoid cross-process writes to a single file; workers emit parts
#   and the parent merges per chunk.

import os
import sys
import math
import glob
import shutil
import random
import argparse
import tempfile
from dataclasses import dataclass, replace as dataclass_replace
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

# ---------------- CSV / unit conversion constants ----------------
SENTINEL_OFF = 4095
CSV_STONE_COUNT = 12

# CSV → meters for your "button frame"
CSV_BUTTON_Y = 800.0
CSV_CENTER_X = 750.0
CSV_TO_M = 0.003048  # (y - 800)*0.003048 → along-sheet meters; (x - 750)*0.003048 → lateral meters

# Small separation for overlapped previous stones (sanitization)
STONE_RADIUS_M = 0.145
MIN_CLEAR = 2 * STONE_RADIUS_M + 1e-3
SEPARATE_PASSES = 6
# -----------------------------------------------------------------


@dataclass(frozen=True)
class ShotKey:
    comp: int
    sess: int
    game: int
    end: int
    shot: int  # ShotID for the "after" row (the one we estimate for)


# --------- CSV helpers (no JAX imports here) ---------
def _csv_y_to_xm(y_csv: float) -> float:
    # Along-sheet, 0 at button
    return (y_csv - CSV_BUTTON_Y) * CSV_TO_M


def _csv_x_to_ym(x_csv: float) -> float:
    # Lateral, 0 at centerline
    return (x_csv - CSV_CENTER_X) * CSV_TO_M


def _valid_xy(x_csv: float, y_csv: float) -> bool:
    # 0 => not yet thrown, 4095 => off sheet (dead)
    if x_csv in (0, SENTINEL_OFF) or y_csv in (0, SENTINEL_OFF):
        return False
    return True


def _get_xy_from_row(row: pd.Series, i: int) -> Tuple[Optional[float], Optional[float]]:
    # Case-robust stone_i_x/y lookup
    for (kx, ky) in ((f"Stone_{i}_x", f"Stone_{i}_y"), (f"stone_{i}_x", f"stone_{i}_y")):
        if kx in row and ky in row:
            xv, yv = row[kx], row[ky]
            return (None if pd.isna(xv) else float(xv),
                    None if pd.isna(yv) else float(yv))
    return (None, None)


def _row_to_state_xy(row: pd.Series) -> Dict[int, Tuple[float, float]]:
    """Return dict: stone_index -> (x_m, y_m) for stones that are in play on this row."""
    out: Dict[int, Tuple[float, float]] = {}
    for i in range(1, CSV_STONE_COUNT + 1):
        xi, yi = _get_xy_from_row(row, i)
        if xi is None or yi is None:
            continue
        if not _valid_xy(xi, yi):
            continue
        xm = _csv_y_to_xm(yi)
        ym = _csv_x_to_ym(xi)
        out[i] = (xm, ym)
    return out


def _separate_overlaps(pts: np.ndarray, min_gap: float = MIN_CLEAR, passes: int = SEPARATE_PASSES) -> np.ndarray:
    """Deterministic relaxation: push apart any overlapping stones (prev state only)."""
    if pts.size == 0:
        return pts
    p = pts.copy()
    n = p.shape[0]
    for _ in range(passes):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = p[j, 0] - p[i, 0]
                dy = p[j, 1] - p[i, 1]
                d = math.hypot(dx, dy)
                if d < 1e-9:
                    dx, dy, d = 1e-6, 0.0, 1e-6
                if d < min_gap:
                    push = 0.5 * (min_gap - d)
                    nx, ny = dx / d, dy / d
                    p[i, 0] -= push * nx
                    p[i, 1] -= push * ny
                    p[j, 0] += push * nx
                    p[j, 1] += push * ny
                    moved = True
        if not moved:
            break
    return p


def _iter_shots_with_prev(df: pd.DataFrame):
    """Yield ShotKey for every row that has a previous row in the same (comp,sess,game,end)."""
    group_cols = ["CompetitionID", "SessionID", "GameID", "EndID"]
    for (comp, sess, game, end), df_end in df.groupby(group_cols, sort=False):
        df_end = df_end.sort_values("ShotID", ascending=True)
        shots = df_end["ShotID"].tolist()
        for idx in range(1, len(shots)):
            yield ShotKey(int(comp), int(sess), int(game), int(end), int(shots[idx]))


def _count_shots_with_prev(df: pd.DataFrame) -> int:
    group_cols = ["CompetitionID", "SessionID", "GameID", "EndID"]
    total = 0
    for _, df_end in df.groupby(group_cols, sort=False):
        df_end = df_end.sort_values("ShotID", ascending=True)
        n = df_end.shape[0]
        if n >= 2:
            total += (n - 1)
    return total


def _load_prev_next_states(df: pd.DataFrame, key: ShotKey):
    """Return dicts keyed by stone_id for prev and next (already converted to meters)."""
    df_end = df[
        (df["CompetitionID"] == key.comp) &
        (df["SessionID"] == key.sess) &
        (df["GameID"] == key.game) &
        (df["EndID"] == key.end)
    ].sort_values("ShotID", ascending=True)
    rows = df_end.to_dict("records")
    shots = df_end["ShotID"].tolist()
    idx = shots.index(key.shot)

    prev_state_m = _row_to_state_xy(rows[idx - 1])
    next_state_m = _row_to_state_xy(rows[idx])
    return prev_state_m, next_state_m


def _fill_flat_arrays(state_m: Dict[int, Tuple[float, float]]):
    """Return flat arrays shaped (12,2) with NaN where absent."""
    arr = np.full((CSV_STONE_COUNT, 2), np.nan, dtype=np.float32)
    for k, (xm, ym) in state_m.items():
        if 1 <= k <= CSV_STONE_COUNT:
            arr[k - 1, 0] = xm
            arr[k - 1, 1] = ym
    return arr


# -------------------- Worker logic --------------------
def _worker_run(
    gpu_local_index: int,
    keys: List[ShotKey],
    df_end_index: pd.DataFrame,  # pre-filtered to a chunk; we pass a whole df for simplicity
    out_dir: str,
    chunk_idx: int,
    base_seed: int,
    flush_every: int,
    verbose: bool = False,
):
    """
    Single-GPU worker. Pins to one GPU and processes its assigned keys,
    writing periodic part CSVs to out_dir.
    """
    # Pin this process to a single GPU before importing jax/curling libs.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_local_index)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")  # safer fragmentation behavior

    # Import JAX + sim libs AFTER pinning.
    import jax
    import jax.numpy as jnp
    from curling_sim_jax import CurlingParams
    from curling_inverse import (
        solve_and_simulate,
        SolveBounds,
        MIN_X, MAX_X, MIN_Y, MAX_Y,
    )

    def _in_bounds_mask_np(pos_xy: np.ndarray) -> np.ndarray:
        if pos_xy.size == 0:
            return np.zeros((0,), dtype=bool)
        x = pos_xy[:, 0]
        y = pos_xy[:, 1]
        return (x > MIN_X) & (x < MAX_X) & (y > MIN_Y) & (y < MAX_Y)

    def run_two_stage_cem(prev_pts: np.ndarray,
                          next_pts_full: np.ndarray,
                          seed: int = 0):
        """Run the same coarse + refine CEM as demo_infer.py. Returns (x_best, hard, pred_final, next_inb)."""
        # Physics (refine) as in demo_infer.py
        p_refine = CurlingParams(dt=0.02, substeps=2, k_penalty=2.5e4, c_damp=220.0, k_curl=0.10)
        p_coarse = dataclass_replace(p_refine, dt=0.03, substeps=1, max_steps=900, k_penalty=2.0e4)

        # Trim NEXT to in-bounds only (targets)
        inb_mask = _in_bounds_mask_np(next_pts_full)
        next_pts_inb = next_pts_full[inb_mask]

        prev_j = jnp.asarray(prev_pts, dtype=jnp.float32)
        target_inb_j = jnp.asarray(next_pts_inb, dtype=jnp.float32)

        # Stage A (coarse)
        x0, hard0, _ = solve_and_simulate(
            p_coarse,
            prev_j,
            target_inb_j,
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
            key=jax.random.PRNGKey(seed),
            init_x=None,
            loss_threshold=0.5,
            log_topk=3,
        )

        # Stage B (refine)
        x, hard, pred_final = solve_and_simulate(
            p_refine,
            prev_j,
            target_inb_j,
            bounds=SolveBounds(),
            pop_size=800,
            generations=80,
            elite_frac=0.30,
            sigma_init=0.10,
            sigma_floor=0.005,
            ema_alpha=0.75,
            use_full_cov=False,
            mix_with_best_frac=0.40,
            jitter_anchor=0.0015,
            key=jax.random.PRNGKey(seed + 1),
            init_x=x0,
            loss_threshold=0.10,
            log_topk=3,
        )

        return x, float(hard), float(hard0), next_pts_inb

    part_idx = 0
    buffer_rows: List[dict] = []

    for local_i, key in enumerate(keys):
        # Build prev/next states in meters
        prev_state_m, next_state_m = _load_prev_next_states(df_end_index, key)

        prev_pts = _fill_flat_arrays(prev_state_m)  # (12,2) NaN-padded
        next_pts = _fill_flat_arrays(next_state_m)

        # Drop NaNs into compact arrays for solver
        prev_compact = prev_pts[~np.isnan(prev_pts).any(axis=1)]
        next_compact = next_pts[~np.isnan(next_pts).any(axis=1)]

        # Optional sanitize prev to avoid starting overlaps (solver stability)
        if prev_compact.shape[0] > 1:
            prev_compact = _separate_overlaps(prev_compact, min_gap=MIN_CLEAR)

        # Run two-stage inverse
        try:
            seed_val = base_seed + (key.comp ^ key.sess ^ key.game ^ key.end ^ key.shot) * 2 + gpu_local_index
            x_best, hard_refine, hard_coarse, next_inb = run_two_stage_cem(
                prev_compact, next_compact, seed=seed_val
            )
            solver_ok = True
            err_msg = ""
        except Exception as e:
            # On failure, emit NaNs but keep row
            import numpy as _np
            x_best = np.array([_np.nan, _np.nan, _np.nan, _np.nan], dtype=np.float32)
            hard_refine = np.nan
            hard_coarse = np.nan
            next_inb = np.empty((0, 2), dtype=np.float32)
            solver_ok = False
            err_msg = repr(e)

        prev_N = int(prev_compact.shape[0])
        next_total_N = int(next_compact.shape[0])
        next_inb_N = int(next_inb.shape[0])

        row = {
            "CompetitionID": key.comp,
            "SessionID": key.sess,
            "GameID": key.game,
            "EndID": key.end,
            "ShotID": key.shot,
            "prev_N": prev_N,
            "next_total_N": next_total_N,
            "next_in_bounds_N": next_inb_N,
            "est_speed": float(x_best[0]),
            "est_angle": float(x_best[1]),
            "est_spin": float(x_best[2]),
            "est_y0": float(x_best[3]),
            "hard_loss_coarse": float(hard_coarse),
            "hard_loss_refine": float(hard_refine),
            "solver_ok": solver_ok,
            "solver_error": err_msg,
        }

        # Add per-stone converted coordinates (meters) for PREV and NEXT rows
        for i in range(1, CSV_STONE_COUNT + 1):
            # prev
            px, py = (np.nan, np.nan)
            if i in prev_state_m:
                px, py = prev_state_m[i]
            row[f"prev_stone_{i}_x_m"] = float(px) if not np.isnan(px) else np.nan
            row[f"prev_stone_{i}_y_m"] = float(py) if not np.isnan(py) else np.nan

            # next
            nx, ny = (np.nan, np.nan)
            if i in next_state_m:
                nx, ny = next_state_m[i]
            row[f"next_stone_{i}_x_m"] = float(nx) if not np.isnan(nx) else np.nan
            row[f"next_stone_{i}_y_m"] = float(ny) if not np.isnan(ny) else np.nan

            # in-bounds for NEXT (consistent with inverse target selection)
            if not (np.isnan(nx) or np.isnan(ny)):
                inb = (MIN_X < nx < MAX_X) and (MIN_Y < ny < MAX_Y)
            else:
                inb = False
            row[f"next_stone_{i}_inbounds"] = int(inb)

        buffer_rows.append(row)

        # Periodic flush to a part CSV
        if (local_i + 1) % flush_every == 0:
            part_path = os.path.join(
                out_dir,
                f"chunk{chunk_idx:04d}.gpu{gpu_local_index}.part{part_idx:03d}.csv",
            )
            pd.DataFrame(buffer_rows).to_csv(part_path, index=False)
            buffer_rows.clear()
            part_idx += 1
            if verbose:
                print(f"[GPU {gpu_local_index}] wrote {part_path}", flush=True)

    # Final flush for remaining rows
    if buffer_rows:
        part_path = os.path.join(
            out_dir,
            f"chunk{chunk_idx:04d}.gpu{gpu_local_index}.part{part_idx:03d}.csv",
        )
        pd.DataFrame(buffer_rows).to_csv(part_path, index=False)
        if verbose:
            print(f"[GPU {gpu_local_index}] wrote {part_path}", flush=True)


# -------------------- Orchestration --------------------
def _gather_keys(df: pd.DataFrame, limit: Optional[int]) -> List[ShotKey]:
    keys = list(_iter_shots_with_prev(df))
    if limit is not None:
        keys = keys[:limit]
    return keys


def _split_into_chunks(keys: List[ShotKey], chunk_size: int) -> List[List[ShotKey]]:
    return [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]


def _round_robin_slices(keys: List[ShotKey], n: int) -> List[List[ShotKey]]:
    """Split keys into n disjoint slices (round-robin) to balance load."""
    buckets = [[] for _ in range(n)]
    for idx, k in enumerate(keys):
        buckets[idx % n].append(k)
    return buckets


def main(
    csv_path: str,
    out_prefix: str,
    base_seed: Optional[int],
    limit: Optional[int],
    chunk_size: int,
    flush_every: int,
    verbose: bool,
):
    # Read once in parent
    df = pd.read_csv(csv_path)

    total_possible = _count_shots_with_prev(df)
    if limit is not None:
        total_possible = min(total_possible, limit)

    print(f"[info] eligible shots with previous row: {total_possible}")

    keys_all = _gather_keys(df, limit)
    if not keys_all:
        print("[info] nothing to do")
        return

    chunks = _split_into_chunks(keys_all, chunk_size)
    num_chunks = len(chunks)

    # Discover visible GPUs
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None:
        # Respect any user-specified mask
        gpu_ids = [g for g in visible.split(",") if g.strip() != ""]
        num_gpus = len(gpu_ids)
    else:
        # Probe with JAX if available; otherwise fallback to 1
        try:
            import jax  # lightweight import in parent to count devices
            num_gpus = len([d for d in jax.devices() if d.platform == "gpu"])
        except Exception:
            num_gpus = 1

    if num_gpus < 1:
        print("[warn] No GPUs detected by JAX — running single worker on CPU.")
        num_gpus = 1

    print(f"[info] using {num_gpus} worker(s) (1 per GPU)")

    # Work directory for parts
    out_dir = os.path.abspath(f"{out_prefix}.parts")
    os.makedirs(out_dir, exist_ok=True)

    # Process each CHUNK of 5000
    for chunk_idx, keys_chunk in enumerate(chunks):
        print(f"\n[chunk {chunk_idx+1}/{num_chunks}] shots: {len(keys_chunk)}")

        # Round-robin across GPUs
        per_gpu_keys = _round_robin_slices(keys_chunk, num_gpus)

        # Spawn processes (spawn method ensures imports happen after pinning)
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        procs = []

        # For worker convenience (filter to only rows used in this chunk’s comps)
        # Using full df is fine; filtering isn't necessary because _load_prev_next_states
        # already queries by indices. We pass df to each worker independently (copy).
        df_for_workers = df

        for gpu_local_index in range(num_gpus):
            if not per_gpu_keys[gpu_local_index]:
                continue
            p = ctx.Process(
                target=_worker_run,
                args=(
                    gpu_local_index,
                    per_gpu_keys[gpu_local_index],
                    df_for_workers,
                    out_dir,
                    chunk_idx,
                    0 if base_seed is None else int(base_seed),
                    flush_every,
                    verbose,
                ),
                daemon=False,
            )
            p.start()
            procs.append(p)

        # Wait for workers
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker process exited with code {p.exitcode}")

        # Merge part CSVs for this chunk
        part_glob = os.path.join(out_dir, f"chunk{chunk_idx:04d}.gpu*.part*.csv")
        part_files = sorted(glob.glob(part_glob))
        if not part_files:
            print(f"[warn] no parts found for chunk {chunk_idx}")
            continue

        dfs = []
        total_rows = 0
        for pf in part_files:
            dfi = pd.read_csv(pf)
            total_rows += len(dfi)
            dfs.append(dfi)

        merged = pd.concat(dfs, ignore_index=True)
        out_chunk_path = f"{out_prefix}.chunk{chunk_idx:04d}.csv"
        merged.to_csv(out_chunk_path, index=False)
        print(f"[done] merged {len(part_files)} parts => {out_chunk_path} ({len(merged)} rows)")

        # Optional: clean up part files for this chunk to save space
        for pf in part_files:
            try:
                os.remove(pf)
            except Exception:
                pass

    print("\n[all done]")
    print(f"Per-chunk CSVs written with prefix: {out_prefix}.chunkXXXX.csv")
    print(f"Temp parts directory retained at: {out_dir} (safe to delete)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="/mnt/data/curling2/brax/2026/Stones.csv", help="Path to Stones.csv")
    ap.add_argument("--out-prefix", type=str, default="stones_with_estimates", help="Output CSV prefix (chunks will be appended)")
    ap.add_argument("--seed", type=int, default=None, help="Base RNG seed")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N eligible shots")
    ap.add_argument("--chunk-size", type=int, default=5000, help="Shots per chunk")
    ap.add_argument("--flush-every", type=int, default=500, help="Rows per per-GPU partial CSV write")
    ap.add_argument("--verbose", action="store_true", help="Extra logging from workers")
    args = ap.parse_args()

    main(
        csv_path=args.csv,
        out_prefix=args.out_prefix,
        base_seed=args.seed,
        limit=args.limit,
        chunk_size=int(args.chunk_size),
        flush_every=int(args.flush_every),
        verbose=bool(args.verbose),
    )
