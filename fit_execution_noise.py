#!/usr/bin/env python3
"""
Estimate execution-noise hyperparameters for Monte Carlo shot sampling.

Reads inverse-solved throws (stones_with_estimates.chunk*.csv), joins task/handle
metadata from Stones.csv, and emits a noise_config.json with per-group diagonal
stds (or full covariances) over [speed, angle, spin, y0].
"""

import argparse
import json
import math
import pathlib
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PARAM_COLS = ["est_speed", "est_angle", "est_spin", "est_y0"]
SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]


def _load_inverse(glob_pattern: str) -> pd.DataFrame:
    paths = sorted(pathlib.Path(".").glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched inverse glob: {glob_pattern}")
    frames: List[pd.DataFrame] = []
    for p in paths:
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    return df


def _attach_metadata(inv_df: pd.DataFrame, stones_csv: str) -> pd.DataFrame:
    use_cols = SHOT_KEY + ["TeamID", "PlayerID", "Task", "Handle"]
    stones_df = pd.read_csv(stones_csv, usecols=use_cols)
    merged = pd.merge(inv_df, stones_df, on=SHOT_KEY, how="left")
    return merged


def _robust_std(arr: np.ndarray, use_mad: bool, min_std: float) -> np.ndarray:
    if arr.size == 0:
        return np.full((arr.shape[1],), min_std, dtype=np.float64)
    if use_mad:
        med = np.nanmedian(arr, axis=0)
        mad = np.nanmedian(np.abs(arr - med), axis=0)
        std = mad * 1.4826  # MAD -> sigma for Gaussian
    else:
        std = np.nanstd(arr, axis=0, ddof=1)
    std = np.where(np.isfinite(std), std, min_std)
    return np.maximum(std, min_std)


def _group_label(task: float, handle: float) -> str:
    t = "nan" if pd.isna(task) else int(task)
    h = "nan" if pd.isna(handle) else int(handle)
    return f"task_{t}_handle_{h}"


def _player_task_label(player_id, task) -> str:
    p = "nan" if pd.isna(player_id) else int(player_id)
    t = "nan" if pd.isna(task) else int(task)
    return f"player_{p}_task_{t}"


def _compute_group_stats(
    df: pd.DataFrame,
    group_keys: Iterable[str],
    use_mad: bool,
    min_std: float,
    scale: float,
    full_cov: bool,
    min_count: int,
    label_func=None,
) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    grouped = df.groupby(list(group_keys), dropna=False)
    for key, g in grouped:
        params = g[PARAM_COLS].dropna()
        n = len(params)
        if n < min_count:
            continue
        arr = params.to_numpy(dtype=np.float64)
        mean = np.nanmean(arr, axis=0)
        std = _robust_std(arr, use_mad=use_mad, min_std=min_std) * scale
        entry = {"count": int(n), "mean": mean.tolist(), "std": std.tolist()}
        if full_cov:
            cov = np.cov(arr.T) if n > 1 else np.diag(std ** 2)
            entry["cov"] = cov.tolist()
        if label_func is not None:
            label = label_func(*key) if isinstance(key, tuple) else label_func(key)
        else:
            label = key if isinstance(key, str) else "_".join(
                str(k) for k in (key if isinstance(key, tuple) else (key,))
            )
        out[label] = entry
    return out


def build_noise_config(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict:
    # Global stats across everything
    params = df[PARAM_COLS].dropna()
    global_mean = params.mean().to_numpy(dtype=np.float64) if len(params) else np.zeros((4,), dtype=np.float64)
    global_std = _robust_std(params.to_numpy(dtype=np.float64), use_mad=args.use_mad, min_std=args.min_std) * args.scale
    cfg: Dict[str, Dict] = {
        "default": {
            "mean": global_mean.tolist(),
            "std": global_std.tolist(),
        }
    }
    if args.full_cov:
        cov = np.cov(params.to_numpy(dtype=np.float64).T) if len(params) > 1 else np.diag(global_std ** 2)
        cfg["default"]["cov"] = cov.tolist()

    # Per task+handle
    task_handle_stats = _compute_group_stats(
        df,
        group_keys=["Task", "Handle"],
        use_mad=args.use_mad,
        min_std=args.min_std,
        scale=args.scale,
        full_cov=args.full_cov,
        min_count=args.min_count,
        label_func=_group_label,
    )
    cfg["by_task_handle"] = task_handle_stats

    # Optional per player x task (light shrinkage option)
    if args.by_player_task:
        player_task_stats = _compute_group_stats(
            df,
            group_keys=["PlayerID", "Task"],
            use_mad=args.use_mad,
            min_std=args.min_std,
            scale=args.scale,
            full_cov=args.full_cov,
            min_count=args.min_count_player,
            label_func=_player_task_label,
        )
        cfg["by_player_task"] = player_task_stats

    cfg["meta"] = {
        "use_mad": args.use_mad,
        "min_std": args.min_std,
        "scale": args.scale,
        "full_cov": args.full_cov,
        "min_count": args.min_count,
        "min_count_player": args.min_count_player,
        "filters": {
            "require_solver_ok": args.require_solver_ok,
            "hard_loss_max": args.hard_loss_max,
        },
    }
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Calibrate execution-noise distribution from inverse solutions.")
    ap.add_argument("--inverse-glob", type=str, default="inverseDataset/stones_with_estimates.chunk*.csv", help="Glob for inverse CSVs")
    ap.add_argument("--stones-csv", type=str, default="2026/Stones.csv", help="Raw Stones.csv for Task/Handle/Player metadata")
    ap.add_argument("--out", type=str, default="noise_config.json", help="Output JSON path")
    ap.add_argument("--hard-loss-max", type=float, default=0.25, help="Keep rows with hard_loss_refine <= this (nan passes).")
    ap.add_argument("--require-solver-ok", action="store_true", help="Drop rows where solver_ok is False.")
    ap.add_argument("--use-mad", action="store_true", help="Use MAD-based std estimate (robust).")
    ap.add_argument("--min-std", type=float, default=0.01, help="Floor for std per dimension.")
    ap.add_argument("--scale", type=float, default=1.0, help="Scale factor applied to stds.")
    ap.add_argument("--full-cov", action="store_true", help="Emit covariance matrices instead of diagonal-only.")
    ap.add_argument("--min-count", type=int, default=20, help="Minimum samples for a task/handle group.")
    ap.add_argument("--by-player-task", action="store_true", help="Also compute per-player-per-task stats (needs enough samples).")
    ap.add_argument("--min-count-player", type=int, default=30, help="Min samples for player-task groups.")
    args = ap.parse_args()

    inv_df = _load_inverse(args.inverse_glob)
    inv_df = _attach_metadata(inv_df, args.stones_csv)

    df = inv_df.copy()
    if args.require_solver_ok and "solver_ok" in df.columns:
        df = df[df["solver_ok"] == True]  # noqa: E712
    if args.hard_loss_max is not None and "hard_loss_refine" in df.columns:
        df = df[df["hard_loss_refine"] <= args.hard_loss_max]

    df_params = df[SHOT_KEY + ["Task", "Handle", "PlayerID"] + PARAM_COLS].dropna(subset=PARAM_COLS, how="any")
    if df_params.empty:
        raise ValueError("No parameter rows left after filtering; cannot fit noise.")

    cfg = build_noise_config(df_params, args)
    out_path = pathlib.Path(args.out)
    out_path.write_text(json.dumps(cfg, indent=2))
    print(f"[done] wrote noise config with {len(cfg.get('by_task_handle', {}))} task/handle entries to {out_path}")


if __name__ == "__main__":
    main()
