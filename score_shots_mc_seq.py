#!/usr/bin/env python3
"""
score_shots_mc_by_competition.py

Run score_shots_mc "one competition at a time":
- Loads all inverse chunks + Stones.csv once
- Merges once
- Then iterates over CompetitionID in order:
    * runs smoke (optional) for that competition only
    * runs full scoring for that competition only
    * writes per-competition output CSV (and optionally a combined CSV)

Notes:
- This keeps memory reasonable and makes it easy to resume/parallelize at the competition level.
- If you want to run competitions in parallel, you can launch multiple processes with --only-competition / --competition-ids.

It is based on your existing script, with minimal behavior changes.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR / "inverse"))
sys.path.append(str(THIS_DIR / "valueModel"))

# ----------------------------
# Optional JAX simulation imports
# ----------------------------
CURLING_IMPORT_ERROR = None
try:
    import jax
    import jax.numpy as jnp
    from curling_sim_jax import CurlingParams, simulate_from_params  # type: ignore
    from curling_inverse import MIN_X, MAX_X, MIN_Y, MAX_Y, SolveBounds  # type: ignore
except Exception as e:  # noqa: BLE001
    CURLING_IMPORT_ERROR = e
    jax = None  # type: ignore
    jnp = None  # type: ignore
    CurlingParams = None  # type: ignore
    simulate_from_params = None  # type: ignore
    SolveBounds = None  # type: ignore
    MIN_X = MIN_Y = -1e9
    MAX_X = MAX_Y = 1e9

import torch

try:
    import xgboost as xgb
except Exception:
    xgb = None  # noqa: N816

from dataset import POS_MAX  # type: ignore
from model import ValueTransformer  # type: ignore

CSV_BUTTON_Y = 800.0
CSV_CENTER_X = 750.0
CSV_TO_M = 0.003048

SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
PARAM_COLS = ["est_speed", "est_angle", "est_spin", "est_y0"]


# ----------------------------
# Coordinate helpers
# ----------------------------
def meters_to_raw_xy(x_m: float, y_m: float) -> Tuple[float, float]:
    raw_x = y_m / CSV_TO_M + CSV_CENTER_X
    raw_y = x_m / CSV_TO_M + CSV_BUTTON_Y
    return float(raw_x), float(raw_y)


def positions_m_to_raw_matrix(pos_m: np.ndarray) -> np.ndarray:
    out = np.full_like(pos_m, POS_MAX, dtype=np.float32)
    for i, (xm, ym) in enumerate(pos_m):
        if np.isnan(xm) or np.isnan(ym):
            continue
        inb = (MIN_X < float(xm) < MAX_X) and (MIN_Y < float(ym) < MAX_Y)
        if not inb:
            continue
        rx, ry = meters_to_raw_xy(float(xm), float(ym))
        out[i, 0] = rx
        out[i, 1] = ry
    return out


def normalize_raw_matrix(pos_raw: np.ndarray) -> np.ndarray:
    arr = np.where(np.isfinite(pos_raw), pos_raw, POS_MAX).astype(np.float32)
    return (arr / POS_MAX).reshape(-1).astype(np.float32)


def extract_state_from_row(row: pd.Series, prefix: str) -> Tuple[np.ndarray, List[int]]:
    mat = np.full((12, 2), np.nan, dtype=np.float32)
    for i in range(1, 13):
        x = row.get(f"{prefix}_stone_{i}_x_m", np.nan)
        y = row.get(f"{prefix}_stone_{i}_y_m", np.nan)
        if not (pd.isna(x) or pd.isna(y)):
            mat[i - 1, 0] = float(x)
            mat[i - 1, 1] = float(y)
    keep_mask = ~np.isnan(mat).any(axis=1)
    stone_ids = [idx + 1 for idx, keep in enumerate(keep_mask) if keep]
    return mat, stone_ids


def compact_positions(mat: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    keep_mask = ~np.isnan(mat).any(axis=1)
    compact = mat[keep_mask]
    ids = [i + 1 for i, flag in enumerate(keep_mask) if flag]
    return compact.astype(np.float32), ids


def assign_final_to_slots(final_pos: np.ndarray, prev_ids: List[int], new_id: int) -> np.ndarray:
    out = np.full((12, 2), np.nan, dtype=np.float32)
    for idx, sid in enumerate(prev_ids):
        if idx < final_pos.shape[0]:
            out[sid - 1] = final_pos[idx]
    if final_pos.shape[0] > len(prev_ids):
        out[new_id - 1] = final_pos[-1]
    return out


def first_missing_id(prev_ids: List[int]) -> int:
    for i in range(1, 13):
        if i not in prev_ids:
            return i
    return prev_ids[-1] if prev_ids else 12


# ----------------------------
# Shot normalization + hammer/order
# ----------------------------
def compute_shot_norm_and_order(stones_df: pd.DataFrame) -> pd.DataFrame:
    df = stones_df.copy()
    df = df.sort_values(SHOT_KEY).reset_index(drop=True)

    end_group = ["CompetitionID", "SessionID", "GameID", "EndID"]

    df["ShotIndex"] = df.groupby(end_group).cumcount()
    df["ShotsInEnd"] = df.groupby(end_group)["ShotID"].transform("count")
    df["shot_norm"] = 0.0
    mask = df["ShotsInEnd"] > 1
    df.loc[mask, "shot_norm"] = df.loc[mask, "ShotIndex"] / (df.loc[mask, "ShotsInEnd"] - 1.0)

    first_team = df.groupby(end_group)["TeamID"].transform("first")
    last_team = df.groupby(end_group)["TeamID"].transform("last")

    df["is_hammer"] = (df["TeamID"] == last_team).astype(np.float32)
    df["team_order"] = (df["TeamID"] != first_team).astype(np.float32)  # first=0, other=1
    return df


def _coerce_context_dim(c_full: np.ndarray, expected_dim: int) -> np.ndarray:
    if expected_dim <= 0:
        return np.zeros((0,), dtype=np.float32)
    if c_full.shape[0] == expected_dim:
        return c_full
    if c_full.shape[0] > expected_dim:
        return c_full[:expected_dim]
    out = np.zeros((expected_dim,), dtype=np.float32)
    out[: c_full.shape[0]] = c_full
    return out


# ----------------------------
# Value model loader
# ----------------------------
def load_value_model(model_path: pathlib.Path, device: str = "cpu"):
    if model_path.suffix.lower() in (".json", ".xgb"):
        if xgb is None:
            raise ImportError("xgboost is not installed but an XGB model path was provided.")
        booster = xgb.Booster()
        booster.load_model(str(model_path))

        def predict(x_flat: np.ndarray, c_vec: np.ndarray) -> float:
            c_vec = np.asarray(c_vec, dtype=np.float32).reshape(-1)
            feat = np.concatenate([x_flat.reshape(-1).astype(np.float32), c_vec], axis=0)
            dmat = xgb.DMatrix(feat.reshape(1, -1))
            return float(booster.predict(dmat)[0])

        return predict, None

    ckpt = torch.load(model_path, map_location=device)
    input_dim = ckpt["input_dim"]
    cond_dim = int(ckpt["cond_dim"])
    hidden_dim = ckpt["hidden_dim"]
    num_stones = ckpt.get("num_stones", 12)
    args_dict = ckpt.get("args", {})
    n_layers = args_dict.get("n_layers", 4)
    n_heads = args_dict.get("n_heads", 4)
    dropout = args_dict.get("dropout", 0.1)

    model = ValueTransformer(
        input_dim=input_dim,
        cond_dim=cond_dim,
        hidden_dim=hidden_dim,
        num_stones=num_stones,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def predict(x_flat: np.ndarray, c_vec: np.ndarray) -> float:
        c_vec = _coerce_context_dim(np.asarray(c_vec, dtype=np.float32).reshape(-1), cond_dim)
        x_t = torch.tensor(x_flat.reshape(1, -1), dtype=torch.float32, device=device)
        c_t = torch.tensor(c_vec.reshape(1, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            val = model(x_t, c_t).item()
        return float(val)

    return predict, cond_dim


# ----------------------------
# Noise sampler (supports local/grouped/uniform)
# ----------------------------
@dataclass
class NoiseSampler:
    mode: str  # "local" or "grouped" or "uniform"
    default_std: np.ndarray
    task_handle: Dict[str, Dict]
    player_task: Dict[str, Dict]
    local_std: np.ndarray
    min_std: float = 1e-3
    uniform_low: np.ndarray | None = None
    uniform_high: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Dict, default_std: Iterable[float]):
        default = np.array(default_std, dtype=np.float32)
        mode = str(cfg.get("mode", "grouped")) if isinstance(cfg, dict) else "grouped"
        task_handle = cfg.get("by_task_handle", {}) if isinstance(cfg, dict) else {}
        player_task = cfg.get("by_player_task", {}) if isinstance(cfg, dict) else {}

        if isinstance(cfg, dict) and "default" in cfg and isinstance(cfg["default"], dict) and "std" in cfg["default"]:
            default = np.array(cfg["default"]["std"], dtype=np.float32)

        local_std = default.copy()
        if isinstance(cfg, dict) and "local" in cfg and isinstance(cfg["local"], dict) and "std" in cfg["local"]:
            local_std = np.array(cfg["local"]["std"], dtype=np.float32)

        min_std = float(cfg.get("local", {}).get("min_std", cfg.get("meta", {}).get("min_std", 1e-3))) if isinstance(cfg, dict) else 1e-3

        uniform_low = None
        uniform_high = None
        if isinstance(cfg, dict) and "uniform" in cfg and isinstance(cfg["uniform"], dict):
            u = cfg["uniform"]
            if "low" in u and "high" in u:
                uniform_low = np.array(u["low"], dtype=np.float32).reshape(4)
                uniform_high = np.array(u["high"], dtype=np.float32).reshape(4)

        return cls(
            mode=mode,
            default_std=default,
            task_handle=task_handle,
            player_task=player_task,
            local_std=local_std,
            min_std=min_std,
            uniform_low=uniform_low,
            uniform_high=uniform_high,
        )

    def _select_entry(self, task, handle, player_id=None) -> Dict | None:
        try:
            if player_id is not None and not pd.isna(player_id) and not pd.isna(task):
                key = f"player_{int(player_id)}_task_{int(task)}"
                if key in self.player_task:
                    return self.player_task[key]
            if pd.isna(task) or pd.isna(handle):
                return None
            th_key = f"task_{int(task)}_handle_{int(handle)}"
            return self.task_handle.get(th_key)
        except Exception:
            return None

    def draw(
        self,
        rng: np.random.Generator,
        center: np.ndarray,
        task,
        handle,
        player_id=None,
        cov_from_cfg: bool = False,
        bounds=None,
    ) -> np.ndarray:
        m = self.mode.lower().strip()

        if m in ("uniform", "global_uniform", "uniform_global"):
            if self.uniform_low is not None and self.uniform_high is not None:
                low = self.uniform_low
                high = self.uniform_high
            else:
                if bounds is None:
                    raise ValueError("NoiseSampler(mode='uniform') requires SolveBounds passed as 'bounds'.")
                low = np.array([bounds.speed_min, bounds.angle_min, bounds.spin_min, bounds.y0_min], dtype=np.float32)
                high = np.array([bounds.speed_max, bounds.angle_max, bounds.spin_max, bounds.y0_max], dtype=np.float32)
            return rng.uniform(low=low, high=high).astype(np.float32)

        if m == "local":
            std = np.maximum(self.local_std.astype(np.float32), self.min_std)
            cov = np.diag(std ** 2)
            return rng.multivariate_normal(center, cov).astype(np.float32)

        entry = self._select_entry(task, handle, player_id)
        if entry is None:
            std = np.maximum(self.default_std.astype(np.float32), self.min_std)
            cov = np.diag(std ** 2)
        else:
            std = np.maximum(np.array(entry.get("std", self.default_std), dtype=np.float32), self.min_std)
            if cov_from_cfg and "cov" in entry:
                cov = np.array(entry["cov"], dtype=np.float32)
            else:
                cov = np.diag(std ** 2)

        return rng.multivariate_normal(center, cov).astype(np.float32)


# ----------------------------
# Data loading / merge
# ----------------------------
def load_inverse(glob_pattern: str) -> pd.DataFrame:
    paths = sorted(pathlib.Path(".").glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No inverse files matched: {glob_pattern}")
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def prepare_dataframe_all(stones_csv: str, inverse_glob: str, only_solver_ok: bool) -> pd.DataFrame:
    inv_df = load_inverse(inverse_glob)

    stones_df = pd.read_csv(stones_csv)
    stones_df = compute_shot_norm_and_order(stones_df)
    stones_df = stones_df.sort_values(SHOT_KEY).reset_index(drop=True)

    end_group = ["CompetitionID", "SessionID", "GameID", "EndID"]
    stones_df["shot_norm_next"] = stones_df["shot_norm"].astype(float)
    stones_df["shot_norm_prev"] = stones_df.groupby(end_group)["shot_norm"].shift(1).astype(float)
    stones_df["ShotID_prev"] = stones_df.groupby(end_group)["ShotID"].shift(1)

    meta_cols = SHOT_KEY + [
        "TeamID",
        "PlayerID",
        "Task",
        "Handle",
        "shot_norm_prev",
        "shot_norm_next",
        "ShotID_prev",
        "is_hammer",
        "team_order",
    ]
    merged = pd.merge(inv_df, stones_df[meta_cols], on=SHOT_KEY, how="left", validate="one_to_one")

    if only_solver_ok and "solver_ok" in merged.columns:
        merged = merged[merged["solver_ok"] == True].copy()  # noqa: E712

    return merged.reset_index(drop=True)


# ----------------------------
# JAX simulator (batched, cached by shape)
# ----------------------------
def clip_to_bounds(x: np.ndarray, bounds: SolveBounds) -> np.ndarray:
    lo = np.array([bounds.speed_min, bounds.angle_min, bounds.spin_min, bounds.y0_min], dtype=np.float32)
    hi = np.array([bounds.speed_max, bounds.angle_max, bounds.spin_max, bounds.y0_max], dtype=np.float32)
    return np.clip(x, lo, hi).astype(np.float32)


class SimFnCache:
    def __init__(self, p: CurlingParams):
        self.p = p
        self._cache: Dict[Tuple[int, int], object] = {}

    def get(self, prev_n: int, batch_size: int):
        key = (int(prev_n), int(batch_size))
        if key in self._cache:
            return self._cache[key]

        p = self.p

        def _sim_one(prev_xy, x_params):
            return simulate_from_params(p, prev_xy, x_params, dynamic=False)

        def _sim_batch(prev_xy, x_batch):
            return jax.vmap(lambda x: _sim_one(prev_xy, x))(x_batch)

        sim_fn = jax.jit(_sim_batch)
        self._cache[key] = sim_fn
        return sim_fn


# ----------------------------
# Stats helpers
# ----------------------------
def percentile_of_score(samples: np.ndarray, obs: float) -> float:
    if samples.size == 0 or not np.isfinite(obs):
        return math.nan
    return float(np.mean(samples <= obs))


def cvar(values: np.ndarray, alpha: float) -> float:
    if values.size == 0:
        return math.nan
    k = max(1, int(values.size * alpha))
    part = np.partition(values, k - 1)[:k]
    return float(np.mean(part))


# ----------------------------
# Core scoring loop
# ----------------------------
def score_dataframe(
    df: pd.DataFrame,
    model_fn,
    sampler: NoiseSampler,
    curl_params: CurlingParams,
    bounds: SolveBounds,
    num_samples: int,
    seed: int,
    use_cov: bool,
    verbose_every: int = 0,
    desc: str = "",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sim_cache = SimFnCache(curl_params)
    B = int(num_samples)

    rows_out: List[dict] = []
    it = df.itertuples(index=False)

    pbar = tqdm(it, total=len(df), desc=desc) if desc else tqdm(it, total=len(df))
    for idx, row in enumerate(pbar):
        row_dict = row._asdict()
        srow = pd.Series(row_dict)

        prev_mat, _ = extract_state_from_row(srow, "prev")
        next_mat, _ = extract_state_from_row(srow, "next")

        prev_compact, prev_ids_compact = compact_positions(prev_mat)
        prev_n = int(prev_compact.shape[0])
        new_id = first_missing_id(prev_ids_compact)

        shot_norm_prev = float(row_dict.get("shot_norm_prev", np.nan))
        shot_norm_next = float(row_dict.get("shot_norm_next", np.nan))
        if not np.isfinite(shot_norm_prev) and np.isfinite(shot_norm_next):
            shot_norm_prev = shot_norm_next
        if not np.isfinite(shot_norm_prev):
            shot_norm_prev = 0.0
        if not np.isfinite(shot_norm_next):
            shot_norm_next = shot_norm_prev

        is_hammer = float(row_dict.get("is_hammer", 0.0))
        team_order = float(row_dict.get("team_order", 0.0))

        c_prev = np.array([shot_norm_prev, is_hammer, team_order], dtype=np.float32)
        c_next = np.array([shot_norm_next, is_hammer, team_order], dtype=np.float32)

        prev_raw_norm = normalize_raw_matrix(positions_m_to_raw_matrix(prev_mat))
        next_raw_norm = normalize_raw_matrix(positions_m_to_raw_matrix(next_mat))

        v_prev = model_fn(prev_raw_norm, c_prev)
        v_next = model_fn(next_raw_norm, c_next)
        dv_obs = v_next - v_prev

        est_params = np.array([row_dict.get(c, np.nan) for c in PARAM_COLS], dtype=np.float32)
        dv_samples = np.empty((0,), dtype=np.float32)

        valid_center = np.all(np.isfinite(est_params)) and np.isfinite(dv_obs) and (0 <= prev_n <= 12)
        if valid_center:
            x_batch = np.zeros((B, 4), dtype=np.float32)
            task = row_dict.get("Task", 0)
            handle = row_dict.get("Handle", 0)
            player_id = row_dict.get("PlayerID", None)

            for b in range(B):
                s = sampler.draw(
                    rng,
                    center=est_params,
                    task=task,
                    handle=handle,
                    player_id=player_id,
                    cov_from_cfg=use_cov,
                    bounds=bounds,
                )
                x_batch[b] = clip_to_bounds(s, bounds)

            sim_fn = sim_cache.get(prev_n, B)
            prev_j = jnp.asarray(prev_compact, dtype=jnp.float32)
            x_j = jnp.asarray(x_batch, dtype=jnp.float32)
            finals = np.asarray(sim_fn(prev_j, x_j))  # (B, prev_n+1,2)

            dv_list = np.empty((B,), dtype=np.float32)
            for b in range(B):
                full_final = assign_final_to_slots(finals[b], prev_ids_compact, new_id)
                final_raw_norm = normalize_raw_matrix(positions_m_to_raw_matrix(full_final))
                v_sim = model_fn(final_raw_norm, c_next)
                dv_list[b] = float(v_sim - v_prev)

            dv_samples = dv_list

        if dv_samples.size > 0:
            dv_mean = float(np.mean(dv_samples))
            dv_std = float(np.std(dv_samples, ddof=1)) if dv_samples.size > 1 else math.nan
            dv_p10 = float(np.percentile(dv_samples, 10))
            dv_p50 = float(np.percentile(dv_samples, 50))
            dv_p90 = float(np.percentile(dv_samples, 90))
            cvar10 = cvar(dv_samples, 0.10)
            pct_obs = percentile_of_score(dv_samples, float(dv_obs))
            z_obs = (float(dv_obs) - dv_mean) / (float(dv_std) + 1e-8) if np.isfinite(dv_std) else math.nan
            se_mean = float(dv_std) / math.sqrt(dv_samples.size) if dv_samples.size > 1 and np.isfinite(dv_std) else math.nan
        else:
            dv_mean = dv_std = dv_p10 = dv_p50 = dv_p90 = cvar10 = pct_obs = z_obs = se_mean = math.nan

        out_row = {k: row_dict.get(k, np.nan) for k in SHOT_KEY + ["TeamID", "PlayerID", "Task", "Handle"]}
        out_row.update(
            dict(
                shot_norm_prev=float(shot_norm_prev),
                shot_norm_next=float(shot_norm_next),
                is_hammer=float(is_hammer),
                team_order=float(team_order),
                v_prev=float(v_prev),
                v_next=float(v_next),
                dv_obs=float(dv_obs) if np.isfinite(dv_obs) else math.nan,
                dv_mean=dv_mean,
                dv_std=dv_std,
                dv_p10=dv_p10,
                dv_p50=dv_p50,
                dv_p90=dv_p90,
                cvar_10=cvar10,
                percentile_obs=pct_obs,
                z_obs=z_obs,
                se_mean=se_mean,
                sample_count=int(dv_samples.size),
                prev_N=prev_n,
                solver_ok=bool(row_dict.get("solver_ok", True)),
                hard_loss=float(row_dict.get("hard_loss_refine", math.nan)),
                est_speed=float(row_dict.get("est_speed", math.nan)),
                est_angle=float(row_dict.get("est_angle", math.nan)),
                est_spin=float(row_dict.get("est_spin", math.nan)),
                est_y0=float(row_dict.get("est_y0", math.nan)),
            )
        )
        rows_out.append(out_row)

        if verbose_every and (idx + 1) % int(verbose_every) == 0:
            print(f"[progress] processed {idx+1}/{len(df)} shots", flush=True)

    return pd.DataFrame(rows_out)


def write_csv(df: pd.DataFrame, out_path: pathlib.Path) -> pathlib.Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Monte Carlo shot-value scorer, per-competition outputs.")

    ap.add_argument("--stones-csv", type=str, default="2026/Stones.csv")
    ap.add_argument("--inverse-glob", type=str, default="inverseDataset/stones_with_estimates.chunk*.csv")
    ap.add_argument("--value-model", type=str, default="valueModel/value_model_synth_v4best.pt")
    ap.add_argument("--noise-config", type=str, default="noise_config_old.json")
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--only-solver-ok", action="store_true")
    ap.add_argument("--use-cov", action="store_true")
    ap.add_argument("--verbose-every", type=int, default=0)

    # competition controls
    ap.add_argument("--out-dir", type=str, default="shot_scores_by_competition", help="Directory for per-competition CSVs")
    ap.add_argument("--write-combined", action="store_true", help="Also write combined CSV after all competitions")
    ap.add_argument("--combined-name", type=str, default="shot_scores_all.csv")

    ap.add_argument("--competition-ids", type=str, default="", help="Comma-separated CompetitionID list to run (subset)")
    ap.add_argument("--only-competition", type=int, default=None, help="Run only this CompetitionID (single)")

    ap.add_argument("--limit-per-competition", type=int, default=None, help="Optional cap on rows per competition (debug)")

    # Smoke test controls (per competition)
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--smoke-limit", type=int, default=32)
    ap.add_argument("--smoke-samples", type=int, default=16)
    ap.add_argument("--smoke-prefix", type=str, default="smoke_", help="Prefix for per-competition smoke CSV files")

    args = ap.parse_args()

    if CURLING_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing simulation dependency (JAX/curling_sim_jax). "
            "Activate the correct environment or install JAX.\n"
            f"Original import error: {CURLING_IMPORT_ERROR}"
        )
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be > 0")

    # Load + merge once
    full_df = prepare_dataframe_all(
        stones_csv=args.stones_csv,
        inverse_glob=args.inverse_glob,
        only_solver_ok=bool(args.only_solver_ok),
    )
    if "CompetitionID" not in full_df.columns:
        raise SystemExit("Merged dataframe missing CompetitionID; check your joins / inputs.")

    # Choose competitions to run
    comp_ids = sorted(pd.unique(full_df["CompetitionID"].dropna()).tolist())
    comp_ids = [int(x) for x in comp_ids]

    if args.only_competition is not None:
        comp_ids = [int(args.only_competition)]

    if args.competition_ids.strip():
        wanted = [int(x.strip()) for x in args.competition_ids.split(",") if x.strip()]
        comp_ids = [c for c in comp_ids if c in set(wanted)]

    if not comp_ids:
        raise SystemExit("No competitions selected (after filtering).")

    print(f"[info] merged dataset rows: {len(full_df)}")
    print(f"[info] competitions selected: {comp_ids} (count={len(comp_ids)})")

    # Model + noise + sim params loaded once
    model_fn, model_cond_dim = load_value_model(pathlib.Path(args.value_model), device=args.device)
    if model_cond_dim is not None and model_cond_dim != 3:
        print(f"[warn] value model checkpoint cond_dim={model_cond_dim}; scorer will coerce context vector accordingly.", flush=True)

    cfg = {}
    cfg_path = pathlib.Path(args.noise_config)
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception as e:
            print(f"[warn] failed to parse noise_config ({cfg_path}): {e} ; using defaults", flush=True)
            cfg = {}
    sampler = NoiseSampler.from_config(cfg, default_std=[0.20, 0.05, 0.50, 0.10])

    curl_params = CurlingParams(dt=0.02, substeps=2, k_penalty=2.5e4, c_damp=220.0, k_curl=0.10)
    bounds = SolveBounds()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_parts: List[pd.DataFrame] = []

    # Iterate competitions
    for comp_id in comp_ids:
        comp_df = full_df[full_df["CompetitionID"] == comp_id].copy()
        if args.limit_per_competition is not None:
            comp_df = comp_df.head(int(args.limit_per_competition)).copy()

        print(f"\n[comp {comp_id}] rows={len(comp_df)}")

        if len(comp_df) == 0:
            print(f"[comp {comp_id}] skipping (no rows)")
            continue

        # Smoke test per competition
        if not args.no_smoke:
            smoke_n = min(int(args.smoke_limit), len(comp_df))
            smoke_df = comp_df.head(smoke_n).copy()
            print(f"[comp {comp_id}][smoke] shots={smoke_n}, samples={int(args.smoke_samples)}")

            smoke_scores = score_dataframe(
                smoke_df,
                model_fn=model_fn,
                sampler=sampler,
                curl_params=curl_params,
                bounds=bounds,
                num_samples=int(args.smoke_samples),
                seed=int(args.seed) + 13 * int(comp_id),
                use_cov=bool(args.use_cov),
                verbose_every=0,
                desc=f"smoke comp {comp_id}",
            )
            smoke_out = out_dir / f"{args.smoke_prefix}{comp_id}.csv"
            write_csv(smoke_scores, smoke_out)
            print(f"[comp {comp_id}][smoke] wrote: {smoke_out}")

            required_cols = ["dv_obs", "dv_mean", "dv_std", "sample_count", "is_hammer", "team_order"]
            for c in required_cols:
                if c not in smoke_scores.columns:
                    raise SystemExit(f"[comp {comp_id}][smoke] missing required output column: {c}")
            if smoke_scores["dv_obs"].notna().sum() == 0:
                raise SystemExit(f"[comp {comp_id}][smoke] dv_obs is all NaN; check value model inputs / joins.")
            print(f"[comp {comp_id}][smoke] OK. Proceeding to full.")

        # Full run per competition
        print(f"[comp {comp_id}][full] samples={int(args.num_samples)}")
        comp_scores = score_dataframe(
            comp_df,
            model_fn=model_fn,
            sampler=sampler,
            curl_params=curl_params,
            bounds=bounds,
            num_samples=int(args.num_samples),
            seed=int(args.seed) + 999 + 37 * int(comp_id),
            use_cov=bool(args.use_cov),
            verbose_every=int(args.verbose_every),
            desc=f"full comp {comp_id}",
        )

        comp_out = out_dir / f"shot_scores_comp_{comp_id}.csv"
        write_csv(comp_scores, comp_out)
        print(f"[comp {comp_id}][done] wrote {len(comp_scores)} rows -> {comp_out}")

        if args.write_combined:
            all_parts.append(comp_scores)

    if args.write_combined and all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
        combined_out = out_dir / args.combined_name
        write_csv(combined, combined_out)
        print(f"\n[all] wrote combined {len(combined)} rows -> {combined_out}")


if __name__ == "__main__":
    main()
