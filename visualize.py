#!/usr/bin/env python3
"""
viz_mc_shot.py

Visualize Monte Carlo neighbor outcomes for one shot.

Context features:
  c = [shot_norm, is_hammer, team_order]

Conditioning:
  - xScore(prev) uses shot_norm_prev (plus is_hammer/team_order for the shot team)
  - xScore(next_obs) and xScore(next_sim) use shot_norm_next (plus same is_hammer/team_order)

Noise:
  - supports noise v2 local schema ("mode":"local")

Stone styling on boards:
  - Throwing team stones: white fill + black outline
  - Opponent stones: black fill + black outline

IMPORTANT FIX (team coloring / "new stone" color):
  The 12-slot state has no explicit team ownership per stone. We use the standard
  "slot parity encodes team within an end" convention. However, the mapping between
  (odd/even slots) and (TeamID) flips by end depending on which team throws first.

  The previous version incorrectly placed the simulated "new stone" into the FIRST
  missing slot, which can have opponent parity, causing the thrown stone to appear
  black in simulated boards.

  This version chooses the missing slot whose parity matches the THROWER parity
  (inferred from (team_order, shot_index)), so the thrown stone is consistently white.

NEW (human-readable labels):
  - CompetitionID -> CompetitionName via Competition.csv
  - TeamID -> Team Name via Teams.csv
  - PlayerID -> Player name via Players.csv or Competitors.csv (if available)

  Note: Many WCF/WCF-like exports have Competitors.csv WITHOUT PlayerID. In that case,
  we cannot deterministically map PlayerID->name. The script will:
    - print Team roster names (from Competitors.csv Reportingname) in the suptitle, and
    - show PlayerID as fallback.

Usage:
  python viz_mc_shot.py \
    --stones-csv 2026/Stones.csv \
    --inverse-glob inverseDataset/stones_with_estimates.chunk*.csv \
    --value-model valueModel/value_model_synth_v4best.pt \
    --noise-config noise_config.json \
    --out mc_viz.png \
    --neighbors-out mc_neighbors.csv

Optional:
  --shot "comp,sess,game,end,shot"
  --only-solver-ok
  --use-cov

Metadata (optional; defaults are sensible for your repo layout):
  --competition-csv 2026/Competition.csv
  --teams-csv 2026/Teams.csv
  --competitors-csv 2026/Competitors.csv
  --players-csv 2026/Players.csv   (if you have it)
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR / "inverse"))
sys.path.append(str(THIS_DIR / "valueModel"))

# --- JAX sim ---
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
# Metadata loading (names)
# ----------------------------
def _read_csv_if_exists(path: str | pathlib.Path) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


@dataclass
class MetaLookup:
    comp_name: Dict[int, str]
    team_name: Dict[Tuple[int, int], str]          # (CompetitionID, TeamID) -> "Canada"
    team_noc: Dict[Tuple[int, int], str]           # (CompetitionID, TeamID) -> "CAN"
    team_roster: Dict[Tuple[int, int], List[str]]  # (CompetitionID, TeamID) -> ["HOMAN Rachel", ...]
    player_name: Dict[int, str]                    # PlayerID -> "Rachel Homan" (if available)

    @classmethod
    def from_files(
        cls,
        competition_csv: str,
        teams_csv: str,
        competitors_csv: str,
        players_csv: str = "",
    ) -> "MetaLookup":
        comp_df = _read_csv_if_exists(competition_csv)
        teams_df = _read_csv_if_exists(teams_csv)
        comps_df = _read_csv_if_exists(competitors_csv)
        players_df = _read_csv_if_exists(players_csv) if players_csv else pd.DataFrame()

        comp_name: Dict[int, str] = {}
        if not comp_df.empty:
            # accept either CompetitionID or competition id variants
            if "CompetitionID" in comp_df.columns and "CompetitionName" in comp_df.columns:
                for r in comp_df.itertuples(index=False):
                    try:
                        comp_name[int(getattr(r, "CompetitionID"))] = str(getattr(r, "CompetitionName"))
                    except Exception:
                        continue

        team_name: Dict[Tuple[int, int], str] = {}
        team_noc: Dict[Tuple[int, int], str] = {}
        if not teams_df.empty and ("CompetitionID" in teams_df.columns) and ("TeamID" in teams_df.columns):
            name_col = "Name" if "Name" in teams_df.columns else ("Reportingname" if "Reportingname" in teams_df.columns else None)
            for r in teams_df.itertuples(index=False):
                try:
                    k = (int(getattr(r, "CompetitionID")), int(getattr(r, "TeamID")))
                    if name_col is not None:
                        team_name[k] = str(getattr(r, name_col))
                    if "NOC" in teams_df.columns:
                        team_noc[k] = str(getattr(r, "NOC"))
                except Exception:
                    continue

        team_roster: Dict[Tuple[int, int], List[str]] = {}
        if not comps_df.empty and ("CompetitionID" in comps_df.columns) and ("TeamID" in comps_df.columns):
            roster_col = None
            for c in ["Reportingname", "ReportingName", "Name", "FullName"]:
                if c in comps_df.columns:
                    roster_col = c
                    break
            if roster_col is not None:
                for (cid, tid), g in comps_df.groupby(["CompetitionID", "TeamID"]):
                    try:
                        roster = [str(x) for x in g[roster_col].dropna().astype(str).tolist()]
                        team_roster[(int(cid), int(tid))] = roster
                    except Exception:
                        continue

        player_name: Dict[int, str] = {}
        if not players_df.empty:
            # Heuristic: look for PlayerID and one of these name columns
            if "PlayerID" in players_df.columns:
                name_col = None
                for c in ["Name", "FullName", "Reportingname", "ReportingName", "GivenName"]:
                    if c in players_df.columns:
                        name_col = c
                        break
                if name_col is not None:
                    for r in players_df.itertuples(index=False):
                        try:
                            pid = int(getattr(r, "PlayerID"))
                            nm = str(getattr(r, name_col))
                            player_name[pid] = nm
                        except Exception:
                            continue

        # Some Competitors.csv variants include PlayerID; add those if present
        if not comps_df.empty and ("PlayerID" in comps_df.columns):
            roster_col = None
            for c in ["Reportingname", "ReportingName", "Name", "FullName"]:
                if c in comps_df.columns:
                    roster_col = c
                    break
            if roster_col is not None:
                for r in comps_df.itertuples(index=False):
                    try:
                        pid = int(getattr(r, "PlayerID"))
                        nm = str(getattr(r, roster_col))
                        if pid not in player_name:
                            player_name[pid] = nm
                    except Exception:
                        continue

        return cls(
            comp_name=comp_name,
            team_name=team_name,
            team_noc=team_noc,
            team_roster=team_roster,
            player_name=player_name,
        )

    def get_comp(self, competition_id: int) -> str:
        return self.comp_name.get(int(competition_id), f"Competition {int(competition_id)}")

    def get_team(self, competition_id: int, team_id: int) -> str:
        k = (int(competition_id), int(team_id))
        nm = self.team_name.get(k, f"Team {int(team_id)}")
        noc = self.team_noc.get(k, "")
        return f"{nm} ({noc})" if noc else nm

    def get_roster_str(self, competition_id: int, team_id: int, max_names: int = 6) -> str:
        k = (int(competition_id), int(team_id))
        roster = self.team_roster.get(k, [])
        if not roster:
            return ""
        if len(roster) <= max_names:
            return ", ".join(roster)
        return ", ".join(roster[:max_names]) + f", +{len(roster) - max_names} more"

    def get_player(self, player_id) -> str:
        try:
            if pd.isna(player_id):
                return ""
            pid = int(player_id)
        except Exception:
            return ""
        return self.player_name.get(pid, f"PlayerID {pid}")


# ----------------------------
# Value model helpers
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


def load_value_model(model_path: pathlib.Path, device: str = "cpu"):
    """
    Returns:
      predict(x_flat, c_vec) -> float
    """
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


@dataclass
class NoiseSampler:
    mode: str
    default_std: np.ndarray
    task_handle: Dict[str, Dict]
    player_task: Dict[str, Dict]
    local_std: np.ndarray
    min_std: float = 1e-3

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

        min_std = float(
            cfg.get("local", {}).get("min_std", cfg.get("meta", {}).get("min_std", 1e-3))
        ) if isinstance(cfg, dict) else 1e-3

        return cls(
            mode=mode,
            default_std=default,
            task_handle=task_handle,
            player_task=player_task,
            local_std=local_std,
            min_std=min_std,
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
    ) -> np.ndarray:
        if self.mode.lower() == "local":
            std = np.maximum(self.local_std.astype(np.float32), self.min_std)
            cov = np.diag(std**2)
            return rng.multivariate_normal(center, cov).astype(np.float32)

        entry = self._select_entry(task, handle, player_id)
        if entry is None:
            std = np.maximum(self.default_std.astype(np.float32), self.min_std)
            cov = np.diag(std**2)
        else:
            std = np.maximum(np.array(entry.get("std", self.default_std), dtype=np.float32), self.min_std)
            if cov_from_cfg and "cov" in entry:
                cov = np.array(entry["cov"], dtype=np.float32)
            else:
                cov = np.diag(std**2)
        return rng.multivariate_normal(center, cov).astype(np.float32)


# ----------------------------
# Data loading / merge
# ----------------------------
def load_inverse(glob_pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No inverse files matched: {glob_pattern}")
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


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
    df["team_order"] = (df["TeamID"] != first_team).astype(np.float32)
    return df


def prepare_merged(stones_csv: str, inverse_glob: str) -> pd.DataFrame:
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
        "Points",
        "is_hammer",
        "team_order",
        "ShotIndex",
        "ShotsInEnd",
    ]
    meta_cols = [c for c in meta_cols if c in stones_df.columns]
    merged = pd.merge(inv_df, stones_df[meta_cols], on=SHOT_KEY, how="left", validate="one_to_one")
    return merged


def extract_state_from_row(row: pd.Series, prefix: str) -> np.ndarray:
    mat = np.full((12, 2), np.nan, dtype=np.float32)
    for i in range(1, 13):
        x = row.get(f"{prefix}_stone_{i}_x_m", np.nan)
        y = row.get(f"{prefix}_stone_{i}_y_m", np.nan)
        if not (pd.isna(x) or pd.isna(y)):
            mat[i - 1, 0] = float(x)
            mat[i - 1, 1] = float(y)
    return mat


def compact_positions(mat: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    keep_mask = ~np.isnan(mat).any(axis=1)
    compact = mat[keep_mask]
    ids = [i + 1 for i, flag in enumerate(keep_mask) if flag]
    return compact.astype(np.float32), ids


def assign_final_to_slots(final_pos: np.ndarray, prev_ids: List[int], new_id: int) -> np.ndarray:
    """
    Place compact sim output back into a 12-slot matrix.

    Assumes final_pos ordering corresponds to prev_ids order for existing stones,
    and final_pos[-1] is the thrown stone.
    """
    out = np.full((12, 2), np.nan, dtype=np.float32)
    for idx, sid in enumerate(prev_ids):
        if idx < final_pos.shape[0]:
            out[sid - 1] = final_pos[idx]
    if final_pos.shape[0] > len(prev_ids):
        out[new_id - 1] = final_pos[-1]
    return out


def clip_to_bounds(x: np.ndarray, bounds: SolveBounds) -> np.ndarray:
    lo = np.array([bounds.speed_min, bounds.angle_min, bounds.spin_min, bounds.y0_min], dtype=np.float32)
    hi = np.array([bounds.speed_max, bounds.angle_max, bounds.spin_max, bounds.y0_max], dtype=np.float32)
    return np.clip(x, lo, hi).astype(np.float32)


def infer_thrower_parity_from_context(shot_norm_next: float, shots_in_end: int, team_order: float) -> int:
    """
    Returns 0 or 1: which slot parity corresponds to the THROWER for this shot.

    Assumptions:
      - slot parity encodes teams consistently within an end (odd/even slots)
      - shots alternate between teams
      - team_order: 0 => this team throws first in the end; 1 => second

    Reconstruct shot_index from shot_norm_next and shots_in_end:
      shot_index ≈ round(shot_norm_next * (shots_in_end - 1))
    """
    if shots_in_end is None or int(shots_in_end) <= 1 or not np.isfinite(shot_norm_next):
        shot_index = 0
    else:
        shots_in_end = int(shots_in_end)
        shot_index = int(np.clip(np.round(float(shot_norm_next) * (shots_in_end - 1)), 0, shots_in_end - 1))
    return (int(round(float(team_order))) + int(shot_index)) % 2


def choose_missing_slot_for_thrower(prev_ids: List[int], thrower_parity: int) -> int:
    """
    Choose the missing slot whose parity matches the thrower parity.

    Slot numbering is 1..12, while parity in plotting uses 0-based slot index (0..11).
    0-based parity:
      - slot 1 -> index 0 -> parity 0
      - slot 2 -> index 1 -> parity 1
    So parity(index) = (slot_id - 1) % 2.
    """
    missing = [i for i in range(1, 13) if i not in prev_ids]
    if not missing:
        return 12
    for sid in missing:
        if ((sid - 1) % 2) == int(thrower_parity):
            return sid
    return missing[0]


def plot_board(ax, positions_m_12x2: np.ndarray, title: str, thrower_parity: int):
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(MIN_Y, MAX_Y)
    ax.set_ylim(MIN_X, MAX_X)
    ax.set_xlabel("lateral (m)")
    ax.set_ylabel("along-sheet (m)")

    rings = [1.8288, 1.2192, 0.6096, 0.1524]
    for r in rings:
        ax.add_patch(plt.Circle((0.0, 0.0), r, fill=False))

    pts = positions_m_12x2
    mask = ~np.isnan(pts).any(axis=1)
    if not mask.any():
        return

    pts2 = pts[mask]
    slot_idx = np.nonzero(mask)[0]  # 0..11
    parity = slot_idx % 2

    thr = parity == int(thrower_parity)
    opp = ~thr

    if np.any(opp):
        ax.scatter(
            pts2[opp, 1],
            pts2[opp, 0],
            s=45,
            marker="o",
            facecolors="black",
            edgecolors="black",
            linewidths=1.1,
            alpha=0.95,
            zorder=3,
        )

    if np.any(thr):
        ax.scatter(
            pts2[thr, 1],
            pts2[thr, 0],
            s=45,
            marker="o",
            facecolors="white",
            edgecolors="black",
            linewidths=1.5,
            alpha=0.98,
            zorder=4,
        )


def main():
    ap = argparse.ArgumentParser(description="Visualize MC neighbor outcomes for one shot.")
    ap.add_argument("--stones-csv", type=str, default="2026/Stones.csv")
    ap.add_argument("--inverse-glob", type=str, default="inverseDataset/stones_with_estimates.chunk*.csv")
    ap.add_argument("--value-model", type=str, default="valueModel/value_model_synth.pt")
    ap.add_argument("--noise-config", type=str, default="noise_config.json")
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=123341)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="mc_viz.png")
    ap.add_argument("--neighbors-out", type=str, default="mc_neighbors.csv")
    ap.add_argument("--shot", type=str, default="", help='Optional shot key "comp,sess,game,end,shot"')
    ap.add_argument("--use-cov", action="store_true")
    ap.add_argument("--only-solver-ok", action="store_true")

    # metadata sources (optional)
    ap.add_argument("--competition-csv", type=str, default="2026/Competition.csv")
    ap.add_argument("--teams-csv", type=str, default="2026/Teams.csv")
    ap.add_argument("--competitors-csv", type=str, default="2026/Competitors.csv")
    ap.add_argument("--players-csv", type=str, default="", help="Optional Players.csv (if you have PlayerID->Name mapping).")

    args = ap.parse_args()

    if CURLING_IMPORT_ERROR is not None:
        raise SystemExit(f"JAX/curling_sim_jax import error: {CURLING_IMPORT_ERROR}")

    meta = MetaLookup.from_files(
        competition_csv=args.competition_csv,
        teams_csv=args.teams_csv,
        competitors_csv=args.competitors_csv,
        players_csv=args.players_csv,
    )

    merged = prepare_merged(args.stones_csv, args.inverse_glob)
    if args.only_solver_ok and "solver_ok" in merged.columns:
        merged = merged[merged["solver_ok"] == True].copy()  # noqa: E712
    merged = merged.reset_index(drop=True)

    if len(merged) == 0:
        raise SystemExit("No rows available after filtering.")

    # Choose shot
    if args.shot.strip():
        parts = [int(x.strip()) for x in args.shot.split(",")]
        if len(parts) != 5:
            raise SystemExit("--shot must be comp,sess,game,end,shot")
        key = dict(zip(SHOT_KEY, parts))
        mask = np.ones(len(merged), dtype=bool)
        for k, v in key.items():
            mask &= (merged[k].astype(int).values == int(v))
        if not mask.any():
            raise SystemExit(f"Shot not found: {args.shot}")
        shot_row = merged.loc[mask].iloc[0]
    else:
        rng_pick = np.random.default_rng(args.seed)
        shot_row = merged.iloc[int(rng_pick.integers(0, len(merged)))]

    model_fn, model_cond_dim = load_value_model(pathlib.Path(args.value_model), device=args.device)
    if model_cond_dim is not None and model_cond_dim != 3:
        print(
            f"[warn] value model checkpoint cond_dim={model_cond_dim}; viz will coerce context vector accordingly.",
            flush=True,
        )

    cfg = {}
    cfg_path = pathlib.Path(args.noise_config)
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    sampler = NoiseSampler.from_config(cfg, default_std=[0.20, 0.05, 0.50, 0.10])

    curl_params = CurlingParams(dt=0.02, substeps=2, k_penalty=2.5e4, c_damp=220.0, k_curl=0.10)
    bounds = SolveBounds()

    prev_mat = extract_state_from_row(shot_row, "prev")
    next_mat_obs = extract_state_from_row(shot_row, "next")

    # Context features
    shot_norm_prev = float(shot_row.get("shot_norm_prev", np.nan))
    shot_norm_next = float(shot_row.get("shot_norm_next", np.nan))
    if not np.isfinite(shot_norm_prev) and np.isfinite(shot_norm_next):
        shot_norm_prev = shot_norm_next
    if not np.isfinite(shot_norm_prev):
        shot_norm_prev = 0.0
    if not np.isfinite(shot_norm_next):
        shot_norm_next = shot_norm_prev

    is_hammer = float(shot_row.get("is_hammer", 0.0))
    team_order = float(shot_row.get("team_order", 0.0))
    shots_in_end = int(shot_row.get("ShotsInEnd", 0)) if pd.notna(shot_row.get("ShotsInEnd", np.nan)) else 0

    # Infer which slot parity corresponds to the throwing team (for coloring)
    thrower_parity = infer_thrower_parity_from_context(shot_norm_next, shots_in_end, team_order)

    prev_compact, prev_ids = compact_positions(prev_mat)
    new_id = choose_missing_slot_for_thrower(prev_ids, thrower_parity)

    c_prev = np.array([shot_norm_prev, is_hammer, team_order], dtype=np.float32)
    c_next = np.array([shot_norm_next, is_hammer, team_order], dtype=np.float32)

    v_prev = model_fn(normalize_raw_matrix(positions_m_to_raw_matrix(prev_mat)), c_prev)
    v_next_obs = model_fn(normalize_raw_matrix(positions_m_to_raw_matrix(next_mat_obs)), c_next)
    dv_obs = v_next_obs - v_prev

    est_params = np.array([shot_row.get(c, np.nan) for c in PARAM_COLS], dtype=np.float32)
    if not np.all(np.isfinite(est_params)):
        raise SystemExit("Chosen shot has non-finite inverse parameters; pick another.")

    rng = np.random.default_rng(args.seed)

    B = int(args.num_samples)
    x_batch = np.zeros((B, 4), dtype=np.float32)
    for b in range(B):
        s = sampler.draw(
            rng,
            center=est_params,
            task=shot_row.get("Task", 0),
            handle=shot_row.get("Handle", 0),
            player_id=shot_row.get("PlayerID", None),
            cov_from_cfg=bool(args.use_cov),
        )
        x_batch[b] = clip_to_bounds(s, bounds)

    prev_j = jnp.asarray(prev_compact, jnp.float32)

    def _sim_one(x_params):
        return simulate_from_params(curl_params, prev_j, x_params, dynamic=False)

    sim_fn = jax.jit(jax.vmap(_sim_one))
    finals = np.asarray(sim_fn(jnp.asarray(x_batch, jnp.float32)))  # (B, prev_n+1, 2)

    dv_sims = np.zeros((B,), dtype=np.float32)
    v_sims = np.zeros((B,), dtype=np.float32)
    finals_12 = np.full((B, 12, 2), np.nan, dtype=np.float32)

    for b in range(B):
        full_final = assign_final_to_slots(finals[b], prev_ids, new_id)
        finals_12[b] = full_final
        v_sim = model_fn(normalize_raw_matrix(positions_m_to_raw_matrix(full_final)), c_next)
        v_sims[b] = float(v_sim)
        dv_sims[b] = float(v_sim - v_prev)

    def pick_idx_for_percentile(p: float) -> int:
        q = np.percentile(dv_sims, p)
        return int(np.argmin(np.abs(dv_sims - q)))

    idx_p10 = pick_idx_for_percentile(10)
    idx_p50 = pick_idx_for_percentile(50)
    idx_p90 = pick_idx_for_percentile(90)

    # Save neighbors table
    neigh_out = []
    for b in range(B):
        neigh_out.append(
            dict(
                b=b,
                est_speed=float(x_batch[b, 0]),
                est_angle=float(x_batch[b, 1]),
                est_spin=float(x_batch[b, 2]),
                est_y0=float(x_batch[b, 3]),
                v_sim=float(v_sims[b]),
                dv_sim=float(dv_sims[b]),
            )
        )
    pd.DataFrame(neigh_out).to_csv(args.neighbors_out, index=False)

    # Plot
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.2])

    ax_scatter = fig.add_subplot(gs[0, 0:3])
    ax_hist = fig.add_subplot(gs[0, 3:5])

    ax_prev = fig.add_subplot(gs[1, 0])
    ax_obs = fig.add_subplot(gs[1, 1])
    ax_p10 = fig.add_subplot(gs[1, 2])
    ax_p50 = fig.add_subplot(gs[1, 3])
    ax_p90 = fig.add_subplot(gs[1, 4])

    ax_scatter.scatter(np.arange(B), dv_sims, s=10)
    ax_scatter.axhline(dv_obs, linestyle="--")
    ax_scatter.set_title("MC neighbors: ΔxScore = xScore(next_sim) - xScore(prev)")
    ax_scatter.set_xlabel("sample index")
    ax_scatter.set_ylabel("ΔxScore")
    ax_scatter.scatter([idx_p10, idx_p50, idx_p90], [dv_sims[idx_p10], dv_sims[idx_p50], dv_sims[idx_p90]], s=40)
    ax_scatter.text(idx_p10, dv_sims[idx_p10], " p10")
    ax_scatter.text(idx_p50, dv_sims[idx_p50], " p50")
    ax_scatter.text(idx_p90, dv_sims[idx_p90], " p90")

    ax_hist.hist(dv_sims, bins=30)
    ax_hist.axvline(dv_obs, linestyle="--")
    ax_hist.axvline(np.percentile(dv_sims, 10), linestyle=":")
    ax_hist.axvline(np.percentile(dv_sims, 50), linestyle=":")
    ax_hist.axvline(np.percentile(dv_sims, 90), linestyle=":")
    ax_hist.set_title("ΔxScore distribution")
    ax_hist.set_xlabel("ΔxScore")
    ax_hist.set_ylabel("count")

    plot_board(ax_prev, prev_mat, f"Prev\nxScore={v_prev:.3f}", thrower_parity)
    plot_board(ax_obs, next_mat_obs, f"Observed\nxScore={v_next_obs:.3f}\nΔxScore={dv_obs:.3f}", thrower_parity)
    plot_board(ax_p10, finals_12[idx_p10], f"Sim p10\nxScore={v_sims[idx_p10]:.3f}\nΔxScore={dv_sims[idx_p10]:.3f}", thrower_parity)
    plot_board(ax_p50, finals_12[idx_p50], f"Sim p50\nxScore={v_sims[idx_p50]:.3f}\nΔxScore={dv_sims[idx_p50]:.3f}", thrower_parity)
    plot_board(ax_p90, finals_12[idx_p90], f"Sim p90\nxScore={v_sims[idx_p90]:.3f}\nΔxScore={dv_sims[idx_p90]:.3f}", thrower_parity)

    comp, sess, game, end, shotid = [int(shot_row[k]) for k in SHOT_KEY]
    team_id = int(shot_row.get("TeamID", -1)) if pd.notna(shot_row.get("TeamID", np.nan)) else -1
    player_id = shot_row.get("PlayerID", np.nan)

    comp_name = meta.get_comp(comp)
    team_label = meta.get_team(comp, team_id) if team_id >= 0 else ""
    player_label = meta.get_player(player_id) if pd.notna(player_id) else ""
    roster_str = meta.get_roster_str(comp, team_id) if team_id >= 0 else ""

    # If we cannot map PlayerID to an actual name, give a small hint via roster.
    player_line = f"Player: {player_label}" if player_label else "Player: (unknown)"
    if player_label.startswith("PlayerID") and roster_str:
        player_line += f" | Team roster: {roster_str}"

    fig.suptitle(
        f"{comp_name}\n"
        f"Shot {comp},{sess},{game},End {end},Shot {shotid} | Team: {team_label}\n"
        f"{player_line} | Task={shot_row.get('Task')} Handle={shot_row.get('Handle')}\n"
        f"context: is_hammer={is_hammer:.0f} team_order={team_order:.0f} | "
        f"Thrower stones=white (outlined), Opponent=black | "
        f"Inverse center: speed={est_params[0]:.3f}, angle={est_params[1]:.3f}, spin={est_params[2]:.3f}, y0={est_params[3]:.3f}",
        y=0.98,
        fontsize=12,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"[done] wrote figure to: {out_path}")
    print(f"[done] wrote neighbor table to: {args.neighbors_out}")
    print(
        f"[info] observed ΔxScore={dv_obs:.4f} | "
        f"p10={np.percentile(dv_sims,10):.4f} | p50={np.percentile(dv_sims,50):.4f} | p90={np.percentile(dv_sims,90):.4f}"
    )


if __name__ == "__main__":
    main()
