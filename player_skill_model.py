#!/usr/bin/env python3
"""
Convert per-shot Monte Carlo scores into player-by-task skill estimates.

Recommended metric: excess = dv_obs - dv_mean (execution above/below expected).
Applies simple empirical-Bayes shrinkage per Task to stabilize low-sample players.

NEW:
- Filter to include only shots where hard_loss < --max-hard-loss (default 0.5).
  (If hard_loss missing, those rows are dropped by default for safety.)
"""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd

SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]


def _load_table(path: str) -> pd.DataFrame:
    """
    Loads shot scores from parquet if possible; otherwise falls back to CSV.

    Fallback behavior:
      - If path ends with .csv -> read_csv(path)
      - If path ends with .parquet:
          * try read_parquet(path)
          * on ImportError / engine errors / file missing:
              - try read_csv(path with suffix replaced by .csv)
              - otherwise re-raise with a clear message
    """
    p = pathlib.Path(path)
    suf = p.suffix.lower()

    if suf == ".csv":
        return pd.read_csv(p)

    if suf == ".parquet":
        try:
            return pd.read_parquet(p)
        except Exception as e:
            csv_fallback = p.with_suffix(".csv")
            if csv_fallback.exists():
                print("[warn] Parquet read failed; falling back to CSV.")
                return pd.read_csv(csv_fallback)
            raise RuntimeError("Failed to read parquet and no CSV fallback found.") from e

    # Unknown extension: try CSV first, then parquet
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_parquet(p)


def _load_player_names(competitors_csv: str) -> Dict[Tuple[int, int], str]:
    """
    Competitors.csv lacks PlayerID; infer PlayerID within team by order of appearance.
    """
    df = pd.read_csv(competitors_csv)
    df["player_ord"] = df.groupby("TeamID").cumcount() + 1
    mapping = {(int(r.TeamID), int(r.player_ord)): str(r.Reportingname) for _, r in df.iterrows()}
    return mapping


def _load_team_names(teams_csv: str) -> Dict[int, str]:
    df = pd.read_csv(teams_csv)
    return {int(r.TeamID): str(r.Name) for _, r in df.iterrows()}


def _make_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "excess":
        return df["dv_obs"] - df["dv_mean"]
    if metric == "z_obs" and "z_obs" in df.columns:
        return df["z_obs"]
    raise ValueError(f"Unknown metric {metric}")


def _shrink_mean(raw_mean: float, n: int, prior_mean: float, prior_strength: float) -> float:
    weight = n / (n + prior_strength)
    return weight * raw_mean + (1 - weight) * prior_mean


def _shrink_se(raw_std: float, n: int, prior_strength: float, prior_std: float) -> float:
    eff_var = max(raw_std, prior_std) ** 2
    denom = max(1.0, n + prior_strength)
    return math.sqrt(eff_var / denom)


def main():
    ap = argparse.ArgumentParser(description="Fit player-by-task skill model from shot_scores.")
    ap.add_argument("--shot-scores", type=str, default="shot_scores_old.csv", help="Output of score_shots_mc.py")
    ap.add_argument("--competitors-csv", type=str, default="2026/Competitors.csv", help="Competitors metadata (for names)")
    ap.add_argument("--teams-csv", type=str, default="2026/Teams.csv", help="Teams metadata")
    ap.add_argument("--metric", type=str, default="excess", choices=["excess", "z_obs"], help="Which metric to model")
    ap.add_argument("--prior-strength", type=float, default=15.0, help="Pseudo-count for shrinkage toward task mean")
    ap.add_argument("--min-shots", type=int, default=8, help="Minimum shots to report a player-task row")

    # NEW: hard_loss filtering
    ap.add_argument("--max-hard-loss", type=float, default=0.5, help="Keep only rows with hard_loss < this value")
    ap.add_argument(
        "--keep-missing-hard-loss",
        action="store_true",
        help="If set, keep rows where hard_loss is missing/NaN (otherwise drop them).",
    )

    ap.add_argument("--out-player-task", type=str, default="player_task_skill.csv")
    ap.add_argument("--out-player-summary", type=str, default="player_summary.csv")
    args = ap.parse_args()

    df = _load_table(args.shot_scores)

    # --- NEW: filter on hard_loss ---
    if "hard_loss" not in df.columns:
        raise SystemExit(
            "shot_scores is missing required column 'hard_loss'. "
            "Ensure score_shots_mc writes hard_loss (it currently writes hard_loss from hard_loss_refine)."
        )

    before = len(df)
    if args.keep_missing_hard_loss:
        keep_mask = df["hard_loss"].isna() | (df["hard_loss"] < float(args.max_hard_loss))
        df = df[keep_mask].copy()
    else:
        df = df[df["hard_loss"].notna() & (df["hard_loss"] < float(args.max_hard_loss))].copy()

    after = len(df)
    print(f"[info] hard_loss filter: kept {after}/{before} rows (max_hard_loss={args.max_hard_loss})")

    # Existing required columns for modeling
    df = df.dropna(subset=["dv_obs", "dv_mean", "Task", "PlayerID", "TeamID"])
    df["metric"] = _make_metric(df, args.metric)

    player_names = _load_player_names(args.competitors_csv)
    team_names = _load_team_names(args.teams_csv)

    task_stats = (
        df.groupby("Task")["metric"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "task_mean", "std": "task_std", "count": "task_count"})
    )
    global_mean = float(df["metric"].mean()) if len(df) else float("nan")
    global_std = float(df["metric"].std(ddof=1)) if len(df) > 1 else float("nan")

    rows = []
    for (player_id, team_id, task), g in df.groupby(["PlayerID", "TeamID", "Task"]):
        n = len(g)
        if n < args.min_shots:
            continue

        raw_mean = float(g["metric"].mean())
        raw_std = float(g["metric"].std(ddof=1)) if n > 1 else 0.0

        task_row = task_stats.loc[task] if task in task_stats.index else None
        prior_mean = float(task_row["task_mean"]) if task_row is not None else global_mean
        prior_std = float(task_row["task_std"]) if task_row is not None else global_std

        effect = _shrink_mean(raw_mean, n, prior_mean, args.prior_strength)
        se = _shrink_se(raw_std, n, args.prior_strength, prior_std)
        ci = 1.96 * se

        pid = int(player_id)
        tid = int(team_id)
        task_int = int(task)
        player_name = player_names.get((tid, pid), "")
        team_name = team_names.get(tid, "")

        rows.append(
            dict(
                PlayerID=pid,
                TeamID=tid,
                Task=task_int,
                shots=n,
                raw_mean=raw_mean,
                raw_std=raw_std,
                prior_mean=prior_mean,
                prior_std=prior_std,
                effect_mean=effect,
                effect_lower=effect - ci,
                effect_upper=effect + ci,
                se=se,
                player_name=player_name,
                team_name=team_name,
            )
        )

    player_task_df = pd.DataFrame(rows)
    if not player_task_df.empty:
        player_task_df = player_task_df.sort_values(["Task", "effect_mean"], ascending=[True, False])

    out_pt = pathlib.Path(args.out_player_task)
    player_task_df.to_csv(out_pt, index=False)

    # Player-level summary: weighted mean across tasks + consistency
    summaries = []
    for (pid, tid), g in player_task_df.groupby(["PlayerID", "TeamID"]):
        total_shots = g["shots"].sum()
        weights = g["shots"] / total_shots
        agg_effect = float((g["effect_mean"] * weights).sum())
        consistency = float(df[(df["PlayerID"] == pid) & (df["TeamID"] == tid)]["metric"].std(ddof=1))
        summaries.append(
            dict(
                PlayerID=int(pid),
                TeamID=int(tid),
                player_name=player_names.get((int(tid), int(pid)), ""),
                team_name=team_names.get(int(tid), ""),
                total_shots=int(total_shots),
                tasks=len(g),
                avg_effect=agg_effect,
                consistency=consistency,
            )
        )

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("avg_effect", ascending=False)

    out_ps = pathlib.Path(args.out_player_summary)
    summary_df.to_csv(out_ps, index=False)

    print(f"[done] wrote {len(player_task_df)} player-task rows -> {out_pt}")
    print(f"[done] wrote {len(summary_df)} player summaries -> {out_ps}")


if __name__ == "__main__":
    main()
