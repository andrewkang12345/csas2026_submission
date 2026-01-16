#!/usr/bin/env python3
"""
coach_report_seaborn.py

Coach-facing report from shot_scores + player skill estimates, with professional seaborn figures.

Adds:
  1) Uses seaborn styling + cleaner plots.
  2) Analyses "Handle" (spin) analogously to Task:
       - handle difficulty summary
       - top players per handle
  3) Outputs a PNG ranking of all players (across all tasks) using player_task_skill.csv
  4) Merges Stones.csv to pull the discrete human "Points" (0–4) label, then:
       - computes correlation between Points and dv metrics (dv_obs, excess=dv_obs-dv_mean)
       - visualizes the relationship (box/violin + trend)

Inputs:
  - shot_scores.csv OR shot_scores.parquet (robust reader)
  - player_task_skill.csv (from your skill fit script)
  - Stones.csv (to get Points; can also serve as source for Handle/Task if needed)

Outputs:
  - <out_dir>/summary.md
  - <out_dir>/figures/*.png

Usage:
  python coach_report_seaborn.py \
    --shot-scores shot_scores.csv \
    --player-task-skill player_task_skill.csv \
    --stones-csv 2026/Stones.csv \
    --competitors-csv 2026/Competitors.csv \
    --teams-csv 2026/Teams.csv \
    --competitions-csv 2026/competitions.csv \
    --games-csv 2026/games.csv \
    --out-dir coach_report
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


SHOT_KEY = ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]

TASK_NAME: Dict[int, str] = {
    0: "Draw",
    1: "Front",
    2: "Guard",
    3: "Raise / Tap-back",
    4: "Wick / Soft Peeling",
    5: "Freeze",
    6: "Take-out",
    7: "Hit and Roll",
    8: "Clearing",
    9: "Double Take-out",
    10: "Promotion Take-out",
    11: "Through",
    13: "No statistics",
}

# Handle labels (spin). Adjust if you have canonical naming.
HANDLE_NAME: Dict[int, str] = {
    0: "Handle 0",
    1: "Handle 1",
    -1: "Unknown",
}


# ----------------------------
# Robust shot_scores loader
# ----------------------------
def _looks_like_parquet(p: pathlib.Path) -> bool:
    try:
        with p.open("rb") as f:
            head = f.read(4)
        return head == b"PAR1"
    except Exception:
        return False


def _load_table(path: str) -> pd.DataFrame:
    p = pathlib.Path(path)
    suf = p.suffix.lower()

    if suf == ".csv":
        return pd.read_csv(p)

    if suf == ".parquet":
        if not _looks_like_parquet(p):
            return pd.read_csv(p)
        try:
            return pd.read_parquet(p)
        except Exception as e:
            print(f"[warn] parquet read failed; falling back to CSV for {p}. error={e}")
            return pd.read_csv(p)

    return pd.read_csv(p)


def _ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Name resolution helpers
# ----------------------------
def _load_player_names(competitors_csv: str) -> Dict[Tuple[int, int], str]:
    """
    Competitors.csv lacks PlayerID; infer PlayerID within team by order of appearance.
    Returns: (TeamID, PlayerID) -> Reportingname
    """
    p = pathlib.Path(competitors_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "TeamID" not in df.columns or "Reportingname" not in df.columns:
        return {}

    df = df.copy()
    df["player_ord"] = df.groupby("TeamID").cumcount() + 1
    return {
        (int(r.TeamID), int(r.player_ord)): str(r.Reportingname)
        for _, r in df.iterrows()
        if pd.notna(r.TeamID) and pd.notna(r.player_ord)
    }


def _load_team_names(teams_csv: str) -> Dict[int, str]:
    p = pathlib.Path(teams_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "TeamID" not in df.columns:
        return {}
    name_col = "Name" if "Name" in df.columns else None
    if name_col is None:
        return {int(r.TeamID): f"Team {int(r.TeamID)}" for _, r in df.iterrows() if pd.notna(r.TeamID)}
    return {int(r.TeamID): str(r[name_col]) for _, r in df.iterrows() if pd.notna(r.TeamID)}


def _load_competition_names(competitions_csv: str) -> Dict[int, str]:
    p = pathlib.Path(competitions_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "CompetitionID" not in df.columns:
        return {}
    name_col = "CompetitionName" if "CompetitionName" in df.columns else None
    if name_col is None:
        return {int(r.CompetitionID): f"Competition {int(r.CompetitionID)}" for _, r in df.iterrows() if pd.notna(r.CompetitionID)}
    return {int(r.CompetitionID): str(r[name_col]) for _, r in df.iterrows() if pd.notna(r.CompetitionID)}


def _load_game_labels(games_csv: str) -> Dict[Tuple[int, int, int], str]:
    p = pathlib.Path(games_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    needed = {"CompetitionID", "SessionID", "GameID"}
    if not needed.issubset(set(df.columns)):
        return {}

    noc1 = "NOC1" if "NOC1" in df.columns else None
    noc2 = "NOC2" if "NOC2" in df.columns else None
    sheet = "Sheet" if "Sheet" in df.columns else None

    out: Dict[Tuple[int, int, int], str] = {}
    for _, r in df.iterrows():
        if pd.isna(r.CompetitionID) or pd.isna(r.SessionID) or pd.isna(r.GameID):
            continue
        c, s, g = int(r.CompetitionID), int(r.SessionID), int(r.GameID)
        a = str(r[noc1]) if noc1 else "NOC1"
        b = str(r[noc2]) if noc2 else "NOC2"
        lbl = f"{a} vs {b}"
        if sheet and pd.notna(r[sheet]):
            lbl += f" (Sheet {str(r[sheet])})"
        out[(c, s, g)] = lbl
    return out


def _task_label(x) -> str:
    try:
        t = int(x)
        return TASK_NAME.get(t, f"Task {t}")
    except Exception:
        return "Task (unknown)"


def _handle_label(x) -> str:
    try:
        h = int(x)
        return HANDLE_NAME.get(h, f"Handle {h}")
    except Exception:
        return "Handle (unknown)"


# ----------------------------
# Styling
# ----------------------------
def _set_plot_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ----------------------------
# Plots
# ----------------------------
def plot_task_difficulty(task_stats: pd.DataFrame, out_path: pathlib.Path):
    df = task_stats.copy()
    df["TaskLabel"] = df["Task"].apply(_task_label)

    fig, ax = plt.subplots(figsize=(10, 6))

    # scatter (no hue / no legend)
    sizes = np.clip(df["count"].to_numpy(dtype=float), 40, 600)
    sns.scatterplot(
        data=df,
        x="dv_mean",
        y="dv_std",
        size="count",
        sizes=(40, 600),
        legend=False,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
        ax=ax,
    )

    # initial text objects
    texts = []
    for _, r in df.iterrows():
        texts.append(
            ax.text(
                float(r["dv_mean"]),
                float(r["dv_std"]),
                str(r["TaskLabel"]),
                fontsize=10,
                ha="left",
                va="bottom",
            )
        )

    # repel / de-overlap
    try:
        from adjustText import adjust_text  # type: ignore

        adjust_text(
            texts,
            ax=ax,
            # pull labels slightly away from points and allow moderate movement
            expand_points=(1.2, 1.2),
            expand_text=(1.2, 1.2),
            force_points=0.35,
            force_text=0.7,
            lim=200,
            arrowprops=dict(arrowstyle="-", lw=0.8, alpha=0.6),
        )
    except Exception as e:
        # fallback if adjustText is not installed: nudge by a fixed offset
        print(f"[warn] adjustText not available ({e}); using fixed offsets for labels.")
        for t in texts:
            x, y = t.get_position()
            t.set_position((x + 0.01, y + 0.01))

    ax.set_xlabel("Expected impact (dv_mean)")
    ax.set_ylabel("Variability (dv_std)")
    ax.set


def plot_handle_difficulty(handle_stats: pd.DataFrame, out_path: pathlib.Path):
    """
    Handle difficulty analog of task difficulty.
    With only a few handles, a clean point/range view is better than a scatter landscape.
    """
    df = handle_stats.copy()
    df["HandleLabel"] = df["Handle"].apply(_handle_label)

    fig, ax = plt.subplots(figsize=(10, 5))
    # Use a point plot with error bars: dv_mean +/- dv_std
    df = df.sort_values("count", ascending=False)
    ax.errorbar(
        x=df["HandleLabel"].astype(str),
        y=df["dv_mean"].to_numpy(dtype=float),
        yerr=df["dv_std"].to_numpy(dtype=float),
        fmt="o",
        capsize=6,
        alpha=0.95,
    )
    ax.set_xlabel("Handle (spin)")
    ax.set_ylabel("Expected impact (dv_mean) ± dv_std")
    ax.set_title("Handle difficulty (mean ± variability)")
    # annotate counts
    for i, r in enumerate(df.itertuples(index=False)):
        ax.annotate(f"n={int(r.count)}", (i, float(r.dv_mean)), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_top_players_by_task(player_task_df: pd.DataFrame, out_dir: pathlib.Path, top_k: int = 8):
    df = player_task_df.copy()
    df["TaskLabel"] = df["Task"].apply(_task_label)

    if "player_name" not in df.columns:
        df["player_name"] = ""
    if "team_name" not in df.columns:
        df["team_name"] = ""

    df["PlayerLabel"] = df.apply(
        lambda r: (
            f"{r['player_name']} ({r['team_name']})".strip()
            if isinstance(r.get("player_name", ""), str) and r.get("player_name", "")
            else f"Player {int(r['PlayerID'])} ({r.get('team_name','')})".strip()
        ),
        axis=1,
    )

    for task, g in df.groupby("Task", dropna=True):
        g = g.sort_values("effect_mean", ascending=False).head(int(top_k)).copy()
        if g.empty:
            continue
        # sort for barh
        g = g.sort_values("effect_mean", ascending=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        ci = (g["se"].fillna(0.0).to_numpy(dtype=float) * 1.96)
        ax.barh(g["PlayerLabel"].astype(str), g["effect_mean"].to_numpy(dtype=float), xerr=ci, alpha=0.92)
        ax.axvline(0.0, linewidth=1.0, alpha=0.35)
        ax.set_xlabel("Estimated skill (excess ΔV), with 95% CI")
        ax.set_title(f"Top players — {_task_label(task)}")
        fig.tight_layout()
        fig.savefig(out_dir / f"top_players_task_{int(task):02d}.png")
        plt.close(fig)


def plot_top_players_by_handle(player_task_df: pd.DataFrame, out_dir: pathlib.Path, top_k: int = 10):
    """
    Your player_task_skill table is grouped by Task, not Handle.
    So for handle analysis, we instead derive player rankings per handle from shot_scores (dv metric),
    unless player_task_df already contains Handle (it typically does not).

    This function expects the caller to pass a dataframe that DOES contain Handle-based effects if desired.
    We keep this plot for completeness; the main handle-player plot is made from shots_df below.
    """
    # Placeholder if you later produce handle-conditioned skill tables.
    pass


def plot_player_ranking_all(player_task_df: pd.DataFrame, out_path: pathlib.Path, top_n: int = 30):
    """
    Ranking across all tasks: weighted average of effect_mean with weights=shots.
    (Matches your earlier "player_summary" logic, but plotted as a professional PNG.)
    """
    df = player_task_df.copy()
    if "player_name" not in df.columns:
        df["player_name"] = ""
    if "team_name" not in df.columns:
        df["team_name"] = ""

    g = df.groupby(["PlayerID", "TeamID"], dropna=True).apply(
        lambda x: pd.Series(
            {
                "total_shots": float(x["shots"].sum()),
                "avg_effect": float((x["effect_mean"] * (x["shots"] / max(1.0, x["shots"].sum()))).sum()),
                "player_name": x["player_name"].iloc[0] if "player_name" in x.columns else "",
                "team_name": x["team_name"].iloc[0] if "team_name" in x.columns else "",
            }
        )
    ).reset_index()

    g["PlayerLabel"] = g.apply(
        lambda r: (
            f"{r['player_name']} ({r['team_name']})".strip()
            if isinstance(r.get("player_name", ""), str) and r.get("player_name", "")
            else f"Player {int(r['PlayerID'])} ({r.get('team_name','')})".strip()
        ),
        axis=1,
    )

    g = g.sort_values("avg_effect", ascending=False).head(int(top_n)).copy()
    # sort for barh
    g = g.sort_values("avg_effect", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=g, x="avg_effect", y="PlayerLabel", ax=ax, orient="h", alpha=0.95)
    ax.axvline(0.0, linewidth=1.0, alpha=0.35)
    ax.set_xlabel("Overall estimated skill (excess ΔV per shot)")
    ax.set_ylabel("")
    ax.set_title(f"Overall player ranking (top {int(top_n)})")
    # annotate shot counts
    for i, r in enumerate(g.itertuples(index=False)):
        ax.text(float(r.avg_effect), i, f"  n={int(r.total_shots)}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_points_vs_dv(shots_df: pd.DataFrame, out_path: pathlib.Path):
    """
    Visualize relationship between human Points (0-4) and dv metrics:
      - dv_obs
      - excess = dv_obs - dv_mean
    Outputs one figure with two panels + correlation annotations.
    """
    df = shots_df.copy()
    if "Points" not in df.columns:
        print("[warn] Stones Points not present after merge; skipping points-vs-dv plots.")
        return

    # Keep rows with points and dv
    df = df.dropna(subset=["Points", "dv_obs", "dv_mean"]).copy()
    if df.empty:
        print("[warn] no rows with Points + dv; skipping points-vs-dv plots.")
        return

    df["Points"] = df["Points"].round().astype(int)
    df["excess"] = df["dv_obs"].astype(float) - df["dv_mean"].astype(float)

    # Compute correlations (Pearson + Spearman) for both dv_obs and excess
    def _corr(x: pd.Series, y: pd.Series):
        pear = float(x.corr(y, method="pearson"))
        spear = float(x.corr(y, method="spearman"))
        return pear, spear

    pear_obs, spear_obs = _corr(df["Points"], df["dv_obs"])
    pear_ex, spear_ex = _corr(df["Points"], df["excess"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # Left: dv_obs by Points (violin + box overlay)
    sns.violinplot(data=df, x="Points", y="dv_obs", inner=None, cut=0, ax=axes[0])
    sns.boxplot(data=df, x="Points", y="dv_obs", width=0.28, showfliers=False, ax=axes[0])
    axes[0].set_title("dv_obs vs human Points (0–4)")
    axes[0].set_xlabel("Points (human label)")
    axes[0].set_ylabel("dv_obs")
    axes[0].text(
        0.02, 0.98,
        f"Pearson={pear_obs:.3f}\nSpearman={spear_obs:.3f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )

    # Right: excess by Points + trend of per-point mean
    sns.violinplot(data=df, x="Points", y="excess", inner=None, cut=0, ax=axes[1])
    sns.boxplot(data=df, x="Points", y="excess", width=0.28, showfliers=False, ax=axes[1])

    # Overlay per-point mean with CI
    means = df.groupby("Points")["excess"].mean().reset_index()
    sns.pointplot(data=means, x="Points", y="excess", ax=axes[1], color="black", markers="D", linestyles="-", errorbar=None)

    axes[1].set_title("Excess (dv_obs - dv_mean) vs Points")
    axes[1].set_xlabel("Points (human label)")
    axes[1].set_ylabel("excess ΔV")
    axes[1].text(
        0.02, 0.98,
        f"Pearson={pear_ex:.3f}\nSpearman={spear_ex:.3f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )

    # fig.suptitle("Human-labeled shot quality vs model-based ΔV metrics", y=1.02, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_handle_player_ranking_from_shots(shots_df: pd.DataFrame, out_path: pathlib.Path, top_n: int = 25):
    """
    Handle-level player ranking derived directly from shot_scores (not from player_task_skill).
    Metric: mean excess (dv_obs - dv_mean), filtered to finite rows.
    """
    if "Handle" not in shots_df.columns:
        return

    df = shots_df.dropna(subset=["Handle", "dv_obs", "dv_mean", "PlayerID", "TeamID"]).copy()
    if df.empty:
        return
    df["HandleLabel"] = df["Handle"].apply(_handle_label)
    df["excess"] = df["dv_obs"].astype(float) - df["dv_mean"].astype(float)

    # Aggregate across handles + players; then rank players overall but show handle breakdown is separate plot below.
    # Here: produce a facet plot? Keep it simple: one plot per handle, top_n per handle.
    handles = sorted(df["Handle"].dropna().unique().tolist())

    nrows = len(handles)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 5 * max(1, nrows)))
    if nrows == 1:
        axes = [axes]

    for ax, h in zip(axes, handles):
        sub = df[df["Handle"] == h].copy()
        agg = (
            sub.groupby(["PlayerID", "TeamID", "player_name", "team_name"], dropna=False)
            .agg(avg_excess=("excess", "mean"), shots=("excess", "count"))
            .reset_index()
        )
        agg = agg.sort_values("avg_excess", ascending=False).head(int(top_n)).copy()
        agg["PlayerLabel"] = agg.apply(
            lambda r: (
                f"{r['player_name']} ({r['team_name']})".strip()
                if isinstance(r.get("player_name", ""), str) and r.get("player_name", "")
                else f"Player {int(r['PlayerID'])} ({r.get('team_name','')})".strip()
            ),
            axis=1,
        )
        agg = agg.sort_values("avg_excess", ascending=True)

        sns.barplot(data=agg, x="avg_excess", y="PlayerLabel", ax=ax, orient="h", alpha=0.95)
        ax.axvline(0.0, linewidth=1.0, alpha=0.35)
        ax.set_title(f"Top players by handle — {_handle_label(h)} (mean excess ΔV)")
        ax.set_xlabel("Mean excess ΔV")
        ax.set_ylabel("")
        for i, r in enumerate(agg.itertuples(index=False)):
            ax.text(float(r.avg_excess), i, f"  n={int(r.shots)}", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_summary_md(
    out_dir: pathlib.Path,
    shots_df: pd.DataFrame,
    task_stats: pd.DataFrame,
    handle_stats: pd.DataFrame,
    player_task_df: pd.DataFrame,
) -> None:
    def _to_md(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_string(index=False)

    lines: list[str] = []
    lines.append("# Coach Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Shots scored: {len(shots_df)}")
    lines.append(f"- Average dv_obs: {float(shots_df['dv_obs'].mean()):.3f}")
    lines.append(f"- Average percentile_obs: {float(shots_df['percentile_obs'].mean()):.3f}" if "percentile_obs" in shots_df.columns else "- Average percentile_obs: (missing)")
    if "Points" in shots_df.columns:
        pts = shots_df["Points"].dropna()
        if len(pts) > 0:
            lines.append(f"- Points coverage: {len(pts)}/{len(shots_df)} ({len(pts)/len(shots_df):.1%})")
    lines.append("")

    lines.append("## Figures")
    lines.append("- Task difficulty: figures/task_difficulty.png")
    lines.append("- Handle (spin) difficulty: figures/handle_difficulty.png")
    lines.append("- Top players per task: figures/top_players_task_XX.png")
    lines.append("- Overall player ranking: figures/player_ranking_all.png")
    lines.append("- Human Points vs dv metrics: figures/points_vs_dv.png")
    lines.append("- Top players per handle (from shot_scores): figures/top_players_by_handle.png")
    lines.append("")

    lines.append("## Task difficulty snapshot (top by volume)")
    snap = task_stats.copy()
    snap["Task"] = snap["Task"].apply(_task_label)
    snap = snap.sort_values("count", ascending=False)
    lines.append(_to_md(snap[["Task", "dv_mean", "dv_std", "count"]].head(12)))
    lines.append("")

    lines.append("## Handle (spin) snapshot")
    hsnap = handle_stats.copy()
    hsnap["Handle"] = hsnap["Handle"].apply(_handle_label)
    hsnap = hsnap.sort_values("count", ascending=False)
    lines.append(_to_md(hsnap[["Handle", "dv_mean", "dv_std", "count"]]))
    lines.append("")

    lines.append("## Player strengths (per task; top 3)")
    df = player_task_df.copy()
    if "player_name" not in df.columns:
        df["player_name"] = ""
    if "team_name" not in df.columns:
        df["team_name"] = ""
    for task, g in df.groupby("Task", dropna=True):
        g = g.sort_values("effect_mean", ascending=False).head(3)
        if g.empty:
            continue
        names = []
        for _, r in g.iterrows():
            pname = r.get("player_name", "")
            if not isinstance(pname, str) or not pname:
                pname = f"Player {int(r['PlayerID'])}"
            tname = r.get("team_name", "")
            label = f"{pname}" + (f" ({tname})" if isinstance(tname, str) and tname else "")
            names.append(f"{label}: {float(r['effect_mean']):.3f}")
        lines.append(f"- {_task_label(task)}: " + "; ".join(names))

    (out_dir / "summary.md").write_text("\n".join(lines))


# ----------------------------
# Main
# ----------------------------
def main():
    _set_plot_style()

    ap = argparse.ArgumentParser(description="Create coach-facing report from shot_scores + player skill tables (seaborn).")
    ap.add_argument("--shot-scores", type=str, default="shot_scores_old.csv", help="shot_scores.parquet or shot_scores.csv")
    ap.add_argument("--player-task-skill", type=str, default="player_task_skill.csv")

    # Needed to pull Points (0-4) and to ensure Handle/Task are present even if shot_scores lacks them
    ap.add_argument("--stones-csv", type=str, default="2026/Stones.csv", help="Stones.csv containing Points/Handle/Task")

    # Optional label sources
    ap.add_argument("--competitors-csv", type=str, default="2026/Competitors.csv")
    ap.add_argument("--teams-csv", type=str, default="2026/Teams.csv")
    ap.add_argument("--competitions-csv", type=str, default="2026/competitions.csv")
    ap.add_argument("--games-csv", type=str, default="2026/games.csv")

    ap.add_argument("--out-dir", type=str, default="coach_report")
    ap.add_argument("--top-k-task", type=int, default=8)
    ap.add_argument("--top-n-overall", type=int, default=30)
    ap.add_argument("--top-n-handle", type=int, default=25)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    fig_dir = out_dir / "figures"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    shots_df = _load_table(args.shot_scores)

    # Load Stones for Points (and as fallback for Task/Handle if needed)
    stones_p = pathlib.Path(args.stones_csv)
    if not stones_p.exists():
        raise SystemExit(f"stones-csv not found: {stones_p}")
    stones_df = pd.read_csv(stones_p)

    # Validate Stones required keys/cols for merge
    for c in SHOT_KEY:
        if c not in stones_df.columns:
            raise SystemExit(f"Stones.csv missing required key column: {c}")
    # We expect these in Stones for enrichment:
    for c in ["TeamID", "PlayerID", "Task", "Handle", "Points"]:
        if c not in stones_df.columns:
            print(f"[warn] Stones.csv missing column {c}; some report elements may be unavailable.")

    # Merge Points into shot_scores (left join keeps existing shot_scores rows)
    enrich_cols = [c for c in (SHOT_KEY + ["Task", "Handle", "Points", "TeamID", "PlayerID"]) if c in stones_df.columns]
    stones_small = stones_df[enrich_cols].copy()
    shots_df = shots_df.merge(stones_small, on=SHOT_KEY, how="left", suffixes=("", "_stones"))

    # If shot_scores lacks Task/Handle/TeamID/PlayerID but Stones has them, fill from *_stones
    for col in ["Task", "Handle", "Points", "TeamID", "PlayerID"]:
        alt = f"{col}_stones"
        if col not in shots_df.columns and alt in shots_df.columns:
            shots_df[col] = shots_df[alt]
        elif col in shots_df.columns and alt in shots_df.columns:
            shots_df[col] = shots_df[col].where(shots_df[col].notna(), shots_df[alt])

    # Clean up any *_stones columns
    drop_cols = [c for c in shots_df.columns if c.endswith("_stones")]
    if drop_cols:
        shots_df = shots_df.drop(columns=drop_cols)

    # Enforce human Points domain: only keep 0–4, drop everything else
    if "Points" in shots_df.columns:
        # Coerce to numeric (handles strings like "3", bad tokens -> NaN)
        shots_df["Points"] = pd.to_numeric(shots_df["Points"], errors="coerce")

        # Keep only valid human labels (0..4); everything else becomes missing
        shots_df.loc[~shots_df["Points"].between(0, 4, inclusive="both"), "Points"] = np.nan
    
    # Required columns for report logic
    required = ["dv_obs", "dv_mean", "dv_std", "Task"]
    missing = [c for c in required if c not in shots_df.columns]
    if missing:
        raise SystemExit(f"shot scores table missing columns: {missing}")

    # Load player skill table
    player_task_df = pd.read_csv(args.player_task_skill)

    # Load names
    player_names = _load_player_names(args.competitors_csv)
    team_names = _load_team_names(args.teams_csv)
    comp_names = _load_competition_names(args.competitions_csv)
    game_labels = _load_game_labels(args.games_csv)

    # Enrich shots with readable labels
    shots_df = shots_df.copy()
    shots_df["task_name"] = shots_df["Task"].apply(_task_label)
    if "Handle" in shots_df.columns:
        shots_df["handle_name"] = shots_df["Handle"].apply(_handle_label)
    else:
        shots_df["handle_name"] = "Handle (missing)"

    if "TeamID" in shots_df.columns:
        shots_df["team_name"] = shots_df["TeamID"].apply(lambda x: team_names.get(int(x), f"Team {int(x)}") if pd.notna(x) else "")
    else:
        shots_df["team_name"] = ""

    if "TeamID" in shots_df.columns and "PlayerID" in shots_df.columns:
        def _pname(row) -> str:
            try:
                tid = int(row["TeamID"])
                pid = int(row["PlayerID"])
                nm = player_names.get((tid, pid), "")
                return nm if nm else f"Player {pid}"
            except Exception:
                return ""
        shots_df["player_name"] = shots_df.apply(_pname, axis=1)
    else:
        shots_df["player_name"] = ""

    if "CompetitionID" in shots_df.columns:
        shots_df["competition_name"] = shots_df["CompetitionID"].apply(
            lambda x: comp_names.get(int(x), f"Competition {int(x)}") if pd.notna(x) else ""
        )
    else:
        shots_df["competition_name"] = ""

    if {"CompetitionID", "SessionID", "GameID"}.issubset(set(shots_df.columns)):
        def _glabel(row) -> str:
            try:
                k = (int(row["CompetitionID"]), int(row["SessionID"]), int(row["GameID"]))
                return game_labels.get(k, f"Game {k[1]}-{k[2]}")
            except Exception:
                return ""
        shots_df["game_label"] = shots_df.apply(_glabel, axis=1)
    else:
        shots_df["game_label"] = ""

    # Enrich player_task_df labels
    player_task_df = player_task_df.copy()
    if "team_name" not in player_task_df.columns and "TeamID" in player_task_df.columns:
        player_task_df["team_name"] = player_task_df["TeamID"].apply(
            lambda x: team_names.get(int(x), f"Team {int(x)}") if pd.notna(x) else ""
        )
    if "player_name" not in player_task_df.columns and {"TeamID", "PlayerID"}.issubset(set(player_task_df.columns)):
        def _pname2(row) -> str:
            try:
                tid = int(row["TeamID"])
                pid = int(row["PlayerID"])
                nm = player_names.get((tid, pid), "")
                return nm if nm else f"Player {pid}"
            except Exception:
                return ""
        player_task_df["player_name"] = player_task_df.apply(_pname2, axis=1)

    # Task difficulty stats
    task_stats = (
        shots_df.groupby("Task", dropna=True)
        .agg(
            dv_mean=("dv_mean", "mean"),
            dv_std=("dv_std", "mean"),
            count=("Task", "count"),
        )
        .reset_index()
    )

    # Handle difficulty stats
    if "Handle" in shots_df.columns:
        handle_stats = (
            shots_df.groupby("Handle", dropna=True)
            .agg(
                dv_mean=("dv_mean", "mean"),
                dv_std=("dv_std", "mean"),
                count=("Handle", "count"),
            )
            .reset_index()
        )
    else:
        handle_stats = pd.DataFrame({"Handle": [], "dv_mean": [], "dv_std": [], "count": []})

    # Produce figures
    plot_task_difficulty(task_stats, fig_dir / "task_difficulty.png")
    plot_handle_difficulty(handle_stats, fig_dir / "handle_difficulty.png")
    plot_top_players_by_task(player_task_df, fig_dir, top_k=int(args.top_k_task))
    plot_player_ranking_all(player_task_df, fig_dir / "player_ranking_all.png", top_n=int(args.top_n_overall))
    plot_points_vs_dv(shots_df, fig_dir / "points_vs_dv.png")
    plot_handle_player_ranking_from_shots(shots_df, fig_dir / "top_players_by_handle.png", top_n=int(args.top_n_handle))

    # Summary markdown
    build_summary_md(out_dir, shots_df, task_stats, handle_stats, player_task_df)

    print(f"[done] report written to {out_dir}")
    print(f"[done] figures written to {fig_dir}")


if __name__ == "__main__":
    main()
