#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

POS_MAX = 4095.0
NUM_STONES = 12

BUTTON_X_RAW = 750.0
BUTTON_Y_RAW = 800.0

HOGLINE_Y_RAW = 2900.0
BACKLINE_Y_RAW = 200.0
UNITS_PER_FOOT = (HOGLINE_Y_RAW - BACKLINE_Y_RAW) / 27.0  # ~100 units/ft
R_12_RAW = 6.0 * UNITS_PER_FOOT
STONE_R_RAW = 0.46 * UNITS_PER_FOOT
HOUSE_RADIUS_RAW = R_12_RAW + STONE_R_RAW


@dataclass
class SynthConfig:
    num_games: int = 50
    ends_per_game: int = 8
    seed: int = 123

    team_a_id: int = 19
    team_b_id: int = 27

    # 12-shot ends (6 stones each)
    shots_per_end: int = 12
    first_throw_team: str = "A"  # "A" or "B"

    # Random-ish shot placement
    out_of_play_prob: float = 0.10
    radial_scale_mult: float = 0.70
    jitter_std: float = 8.0

    out_stones: Path = Path("synth_stones.csv")
    out_ends: Path = Path("synth_ends.csv")


def _is_in_play(x: float, y: float) -> bool:
    if (x == 0.0 and y == 0.0):
        return False
    if x == POS_MAX or y == POS_MAX:
        return False
    return True


def _compute_points_final_board(board_flat: np.ndarray) -> Tuple[int, int]:
    """
    Simplified scoring from FINAL board:
      - stones 1–6 are Team A, 7–12 are Team B
      - count stones of winning side closer than opponent's closest, within house
    Returns (points_a, points_b) with opposite signs (or both 0).
    """
    coords = board_flat.reshape(NUM_STONES, 2).astype(np.float32)
    button = np.array([BUTTON_X_RAW, BUTTON_Y_RAW], dtype=np.float32)

    def dists(side_coords: np.ndarray):
        out = []
        for x, y in side_coords:
            if not _is_in_play(float(x), float(y)):
                continue
            dist = float(np.linalg.norm(np.array([x, y], dtype=np.float32) - button))
            if dist <= HOUSE_RADIUS_RAW:
                out.append(dist)
        return out

    da = dists(coords[:6])
    db = dists(coords[6:])

    if not da and not db:
        return 0, 0

    mina = min(da) if da else float("inf")
    minb = min(db) if db else float("inf")

    if np.isclose(mina, minb):
        return 0, 0

    if mina < minb:
        opp = minb
        pts = sum(1 for d in da if d < opp)
        return int(pts), int(-pts)
    else:
        opp = mina
        pts = sum(1 for d in db if d < opp)
        return int(-pts), int(pts)


def _sample_pos(rng: np.random.Generator, cfg: SynthConfig):
    if rng.random() < cfg.out_of_play_prob:
        return POS_MAX, POS_MAX
    angle = rng.uniform(0.0, 2.0 * np.pi)
    radius = rng.exponential(scale=R_12_RAW * cfg.radial_scale_mult)
    jitter = rng.normal(0.0, cfg.jitter_std, size=2)
    x = BUTTON_X_RAW + radius * np.cos(angle) + float(jitter[0])
    y = BUTTON_Y_RAW + radius * np.sin(angle) + float(jitter[1])
    x = float(np.clip(x, 0.0, POS_MAX))
    y = float(np.clip(y, 0.0, POS_MAX))
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_games", type=int, default=50)
    ap.add_argument("--ends_per_game", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--team_a_id", type=int, default=19)
    ap.add_argument("--team_b_id", type=int, default=27)
    ap.add_argument("--first_throw_team", type=str, default="A", choices=["A", "B"])
    ap.add_argument("--out_stones", type=str, default="synth_stones.csv")
    ap.add_argument("--out_ends", type=str, default="synth_ends.csv")

    ap.add_argument("--out_of_play_prob", type=float, default=0.10)
    ap.add_argument("--radial_scale_mult", type=float, default=0.70)
    ap.add_argument("--jitter_std", type=float, default=8.0)
    args = ap.parse_args()

    cfg = SynthConfig(
        num_games=args.num_games,
        ends_per_game=args.ends_per_game,
        seed=args.seed,
        team_a_id=args.team_a_id,
        team_b_id=args.team_b_id,
        first_throw_team=args.first_throw_team,
        out_of_play_prob=args.out_of_play_prob,
        radial_scale_mult=args.radial_scale_mult,
        jitter_std=args.jitter_std,
        out_stones=Path(args.out_stones),
        out_ends=Path(args.out_ends),
    )

    rng = np.random.default_rng(cfg.seed)

    stone_rows = []
    end_rows = []

    # Alternate shots: 12 shots total, 6 per team
    # Slots: Team A uses stones 1..6, Team B uses 7..12, in throw order.
    for game_id in range(cfg.num_games):
        for end_id in range(1, cfg.ends_per_game + 1):
            board = np.zeros((NUM_STONES, 2), dtype=np.float32)  # unthrown = (0,0)

            # Define alternation and hammer:
            # - If A throws first, B throws last (hammer), and vice-versa.
            first_team = cfg.team_a_id if cfg.first_throw_team == "A" else cfg.team_b_id
            other_team = cfg.team_b_id if first_team == cfg.team_a_id else cfg.team_a_id

            a_throw_count = 0
            b_throw_count = 0

            for shot_idx in range(cfg.shots_per_end):
                team_id = first_team if (shot_idx % 2 == 0) else other_team

                # Place this team’s next stone into its next slot
                x, y = _sample_pos(rng, cfg)

                if team_id == cfg.team_a_id:
                    if a_throw_count < 6:
                        board[a_throw_count, 0] = x
                        board[a_throw_count, 1] = y
                    a_throw_count += 1
                else:
                    if b_throw_count < 6:
                        board[6 + b_throw_count, 0] = x
                        board[6 + b_throw_count, 1] = y
                    b_throw_count += 1

                # Emit a Stones.csv row for this shot (shooter perspective)
                row = {
                    "CompetitionID": 0,
                    "SessionID": 0,
                    "GameID": game_id,
                    "EndID": end_id,
                    "ShotID": 7 + shot_idx,  # mimic your real ShotID range (optional)
                    "TeamID": team_id,
                    "PlayerID": 0,
                    "Task": int(rng.integers(0, 4)),
                    "Handle": int(rng.integers(0, 2)),
                    "Points": 0,
                    "TimeOut": "",
                }
                for i in range(NUM_STONES):
                    row[f"stone_{i+1}_x"] = float(board[i, 0]) if board[i, 0] != 0 else 0.0
                    row[f"stone_{i+1}_y"] = float(board[i, 1]) if board[i, 1] != 0 else 0.0
                stone_rows.append(row)

            # Ends.csv from FINAL board
            final_flat = board.reshape(-1)
            points_a, points_b = _compute_points_final_board(final_flat)

            end_rows.append(
                {
                    "CompetitionID": 0,
                    "SessionID": 0,
                    "GameID": game_id,
                    "TeamID": cfg.team_a_id,
                    "EndID": end_id,
                    "Result": int(max(points_a, 0)),
                    "PowerPlay": 0,
                }
            )
            end_rows.append(
                {
                    "CompetitionID": 0,
                    "SessionID": 0,
                    "GameID": game_id,
                    "TeamID": cfg.team_b_id,
                    "EndID": end_id,
                    "Result": int(max(points_b, 0)),
                    "PowerPlay": 0,
                }
            )

    stones_df = pd.DataFrame(stone_rows)
    ends_df = pd.DataFrame(end_rows)

    cfg.out_stones.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_ends.parent.mkdir(parents=True, exist_ok=True)

    stones_df.to_csv(cfg.out_stones, index=False)
    ends_df.to_csv(cfg.out_ends, index=False)

    print(f"Saved synth Stones: {cfg.out_stones} ({len(stones_df)} rows)")
    print(f"Saved synth Ends:   {cfg.out_ends} ({len(ends_df)} rows)")


if __name__ == "__main__":
    main()
