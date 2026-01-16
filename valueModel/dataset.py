import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Constants
POS_MAX = 4095.0  # sentinel and upper bound
MAX_ENDS = 8
NUM_STONES = 12


def _compute_end_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - ShotIndex, ShotsInEnd, shot_norm
      - is_hammer (team that throws last in end)
      - team_order (0=throws first in end, 1=other team)
    All computed within (CompetitionID,SessionID,GameID,EndID).
    """
    group_cols = ["CompetitionID", "SessionID", "GameID", "EndID"]
    df = df.sort_values(group_cols + ["ShotID"]).reset_index(drop=True)

    # ShotIndex / shot_norm
    df["ShotIndex"] = df.groupby(group_cols).cumcount()
    df["ShotsInEnd"] = df.groupby(group_cols)["ShotID"].transform("count")
    df["shot_norm"] = 0.0
    mask = df["ShotsInEnd"] > 1
    df.loc[mask, "shot_norm"] = df.loc[mask, "ShotIndex"] / (df.loc[mask, "ShotsInEnd"] - 1.0)

    # First/last team per end (based on ShotID ordering)
    first_team = df.groupby(group_cols)["TeamID"].transform("first")
    last_team = df.groupby(group_cols)["TeamID"].transform("last")

    df["is_hammer"] = (df["TeamID"] == last_team).astype(np.float32)
    df["team_order"] = (df["TeamID"] != first_team).astype(np.float32)  # first=0, other=1

    return df


class ValueDataset(Dataset):
    """
    Builds (state, condition) -> value samples from Stones.csv and Ends.csv.

    Label:
      y = Result_team - Result_opponent in that end
        (implemented as 2*Result - sum(Result) over both teams)

    Condition c (cond_dim=3):
      c = [shot_norm, is_hammer, team_order]
        - shot_norm: 0..1 within the end (by ShotID order)
        - is_hammer: 1 if this TeamID throws last in the end
        - team_order: 0 if this TeamID throws first in end, else 1

    Augmentation:
      - shuffles stones within each team block (1–6 vs 7–12),
        but only among stones that are actually thrown (coords < POS_MAX).
    """

    def __init__(
        self,
        stones_csv_path,
        ends_csv_path,
        normalize=True,
        max_ends=MAX_ENDS,
        min_shots_per_end=1,
        augment_positions=True,
    ):
        self.stones_csv_path = stones_csv_path
        self.ends_csv_path = ends_csv_path
        self.normalize = normalize
        self.max_ends = max_ends
        self.augment_positions = augment_positions

        # -------- Load Stones --------
        df_s = pd.read_csv(stones_csv_path)

        # Stone position columns
        self.stone_cols = []
        for i in range(1, NUM_STONES + 1):
            self.stone_cols.append(f"stone_{i}_x")
            self.stone_cols.append(f"stone_{i}_y")

        stones_critical = [
            "CompetitionID",
            "SessionID",
            "GameID",
            "EndID",
            "ShotID",
            "TeamID",
            "Task",
            "Handle",
        ] + self.stone_cols
        missing_stones_crit = [c for c in stones_critical if c not in df_s.columns]
        if missing_stones_crit:
            raise ValueError(f"Stones CSV is missing columns: {missing_stones_crit}")

        # Drop rows with NaNs in critical columns (zeros and 4095 are fine)
        df_s = df_s.dropna(subset=stones_critical).reset_index(drop=True)

        # Compute per-end context features from Stones ordering
        df_s = _compute_end_context(df_s)

        # Optional filtering for max_ends / min_shots_per_end if you want later
        # (kept as-is; not enforced here to avoid surprising drops)

        # -------- Load Ends --------
        df_e = pd.read_csv(ends_csv_path)

        ends_critical = [
            "CompetitionID",
            "SessionID",
            "GameID",
            "TeamID",
            "EndID",
            "Result",
            "PowerPlay",
        ]
        missing_ends_crit = [c for c in ends_critical if c not in df_e.columns]
        if missing_ends_crit:
            raise ValueError(f"Ends CSV is missing columns: {missing_ends_crit}")

        df_e = df_e.dropna(subset=["Result"]).reset_index(drop=True)
        df_e["Result"] = df_e["Result"].astype(float)

        # -------- Convert Result -> score differential per team --------
        merge_keys = ["CompetitionID", "SessionID", "GameID", "EndID", "TeamID"]
        end_keys_no_team = ["CompetitionID", "SessionID", "GameID", "EndID"]

        df_e["TotalResultInEnd"] = df_e.groupby(end_keys_no_team)["Result"].transform("sum")
        df_e["ValueDiff"] = 2.0 * df_e["Result"] - df_e["TotalResultInEnd"]

        # -------- Merge Stones with Ends (attach differential value) --------
        df = pd.merge(
            df_s,
            df_e[merge_keys + ["ValueDiff"]],
            on=merge_keys,
            how="inner",
        )

        df = df.sort_values(
            ["CompetitionID", "SessionID", "GameID", "EndID", "ShotID"]
        ).reset_index(drop=True)

        # Determine number of tasks (from Stones)
        self.num_tasks = int(df["Task"].max()) + 1 if len(df) else 0

        # Regression target
        df["value_target"] = df["ValueDiff"].astype(float)

        self.df = df.reset_index(drop=True)
        self.pos_dim = NUM_STONES * 2

        # Condition is [shot_norm, is_hammer, team_order]
        self.cond_dim = 3
        self.input_dim = self.pos_dim
        self.output_dim = 1  # scalar value

    def __len__(self):
        return len(self.df)

    def _augment_positions(self, raw_vals: np.ndarray) -> np.ndarray:
        mat = raw_vals.reshape(NUM_STONES, 2).copy()

        # Team A: stones 1–6; Team B: stones 7–12 (dataset convention)
        for start in (0, 6):
            idxs = np.arange(start, start + 6)
            coords = mat[idxs]

            # "thrown" if at least one coord < POS_MAX
            thrown_mask = np.any(coords < POS_MAX, axis=1)
            thrown_idxs = idxs[thrown_mask]

            if len(thrown_idxs) > 1:
                shuffled_local = np.random.permutation(len(thrown_idxs))
                original_vals = mat[thrown_idxs].copy()
                mat[thrown_idxs] = original_vals[shuffled_local]

        return mat.reshape(-1)

    def _extract_positions(self, row: pd.Series) -> np.ndarray:
        raw_vals = row[self.stone_cols].to_numpy(dtype=np.float32)

        if self.augment_positions:
            raw_vals = self._augment_positions(raw_vals)

        if self.normalize:
            return (raw_vals / POS_MAX).astype(np.float32)
        return raw_vals.astype(np.float32)

    def _make_condition(self, row: pd.Series) -> np.ndarray:
        shot_norm = float(row["shot_norm"])
        is_hammer = float(row["is_hammer"])
        team_order = float(row["team_order"])
        return np.array([shot_norm, is_hammer, team_order], dtype=np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        x = self._extract_positions(row)       # (24,)
        c = self._make_condition(row)          # (3,)
        y = np.array([row["value_target"]], dtype=np.float32)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(c).float(),
            torch.from_numpy(y).float(),
        )


def denormalize_positions(pos_vec, normalize=True):
    arr = np.asarray(pos_vec, dtype=np.float32)
    if normalize:
        arr = arr * POS_MAX
    return arr


def positions_to_matrix(pos_vec):
    arr = np.asarray(pos_vec, dtype=np.float32)
    assert arr.size == NUM_STONES * 2
    return arr.reshape(NUM_STONES, 2)
