# Scoring the *Intended* Shot in Curling (Decision vs. Execution)

This repository implements a physics-grounded shot evaluation framework for curling that **separates**:

- **Decision-making (intention / plan):** what the team *chose* to try from a given position.
- **Execution:** how well the delivered stone matched that plan.
- **Risk / fragility:** how sensitive the outcome was to small execution errors.

The core idea is to evaluate each shot by comparing the observed outcome to **counterfactual outcomes** generated from the *same* pre-shot state via **physics-based Monte Carlo simulation**.

## Method Overview

### 1) `xScore`: expected end score differential for any board state
We learn a state-value model $xScore(s, c)$ that maps a static curling position (stone layout $s$ plus context $c$, e.g., shot index / hammer) to:

$$
\mathbb{E}\left[\text{end points (team)} - \text{end points (opponent)} \mid s, c\right].
$$

This provides a consistent value scale (curling’s analogue of expected points / xG-style models).

### 2) Reconstructing execution via simulation + inverse solving
Observed datasets typically provide state transitions $(s_t \rightarrow s_{t+1})$, not the release parameters that caused them.  
We infer each shot’s executed release vector by solving:

$$
\hat{x}_t = \underset{x}{\mathrm{arg\,min}} \;\; \mathcal{L}\!\left(\mathrm{Sim}(s_t, x),\; s_{t+1}\right)
$$

where $\mathrm{Sim}$ is a physics simulator and $\mathcal{L}$ is an identity-agnostic layout loss that handles collisions, removals, and stone identity ambiguity.
In practice, the inverse problem is nonconvex, so we use a population-based optimizer (e.g., CEM-style) rather than relying on local gradients alone.

### 3) Counterfactual Monte Carlo for execution (local) vs decision (global)
From the same pre-shot state $s_t$, we generate many simulated alternatives and score them with `xScore`:

- **Local sampling:** small perturbations around the inferred executed release $\hat{x}_t$.
  Estimates plan value under realistic execution noise and fragility.
- **Global sampling:** broad sampling across feasible releases.
  Approximates the opportunity set and supports decision-quality comparisons.

Each simulated throw produces:

$$
\Delta xScore(x) = xScore\!\left(\operatorname{Sim}(s_t, x), \; c_{t+1}\right) - xScore(s_t, c_t).
$$

## Per-shot Metrics (Coach-facing)

Let `ΔxScore_obs(t)` be the observed value swing from the dataset, and let the **local** Monte Carlo distribution yield a mean `μ_L` and a downside tail metric (e.g., `CVaR_10%`).

- **Plan Value (mean under noise):**  
  `μ_L` — “How good is this plan on average if we throw it again?”
- **Plan Risk / Fragility (downside tail):**  
  `CVaR_10%` — “How bad does it get when we miss a little?”
- **Execution Surplus (execution relative to the call):**  
  `ExecSurplus(t) = ΔxScore_obs(t) - μ_L`  
  Positive means the delivered result exceeded what the plan typically produces; negative means it underperformed.
- **Decision Regret (call quality vs alternatives):**  
  `DecisionRegret(t) = max(ΔxScore_global) - μ_L`  
  Small regret: near-optimal call even if execution failed. Large regret: the call likely left value on the table.

## Practical Notes / Limitations
- **Simulation fidelity matters.** Absolute calibration (ice, sweeping, friction/contact parameters) may be imperfect, but relative comparisons are still informative because counterfactuals are generated from the *same* state under the *same* simulator.
- **Inverse solutions may be multi-modal.** When shots are badly missed or affected by unmodeled factors, multiple releases can match the same observed transition; multi-start inversion cana## Overview

This repository implements the pipeline described in **“Scoring the Intended Shot: Separating Decision-making from Execution with Physics-based Monte Carlo Simulation for Curling”** (see `report/report.tex`). At a high level, the codebase:

1. Trains **xScore**, a state-value model that maps a curling layout + context to expected end score differential.
2. Uses a **physics simulator** + **inverse solver** to infer per-shot release parameters (speed/angle/spin/lateral offset) from observed state transitions.
3. Runs **Monte Carlo counterfactuals**:
   - **Local** sampling around the inferred throw to evaluate execution and fragility.
   - **Global** sampling across feasible throws to evaluate decision quality.
4. Produces per-shot metrics (decision value/risk, execution surplus) and aggregates them into player/team analyses and figures.

---

## Scripts (overview)

### Inverse + physics (execution reconstruction)

- `inverse/curling_sim_jax.py`: JAX forward simulator for curling shots. Given a pre-shot layout and release params `[speed, angle, spin, y0]`, rolls out stone dynamics (ice drag + curl + stable penalty contacts) and returns final settled positions or a full trajectory.

- `inverse/curling_inverse.py`: CEM-based inverse solver. Finds release params that best reproduce an observed transition using a hard, identity-agnostic loss that supports stones going off-sheet (targets are in-bounds only; unmatched in-bounds predictions / missing targets are penalized with boundary-aware weighting).

- `demo_infer.py`: End-to-end inversion demo with removals. Fabricates a next-state with at least one off-sheet stone, trims targets to in-bounds stones, runs coarse→refine CEM, and optionally renders a GIF.

- `make_bc_dataset.py` (aka `make_BC_data.py`): GPU-parallel parameter estimation over `Stones.csv`. Converts CSV states to meters, trims next-state targets to in-bounds stones, runs two-stage inverse solving per shot, and writes merged per-chunk CSVs of estimated release parameters and diagnostics.

### Value model (xScore)

- `valueModel/dataset.py`: PyTorch `Dataset` for xScore training. Loads `Stones.csv` + `Ends.csv`, builds per-shot inputs as flattened 12-stone coordinates (optionally normalized by `POS_MAX`), attaches context features `[shot_norm, is_hammer, team_order]`, and provides per-team end score differential labels (`ValueDiff`). Includes optional augmentation by shuffling stones within team blocks (1–6 vs 7–12) among “thrown” stones.

- `valueModel/model.py` (the `ValueTransformer` file shown): Transformer value network. Treats each stone as a token, prepends a context-conditioned global token, mixes with a Transformer encoder, and outputs a scalar state value per row/team perspective.

- `valueModel/synth.py`: Synthetic data generator for value-model pretraining/regularization. Produces toy `synth_stones.csv` and `synth_ends.csv` by sampling random final-board stone placements (with some out-of-play stones) and computing simplified end scoring from the final layout.

- `valueModel/train_with_synth_wandb.py`: W&B-enabled trainer for `ValueTransformer` using real + subsampled synthetic data. Splits validation from REAL only, concatenates real-train with a configurable synthetic subset (`--synth_frac`, `--synth_max`), logs metrics to W&B, and saves/logs the best checkpoint as an artifact.

### Main Analysis

- `score_shots_mc_seq.py`: Runs Monte Carlo shot scoring per CompetitionID. Writes per-competition score CSVs plus optional combined output and per-competition smoke tests.

- `fit_execution_noise.py`: Calibrates execution-noise parameters from inverse-solved throws, estimating std (and optional covariance) over `[speed, angle, spin, y0]` globally and by `(Task, Handle)` (optionally `(Player, Task)`). Emits `noise_config.json` used by the MC scorers.

- `visualize.py`: Single-shot Monte Carlo visualization: samples neighbors around the inverse center, simulates outcomes, scores with xScore, and plots prev/observed plus p10/p50/p90 boards with distribution summaries. Also exports a neighbors CSV of sampled params and resulting `ΔxScore`.

- `player_skill_model.py`: Aggregates per-shot MC outputs into player-by-task skill estimates using `excess = dv_obs − dv_mean` with empirical-Bayes shrinkage for stability. Writes `player_task_skill.csv` and a shot-weighted `player_summary.csv` (with optional hard-loss filtering).

- `make_coach_report.py`: Produces a coach-facing report from `shot_scores` + player skill tables, with seaborn figures for task/handle difficulty, top players, overall rankings, and Points-vs-metric relationships. Outputs `summary.md` and `figures/*.png`.

- `make_coach_report_mc.py`: Compares local vs global MC scorings to decompose decision quality vs execution: `decision_value = dv_mean_local − dv_mean_global`, `decision_risk = dv_std_local`, `execution_value = dv_obs − dv_mean_local`. Outputs a merged per-shot table plus report markdown and figures.
