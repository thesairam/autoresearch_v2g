# autoresearch — V2G Fleet Optimization

This is an autonomous research experiment: the agent discovers better strategies for
V2G (Vehicle-to-Grid) fleet scheduling by iterating on `strategy.py`, running backtests,
and keeping only improvements — indefinitely, without human intervention.

---

## Setup

To start a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may25`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files** — the entire repo is small; read all of them:
   - `README.md` — repository context.
   - `prepare_v2g.py` — **fixed and immutable**. Contains fleet constants, data pipeline, simulation engine, and the `evaluate_v2g()` evaluation harness. **Never modify this file.**
   - `strategy.py` — the **only** file you modify. Implements price forecasting and dispatch scheduling.
   - `program_v2g.md` — this file. All rules live here.
4. **Verify data is cached**: Check that `~/.cache/autoresearch_v2g/` exists and contains `prices_eur_mwh.npy`, `sessions.json`, `dates.json`. If not, tell the human to run: `uv run prepare_v2g.py`
5. **Initialize results_v2g.tsv**: Create it with just the header row (see Logging section). The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good, then kick off the experimentation loop.

---

## The Problem

A fleet of **30 EVs** at a workplace depot (15 with V2G discharge capability) must be
scheduled for charging and discharging across 24 hours each day.

The strategy receives:
- Hourly electricity spot prices for the past 30 days (from Nord Pool DK1)
- Historical vehicle session data for the past 30 days
- Today's vehicle sessions: arrival/departure times, initial SoC, battery specs

The strategy must return a charge/discharge schedule (kW per vehicle per hour) for
the full upcoming 24 hours. The schedule is then simulated against **actual** prices.

**What the agent optimizes:**

```
val_revenue_per_kwh = (V2G_revenue - charging_cost - degradation_penalty) / kWh_transacted
```

This is computed by the **fixed harness** (`evaluate_v2g()` in `prepare_v2g.py`) over the
pinned validation period (full year 2023, 365 days). Higher is better.

**Hard constraints** (enforced by the harness — you cannot relax these):
- SoC must stay within [10%, 95%] at all times
- Each vehicle must reach ≥80% SoC before departure (3× price penalty if violated)
- Max charge rate: 11 kW per vehicle (AC Type 2)
- Max discharge rate: 7.4 kW per V2G vehicle
- Total site grid limit: 300 kW

**Degradation** (fixed model — you cannot change this):
- Each kWh discharged via V2G costs 0.050 EUR (battery wear)

---

## Experimentation

**What you CAN do:**
- Modify `strategy.py` — this is the **only** file you edit. Everything in it is fair game:
  - Price forecasting model (any architecture: LSTM, Transformer, XGBoost, statistical, etc.)
  - Availability/session forecasting
  - Dispatch optimization method (rule-based, LP, MPC, RL, heuristic, etc.)
  - Feature engineering from price history and session history
  - How you handle uncertainty, SoC constraints, departure deadlines
  - Internal module state, helper functions, global variables

**What you CANNOT do:**
- **Never modify `prepare_v2g.py`**. It is permanently read-only. It contains the fixed
  fleet constants, simulation engine, and evaluation harness. Modifying it invalidates
  all comparisons and breaks the experiment.
- **Never modify the evaluation harness**. The `evaluate_v2g()` and `simulate_day()`
  functions in `prepare_v2g.py` are the ground truth. You cannot change degradation
  rates, constraint penalties, SoC bounds, or any other harness parameter.
- **Do not install new packages** or modify `pyproject.toml`. Use only what is already
  available: `numpy`, `requests`, `torch`, `pandas`, `matplotlib`, and stdlib.
- Do not modify `program_v2g.md`, `README.md`, or any file other than `strategy.py`.

**The goal is simple: maximize `val_revenue_per_kwh`.** Everything in `strategy.py` is
yours to change. Better price forecasting leads to better dispatch timing. Better dispatch
optimization extracts more value from the same forecasts. Both directions matter.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement
from 20 extra lines of complex code is not worth it. A simplification that maintains
equal or better performance is always a win. When deciding whether to keep a change,
weigh complexity cost against improvement magnitude.

**The first run**: Your very first run must be with the unmodified baseline `strategy.py`
to establish the baseline `val_revenue_per_kwh`.

---

## Running an Experiment

```bash
uv run prepare_v2g.py --eval > run.log 2>&1
```

This runs the full backtest on the **val period (2023, 365 days)** and prints structured
output. The run takes approximately 2–4 minutes depending on strategy complexity.

For a quick smoke test during development (7 days only):
```bash
uv run prepare_v2g.py --smoke > run.log 2>&1
```

**Timeout**: If a run exceeds **10 minutes**, kill it and treat it as a crash.

---

## Output Format

When the backtest finishes, `prepare_v2g.py` prints:

```
---
val_revenue_per_kwh:          0.023841     ← primary metric (higher is better)
total_net_revenue_eur:         4312.50
total_charged_kwh:             72450.1
total_discharged_kwh:          18320.4
total_constraint_violations:   12
num_eval_days:                 365
total_seconds:                 142.3
forecast_rmse_eur_mwh:         8.42        ← only shown if strategy exposes last_price_forecast
```

Extract the primary metric from the log:

```bash
grep "^val_revenue_per_kwh:" run.log
```

---

## Logging Results

Record each experiment in `results_v2g.tsv` (tab-separated, NOT comma-separated).
Do **not** commit this file — leave it untracked by git.

Header and columns:

```
commit	val_revenue_per_kwh	violations	status	description
```

1. git commit hash (short, 7 chars)
2. `val_revenue_per_kwh` (e.g. `0.023841`) — use `0.000000` for crashes
3. `total_constraint_violations` count — use `0` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried

Example:

```
commit	val_revenue_per_kwh	violations	status	description
a1b2c3d	0.023841	12	keep	baseline: persistence forecast + greedy dispatch
b2c3d4e	0.027312	8	keep	LP dispatch with scipy.optimize.linprog
c3d4e5f	0.019500	45	discard	RL policy (SAC) — high violations, needs more tuning
d4e5f6g	0.000000	0	crash	LSTM price model — missing torch.nn import
```

---

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/may25`).

**LOOP FOREVER:**

1. Check git state: current branch and commit.
2. Edit `strategy.py` with an experimental idea.
3. `git commit`
4. Run the backtest: `uv run prepare_v2g.py --eval > run.log 2>&1`
5. Extract results: `grep "^val_revenue_per_kwh:\|^total_constraint_violations:" run.log`
6. If grep output is empty → the run crashed. Run `tail -n 50 run.log` to see the traceback. Attempt a fix if it's trivial. If the idea is fundamentally broken, skip it.
7. Record results in `results_v2g.tsv`.
8. If `val_revenue_per_kwh` improved (**higher**), advance the branch (keep the commit).
9. If equal or worse, `git reset --hard HEAD~1` to discard the change.

---

## Crash Handling

- **Simple crash** (typo, missing variable, import error): fix it, re-run on the same commit.
- **Fundamentally broken idea** (e.g. LP infeasible by design, OOM, infinite loop): log as `crash`, reset, move on.
- **Constraint explosion** (violations >> baseline): the strategy is probably not honoring departure SoC requirements. Fix the safety pass in dispatch or discard.

---

## Research Directions

Ideas the agent should explore (not exhaustive):

**Forecasting improvements:**
- Replace persistence with rolling mean, ETS, or ARIMA for price forecasting
- Add calendar features (hour-of-day, day-of-week, month) for price patterns
- Use a Transformer or LSTM trained on price sequences
- Probabilistic forecasting (predict price quantiles, plan conservatively)
- Cluster historical days by price profile, use nearest-neighbor forecast

**Dispatch improvements:**
- Replace greedy rules with Linear Programming (`scipy.optimize.linprog` or `cvxpy`)
- Model Predictive Control: solve LP at each hour with rolling forecast
- Multi-vehicle coordination (minimize peak grid demand while maximizing revenue)
- Risk-aware dispatch: if forecast uncertainty is high, charge conservatively first
- Separate fast-charge pass (meet SoC target) from V2G pass (maximize revenue)

**Combined improvements:**
- Train a simple neural price forecaster + LP dispatch
- Predict session arrival/departure distributions and use in dispatch planning
- Per-vehicle personalization based on historical session patterns

---

## NEVER STOP

Once the loop begins, **do NOT pause to ask the human if you should continue**. Do not
ask "should I keep going?" or "is this a good stopping point?". The human may be asleep
or away from the computer and expects indefinite autonomous operation. You are a
researcher. If you run out of ideas, think harder — revisit near-misses, combine
approaches, try more radical changes. The loop runs until the human interrupts you.
