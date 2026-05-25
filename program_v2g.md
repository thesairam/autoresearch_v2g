# autoresearch — V2G Fleet Optimization

This is an autonomous research experiment: the agent discovers better strategies for
V2G (Vehicle-to-Grid) fleet scheduling by iterating on `strategy.py`, running backtests,
and keeping only improvements — indefinitely, without human intervention.

---

## Setup

Work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may25`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read all in-scope files**:
   - `README.md` — project context, data sources, metric definition.
   - `prepare_v2g.py` — **fixed and immutable**. Fleet constants, data pipeline, battery degradation model, simulation engine, `evaluate_v2g()`. Never modify this file.
   - `strategy.py` — the **only** file you modify.
   - `program_v2g.md` — this file. All rules live here.
4. **Verify data is cached**: Check that `~/.cache/autoresearch_v2g/` exists with `prices_eur_mwh.npy`, `sessions.json`, `dates.json`. If not: `uv run prepare_v2g.py`
5. **Initialize results_v2g.tsv**: Create with just the header row (see Logging).
6. **Confirm and go**.

---

## The Problem

A fleet of **30 EVs** at a European workplace depot must be scheduled for charging
and V2G discharging across each 24-hour day.

**Fleet mix** (2024-era models, fixed in harness):
- Hyundai IONIQ 5 (V2G, 77.4 kWh, 3.6 kW discharge)
- Nissan Leaf e+ (V2G via CHAdeMO, 62 kWh, 3.3 kW discharge)
- Renault Megane E-Tech (V2G, 60 kWh, 3.7 kW discharge)
- VW ID.4, Tesla Model 3 LR, Renault Zoe (charge-only)

**Data available to strategy** (passed by harness on each day):
- Spot price history: Nord Pool DK1 day-ahead prices EUR/MWh (30-day window)
- Balancing market history: up/down regulation prices EUR/MWh (30-day window)
- Session history: which vehicles arrived/departed and when (30 days)
- Today's vehicle states: arrival hour, departure hour, SoC on arrival, battery specs

**What the strategy must return**: A 24-hour charge/discharge schedule (kW per vehicle).

---

## Metric

```
val_revenue_per_kwh = (V2G_revenue + arbitrage_savings − degradation) / kWh_transacted
```

Computed by the **fixed harness** over the pinned 2023 validation period (365 days).
Higher is better. Baseline: ~−0.076 EUR/kWh. A good strategy reaches positive values.

**Revenue model** (in `simulate_day`, harness):
- **Discharging**: paid at `max(spot_price, balancing_up_price)` per kWh exported
- **Charging**: costs `min(spot_price, balancing_down_price)` per kWh consumed
- **Degradation**: SoC-stress-weighted per kWh discharged (NMC model, fixed in harness)
  - Base rate ~0.040 EUR/kWh at 50% SoC sweet spot
  - Rises to ~0.092 EUR/kWh at 95% SoC (high-SoC lithium plating penalty)
  - Incentive: keep SoC in 20–80% range to minimize degradation
- **Departure violation**: 3× peak-price penalty per kWh short of 80% SoC at departure

---

## Rules

**You CAN modify** `strategy.py` — everything in it is fair game:
- Price and availability forecasting models (any architecture)
- Feature engineering from the history windows
- Dispatch optimization method (rule-based, LP, MPC, RL, anything)
- Internal state, helper functions, global variables, imports

**You CANNOT**:
- **Never modify `prepare_v2g.py`** — permanently read-only. It is the fixed evaluation ground truth. Modifying it invalidates all experiment comparisons.
- **Never modify `program_v2g.md`**, `README.md`, or any file other than `strategy.py`.
- **Do not modify the degradation model** in `prepare_v2g.py`. It is the fixed battery cost model.
- **Do not install new packages** beyond what is in `pyproject.toml`. All major tools are already available: `numpy`, `scipy`, `cvxpy`, `torch`, `stable-baselines3`, `scikit-learn`, `pandas`.

**Simplicity criterion**: A small improvement from adding 30 lines of complex code is not worth it. A simplification that maintains equal or better performance is always a win.

**First run**: Run the unmodified baseline to establish `val_revenue_per_kwh`.

---

## Running an Experiment

```bash
uv run prepare_v2g.py --eval > run.log 2>&1      # full val backtest (~2-4 min)
uv run prepare_v2g.py --smoke > run.log 2>&1     # 7-day smoke test (fast, for debugging)
```

**Timeout**: Kill any run exceeding **10 minutes**. Treat as crash.

---

## Output Format

```
---
val_revenue_per_kwh:          0.023841     ← primary metric (higher is better)
total_net_revenue_eur:         4312.50
total_charged_kwh:             72450.1
total_discharged_kwh:          18320.4
total_constraint_violations:   12
num_eval_days:                 365
total_seconds:                 142.3
forecast_rmse_eur_mwh:         8.42        ← only shown if last_price_forecast is set
```

Parse: `grep "^val_revenue_per_kwh:" run.log`

---

## Logging Results

`results_v2g.tsv` — tab-separated, **not** git-tracked.

```
commit	val_revenue_per_kwh	violations	status	description
```

- `status`: `keep`, `discard`, or `crash`
- Use `0.000000` and `0` for crashes

---

## The Experiment Loop

**LOOP FOREVER:**

1. Check git state (branch and current commit).
2. Edit `strategy.py`.
3. `git commit`
4. `uv run prepare_v2g.py --eval > run.log 2>&1`
5. `grep "^val_revenue_per_kwh:\|^total_constraint_violations:" run.log`
6. Empty output → crash. Check `tail -n 50 run.log`. Fix if trivial, skip if not.
7. Log to `results_v2g.tsv`.
8. If `val_revenue_per_kwh` improved (**higher**): keep the commit.
9. If same or worse: `git reset --hard HEAD~1`.

---

## Research Directions

Draw on the following established approaches. These are not exhaustive — be creative.

### Tier 1: High-impact, proven approaches

**LP/QP dispatch (gold standard in V2G literature)**

The optimal dispatch given a price forecast is a convex optimization problem.
Use `cvxpy` to formulate it exactly:

```python
import cvxpy as cp

def lp_dispatch(vehicle_states, price_forecast, bal_up_forecast, bal_dn_forecast):
    """Solve the day-ahead V2G scheduling LP for all vehicles.

    For each vehicle i, hour h:
      charge[i,h] >= 0, discharge[i,h] >= 0
      soc[i,h+1] = soc[i,h] + charge[i,h]*eff/cap - discharge[i,h]/(eff*cap)
      soc ∈ [SOC_MIN, SOC_MAX]
      soc[dep] >= 0.80 (departure constraint)

    Objective: maximize Σ_h (discharge[i,h]*sell_price[h] - charge[i,h]*buy_price[h])
               minus degradation proxy
    """
    ...
```

Reference: Sortomme & El-Sharkawi (2012), "Optimal Charging Strategies for Unidirectional
Vehicle-to-Grid." IEEE Trans. Smart Grid.

**Model Predictive Control (MPC)**

Re-solve the LP at each hour using updated price forecasts, rolling forward:
- Hour 0: plan 24h, execute hour 0
- Hour 1: replan remaining 23h with any new information, execute hour 1
- More robust to forecast errors than plan-once dispatch

Reference: Halvgaard et al. (2012), "Electric Vehicle Charge Planning using Economic MPC."

**Price forecasting → better dispatch**

Any improvement in price forecasting directly improves dispatch decisions.
The current baseline (persistence, RMSE ~36 EUR/MWh) leaves a lot on the table.

Approaches (from best in practice):
- **SARIMA / ETS**: 15–25 EUR/MWh RMSE on DK1 day-ahead
- **Gradient Boosting (XGBoost/LightGBM)**: 10–18 EUR/MWh RMSE; use features:
  hour-of-day, day-of-week, month, lagged prices (t-24h, t-48h, t-168h)
- **Transformer / LSTM**: 8–15 EUR/MWh RMSE with sufficient training data
- **Day-type clustering**: cluster past days by price profile shape, use nearest-neighbor

Reference: Hong & Fan (2016), "Probabilistic Electric Load Forecasting." Int'l Journal of Forecasting.

### Tier 2: Moderate complexity, worth trying

**Reinforcement Learning (EV2Gym approach)**

EV2Gym (StavrosOrf/EV2Gym, NeurIPS 2024) frames V2G as an RL problem:
- State: current SoC per vehicle, hour of day, price history, remaining session time
- Action: charge/discharge power per vehicle (continuous)
- Reward: revenue − degradation − violation penalty
- Algorithm: SAC or PPO (both in `stable-baselines3`)

To implement: wrap the backtest loop as a gym environment, train offline on
train-set days, evaluate on val-set days. The harness's `simulate_day()` can
be reused as the step function.

Reference: Orfanoudakis et al. (2024), "EV2Gym: A Flexible V2G Simulator for EV Smart Charging."

**Uncertainty-aware scheduling (FlexMeasures approach)**

FlexMeasures (SeitaBV/flexmeasures) uses probabilistic forecasts to build
robust schedules:
- Produce a distribution of price forecasts (quantiles or samples)
- Solve a robust or stochastic LP: guarantee departure SoC even in worst-case scenario
- Accept lower expected revenue in exchange for fewer violations

This is valuable when the penalty for violations is high (it is — 3× peak price).

**Battery-aware dispatch**

The harness penalizes high-SoC operation via the SoC stress factor in `degradation_eur()`.
Exploit this explicitly:
- Reserve high-SoC capacity for rare very-high-price events only
- Use shallow cycles (20–60% SoC range) for routine arbitrage
- Track cumulative degradation per vehicle and throttle V2G for high-mileage units

Reference: Schmalstieg et al. (2014), "A Holistic Aging Model for Li(NiMnCo)O2."
Journal of Power Sources.

### Tier 3: Advanced combinations

**Joint forecast + dispatch training (end-to-end)**

Train a differentiable LP layer (cvxpylayers) where the upstream forecasting
network and downstream dispatch optimizer are trained jointly to minimize regret,
not just forecast error. This directly optimizes the business metric.

Reference: Elmachtoub & Grigas (2022), "Smart Predict, then Optimize." Management Science.

**Multi-market participation**

The harness passes both spot and balancing market prices. A sophisticated strategy
can decide, per hour, whether to participate in the spot market or the regulation
market (whichever pays more):
- Discharge hours where `bal_up > spot`: participate in up-regulation
- Charge hours where `bal_dn < spot`: participate in down-regulation
- Track capacity commitments (you can't offer more than physically available)

**Session arrival forecasting**

The harness reveals session data at the start of the day (simplified day-ahead
assumption). But you can also model session uncertainty:
- Predict arrival probability per vehicle using historical patterns
- Use robust dispatch that handles both "vehicle arrives" and "vehicle doesn't arrive"
- This reduces violations on days with unexpected low attendance

---

## Key References

Papers the agent should consider when proposing experiments:

1. **Kempton & Tomić (2005)** — "Vehicle-to-grid power fundamentals." J. Power Sources.
   *The foundational V2G paper. Read if you haven't — defines the core economics.*

2. **Yilmaz & Krein (2013)** — "Review of Battery Charger Topologies, Charging Power Levels,
   and Infrastructure for Plug-In Electric and Hybrid Vehicles." IEEE Trans. Power Electronics.
   *Technical specs for charge/discharge rates used in fleet models.*

3. **Sortomme & El-Sharkawi (2012)** — "Optimal Scheduling of Vehicle-to-Grid Energy and
   Ancillary Services." IEEE Trans. Smart Grid.
   *LP formulation of V2G dispatch — the template for cvxpy-based strategies.*

4. **Schmalstieg et al. (2014)** — "A Holistic Aging Model for Li(NiMnCo)O2."
   Journal of Power Sources. *The SoC-stress degradation model used in this harness.*

5. **Orfanoudakis et al. (2024)** — "EV2Gym: A Flexible V2G Simulator."
   *Reference RL environment; architecture applicable to this problem.*

---

## Open Source Reference Implementations

| Project | What to borrow |
|---------|---------------|
| [EV2Gym](https://github.com/StavrosOrf/EV2Gym) | RL environment structure, SAC/PPO dispatch, EV specs |
| [FlexMeasures](https://github.com/FlexMeasures/flexmeasures) | Probabilistic forecasting, robust LP scheduling, ENTSO-E integration |
| [PyBaMM](https://github.com/pybamm-team/PyBaMM) | Electrochemical battery degradation (detailed physics, heavy) |
| [CVXPY examples](https://www.cvxpy.org/examples/index.html) | LP/QP dispatch formulations |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | SAC, PPO, TD3 implementations for RL dispatch |

---

## NEVER STOP

Once the loop begins, **do NOT pause to ask the human if you should continue**.
You are autonomous. If you run out of ideas: re-read the papers above, revisit
near-misses, try combining approaches, try more radical changes. The loop runs
until the human interrupts you.
