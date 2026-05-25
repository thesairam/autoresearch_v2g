# autoresearch_v2g

Autonomous AI research for **V2G (Vehicle-to-Grid) fleet optimization**.

An AI agent iterates on a charge/discharge scheduling strategy overnight —
forecasting electricity prices, optimizing fleet dispatch, and keeping only
improvements. You wake up to a log of experiments and a smarter strategy.

---

## What it does

A fleet of **30 EVs** at a workplace depot (15 with V2G bidirectional capability)
must be scheduled for charging and discharging across each 24-hour day.

The agent autonomously:
1. Modifies `strategy.py` with an experimental idea (better forecasting, smarter dispatch)
2. Runs a full-year backtest against real Nord Pool DK1 electricity prices (2023)
3. If `val_revenue_per_kwh` improved → keep the change
4. If not → revert and try something else
5. Repeat forever

You can run ~15–20 experiments/hour. Leave it running overnight and wake up to
100+ experiments, a better strategy, and a full results log.

## The metric

```
val_revenue_per_kwh = (V2G_revenue - charging_cost - battery_degradation) / kWh_transacted
```

Evaluated on a **fixed validation period (2023, 365 days)** using real Nord Pool DK1 spot prices.
Higher is better. The baseline (greedy rule) starts at ~−0.076 EUR/kWh —
the agent's job is to get this toward positive by smarter price forecasting and dispatch.

---

## How it works

Three files that matter:

| File | Role | Editable by |
|------|------|-------------|
| `prepare_v2g.py` | Data pipeline + evaluation harness | **Never** — fixed ground truth |
| `strategy.py` | Price forecast + dispatch policy | **Agent only** |
| `program_v2g.md` | Agent instructions and all rules | **Human only** |

**Data sources:**
- **Electricity prices**: Nord Pool DK1 day-ahead spot prices (Energidataservice API, free)
- **EV fleet sessions**: Synthetic workplace depot sessions based on real ACN-Data statistics

**Fixed constraints (enforced by harness):**
- Battery SoC: [10%, 95%]
- Vehicle must reach ≥80% SoC before departure (3× penalty if violated)
- Max charge: 11 kW per vehicle | Max V2G discharge: 7.4 kW
- Site grid limit: 300 kW

---

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), internet connection for data download.
No GPU required — backtesting runs on CPU in 1–3 minutes.

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download price data + generate fleet sessions (one-time, ~1 min)
uv run prepare_v2g.py

# 4. Verify the pipeline works (7-day smoke test)
uv run prepare_v2g.py --smoke

# 5. Run a full baseline evaluation (365 days, ~2 min)
uv run prepare_v2g.py --eval
```

---

## Running the agent

Spin up Claude Code (or any LLM agent) in this repo and start with:

```
Have a look at program_v2g.md and let's kick off a new V2G experiment! Let's do the setup first.
```

The agent will:
1. Create a branch `autoresearch/<tag>`
2. Read all in-scope files
3. Run the baseline
4. Loop forever: modify → backtest → keep or revert → repeat

**Do not interrupt it.** It's autonomous.

---

## Experiment loop (agent runs this)

```bash
# One experiment:
git commit -m "try LP dispatch with scipy"
uv run prepare_v2g.py --eval > run.log 2>&1
grep "^val_revenue_per_kwh:" run.log
# → keep if improved, git reset if not
```

Results logged to `results_v2g.tsv`:
```
commit   val_revenue_per_kwh   violations   status   description
a1b2c3d  -0.075565             34           keep     baseline: persistence + greedy
b2c3d4e  -0.041230             18           keep     LP dispatch with scipy.optimize
c3d4e5f  -0.063000             22           discard  LSTM price model (overfit)
```

---

## Visualization

Open `analysis.ipynb` to explore:
- Nord Pool DK1 price history and seasonal patterns
- Fleet session arrival/departure distributions
- Experiment progress (val_revenue_per_kwh over time)
- Strategy dispatch deep-dive

---

## Project structure

```
prepare_v2g.py     — data pipeline + evaluation harness (never modify)
strategy.py        — forecast + dispatch strategy (agent modifies this)
program_v2g.md     — agent instructions and all rules
pyproject.toml     — dependencies (numpy, scipy, scikit-learn, torch, ...)
analysis.ipynb     — visualization and result analysis
results_v2g.tsv    — experiment log (untracked by git, created during run)
```

---

## License

MIT
