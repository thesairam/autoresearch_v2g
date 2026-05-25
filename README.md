# autoresearch_v2g

Autonomous AI research for **V2G (Vehicle-to-Grid) fleet optimization**.

An AI agent iterates on a charge/discharge scheduling strategy overnight —
forecasting electricity prices, optimizing fleet dispatch, and keeping only
improvements. You wake up to a log of experiments and a smarter strategy.

> **Open source — PRs welcome.** See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## How it works

A fleet of **30 EVs** at a European workplace depot (15 V2G-capable) must be
scheduled across each 24-hour day to maximize revenue from energy arbitrage
and grid services while keeping every vehicle charged for departure.

The agent autonomously:
1. Modifies `strategy.py` with an experimental idea
2. Runs a full-year backtest against real 2023 market data (~2–4 min)
3. If `val_revenue_per_kwh` improved → keep; otherwise → revert
4. Repeat forever — ~15–20 experiments/hour

### The metric

```
val_revenue_per_kwh = (V2G revenue + arbitrage savings − battery degradation) / kWh_transacted
```

Evaluated on a **fixed 2023 validation period (365 days)** using real Nord Pool prices.
Higher is better. The baseline (greedy rule) starts at ~−0.076 EUR/kWh.
A good strategy reaches positive values by timing charges to cheap/negative-price
hours and discharges to high-priced or regulation-market hours.

---

## Three files that matter

| File | Role | Who edits |
|------|------|-----------|
| `prepare_v2g.py` | Data pipeline + evaluation harness | **Never** — fixed ground truth |
| `strategy.py` | Price forecast + dispatch policy | **Agent only** |
| `program_v2g.md` | All rules + research directions | **Human only** |

---

## Data sources

All primary sources are **free and open** — no API key required.

### Electricity prices — Energidataservice (Danish TSO, official)
- **Day-ahead spot prices**: Nord Pool DK1, hourly EUR/MWh
- **Balancing market prices**: Up/down-regulation prices DK1 (via ENTSO-E if token available)
- Source: [api.energidataservice.dk](https://api.energidataservice.dk) — operated by Energinet
- 2019–2024 coverage (2,193 days). No authentication required.

### EV fleet — realistic 2024-era European models
Synthetic sessions calibrated from [ACN-Data](https://ev.caltech.edu/dataset) statistics:

| Model | Share | Battery | Max charge | V2G discharge |
|-------|-------|---------|-----------|--------------|
| Hyundai IONIQ 5 | 20% | 77.4 kWh | 11 kW | 3.6 kW (ISO 15118-20) |
| Nissan Leaf e+ | 15% | 62.0 kWh | 6.6 kW | 3.3 kW (CHAdeMO) |
| Renault Megane E-Tech | 15% | 60.0 kWh | 11 kW | 3.7 kW |
| Volkswagen ID.4 | 20% | 77.0 kWh | 11 kW | — |
| Tesla Model 3 LR | 15% | 75.0 kWh | 11 kW | — |
| Renault Zoe | 15% | 52.0 kWh | 22 kW | — |

### Battery degradation model — NMC Li-ion empirical
Based on Schmalstieg et al. (2014) and Wang et al. (2011):
- Base rate: **0.040 EUR/kWh** discharged at 50% SoC (sweet spot)
- SoC stress multiplier: rises to **0.092 EUR/kWh** at 95% SoC (lithium plating penalty)
- Incentivizes 20–80% SoC operation range — consistent with real fleet contracts

### Optional enhanced data (set env vars to enable)
```bash
# ENTSO-E: broader European coverage + balancing market prices
# Free token at https://transparency.entsoe.eu/usrm/user/createPublicUser
export ENTSOE_TOKEN=your_token_here

# Real ACN-Data EV sessions (Caltech), free after registration
# Register at https://ev.caltech.edu
export ACN_TOKEN=your_token_here
```

---

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/). No GPU required.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download data — one-time, ~1 minute
uv run prepare_v2g.py

# Smoke test (7 days)
uv run prepare_v2g.py --smoke

# Full baseline evaluation (365 days)
uv run prepare_v2g.py --eval
```

## Running the agent

```
Have a look at program_v2g.md and let's kick off a new V2G experiment!
```

The agent will create a branch, establish the baseline, then loop forever.

## Visualization

Open `analysis.ipynb` to explore price data, fleet session patterns,
experiment progress, and strategy dispatch deep-dives.

---

## Available tools for strategy.py

The agent can use any of these — all are pre-installed via `pyproject.toml`:

| Tool | Use case |
|------|----------|
| `cvxpy` | LP/QP optimal dispatch (gold standard, used in FlexMeasures) |
| `scipy.optimize` | Lighter LP, good for fast experiments |
| `stable-baselines3` | PPO, SAC, TD3 RL policies (EV2Gym approach) |
| `torch` | Neural forecasters — LSTMs, Transformers |
| `scikit-learn` | Gradient boosting, clustering, regression forecasters |
| `numpy / pandas` | Feature engineering, rule-based heuristics |

---

## Related open-source projects

| Project | What it does | What to borrow |
|---------|-------------|----------------|
| [EV2Gym](https://github.com/StavrosOrf/EV2Gym) | OpenAI Gym for V2G with RL (NeurIPS 2024) | RL formulation, SAC/PPO dispatch, EV specs |
| [FlexMeasures](https://github.com/FlexMeasures/flexmeasures) | Energy flexibility scheduling platform | Probabilistic forecasting, robust LP, ENTSO-E integration |
| [PyBaMM](https://github.com/pybamm-team/PyBaMM) | Physics-based battery modelling | Detailed electrochemical degradation |
| [CVXPY](https://www.cvxpy.org/examples/index.html) | Convex optimization in Python | LP/QP dispatch formulations |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | RL algorithms | SAC, PPO, TD3 policy training |
| [acnportal](https://github.com/zach401/acnportal) | Caltech ACN-Data client | Real EV session data |

---

## Key papers

1. **Kempton & Tomić (2005)** — "Vehicle-to-grid power fundamentals." *J. Power Sources.* — Foundational V2G economics.
2. **Sortomme & El-Sharkawi (2012)** — "Optimal Scheduling of Vehicle-to-Grid Energy." *IEEE Trans. Smart Grid.* — LP dispatch formulation.
3. **Schmalstieg et al. (2014)** — "A Holistic Aging Model for Li(NiMnCo)O2." *J. Power Sources.* — Degradation model used in this harness.
4. **Orfanoudakis et al. (2024)** — "EV2Gym: A Flexible V2G Simulator." — RL-based V2G benchmark.
5. **Elmachtoub & Grigas (2022)** — "Smart Predict, then Optimize." *Management Science.* — End-to-end forecast + dispatch training.

---

## Project structure

```
prepare_v2g.py   — data pipeline + evaluation harness (never modify)
strategy.py      — forecast + dispatch strategy (agent modifies this)
program_v2g.md   — all agent rules and research directions
analysis.ipynb   — visualization: prices, fleet, experiments, dispatch
pyproject.toml   — dependencies (cvxpy, stable-baselines3, torch, scipy, ...)
CONTRIBUTING.md  — how to contribute
LICENSE          — MIT
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs for new strategies, data sources,
degradation models, and additional price zones are especially welcome.

## License

MIT — see [LICENSE](LICENSE).
