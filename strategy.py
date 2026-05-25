"""
V2G fleet scheduling strategy — the ONLY file the agent may modify.

Required functions:
    fit(train_prices, train_sessions, *, train_bal_up, train_bal_dn) -> None
    plan_day(date, price_history, session_history, vehicle_states,
             *, bal_up_history, bal_dn_history) -> np.ndarray

Optional attribute:
    last_price_forecast: np.ndarray | None  # shape (24,) — logged as diagnostic

Metric: val_revenue_per_kwh (higher is better)
  = (V2G revenue via up-regulation + energy arbitrage
     - charging cost via down-regulation
     - SoC-stress-weighted battery degradation
     - departure violation penalties)
  / total kWh transacted, over 365 days of the 2023 val period.

Baseline strategy:
  - Price forecast: persistence (yesterday's spot prices)
  - Dispatch: greedy — charge in cheapest 8h, discharge in 4 most expensive
  - Uses balancing prices to pick the better market (spot vs regulation)
"""

import datetime
import numpy as np

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_train_prices: np.ndarray | None = None
last_price_forecast: np.ndarray | None = None


# ---------------------------------------------------------------------------
# fit — called once before the backtest begins
# ---------------------------------------------------------------------------

def fit(
    train_prices: np.ndarray,
    train_sessions: list,
    *,
    train_bal_up: np.ndarray | None = None,
    train_bal_dn: np.ndarray | None = None,
) -> None:
    """Store training data; fit any models here."""
    global _train_prices
    _train_prices = train_prices.copy()


# ---------------------------------------------------------------------------
# plan_day — called once per evaluation day
# ---------------------------------------------------------------------------

def plan_day(
    date: datetime.date,
    price_history: np.ndarray,     # (H, 24) spot prices EUR/MWh
    session_history: list,
    vehicle_states: list,
    *,
    bal_up_history: np.ndarray | None = None,  # (H, 24) up-regulation prices
    bal_dn_history: np.ndarray | None = None,  # (H, 24) down-regulation prices
) -> np.ndarray:
    """Return a (N_vehicles, 24) schedule in kW.

    Positive = charging from grid.
    Negative = discharging to grid (V2G — only for vehicles with max_discharge_kw > 0).
    """
    global last_price_forecast

    n = len(vehicle_states)
    if n == 0:
        last_price_forecast = None
        return np.zeros((0, 24))

    # --- Persistence price forecast: yesterday's spot ---
    price_forecast = price_history[-1].copy().astype(np.float64) if len(price_history) > 0 \
                     else np.full(24, 50.0)
    last_price_forecast = price_forecast

    # Effective sell price per hour: max of spot and up-regulation forecast
    if bal_up_history is not None and len(bal_up_history) > 0:
        bal_up_fc = bal_up_history[-1].astype(np.float64)
    else:
        bal_up_fc = price_forecast * 1.10

    # Effective buy price per hour: min of spot and down-regulation forecast
    if bal_dn_history is not None and len(bal_dn_history) > 0:
        bal_dn_fc = bal_dn_history[-1].astype(np.float64)
    else:
        bal_dn_fc = price_forecast * 0.90

    sell_fc = np.maximum(price_forecast, bal_up_fc)   # best price to discharge
    buy_fc  = np.minimum(price_forecast, bal_dn_fc)   # cheapest price to charge

    # Rank hours for charging (cheapest buy price) and discharging (highest sell price)
    hour_rank_cheap     = np.argsort(buy_fc)           # cheapest → most expensive
    hour_rank_expensive = np.argsort(sell_fc)[::-1]    # most expensive first

    cheap_hours     = set(hour_rank_cheap[:8].tolist())      # 8h for charging
    expensive_hours = set(hour_rank_expensive[:4].tolist())  # 4h for V2G discharge

    schedule = np.zeros((n, 24), dtype=np.float64)

    for i, sess in enumerate(vehicle_states):
        arr    = int(sess["arrival_hour"])
        dep    = int(sess["departure_hour"])
        soc    = float(sess["soc_arrival"])
        cap    = float(sess["battery_capacity_kwh"])
        max_ch = float(sess["max_charge_kw"])
        max_ds = float(sess["max_discharge_kw"])

        plugged = list(range(arr, dep))
        soc_sim = soc

        for h in plugged:
            if h in expensive_hours and max_ds > 0:
                # Only discharge if we have headroom above the departure target
                if soc_sim > 0.85:
                    schedule[i, h] = -max_ds
                    soc_sim = max(soc_sim - max_ds / 0.92 / cap, 0.10)
            elif h in cheap_hours:
                if soc_sim < 0.95:
                    schedule[i, h] = max_ch
                    soc_sim = min(soc_sim + max_ch * 0.92 / cap, 0.95)

        # Safety pass: if projected SoC at departure is below target, charge greedily
        soc_proj = soc
        for h in plugged:
            kw = schedule[i, h]
            if kw > 0:
                soc_proj = min(soc_proj + kw * 0.92 / cap, 0.95)
            elif kw < 0:
                soc_proj = max(soc_proj - (-kw) / 0.92 / cap, 0.10)

        if soc_proj < 0.80:
            for h in plugged:
                if soc_proj >= 0.80:
                    break
                if schedule[i, h] <= 0:
                    schedule[i, h] = max_ch
                    soc_proj = min(soc_proj + max_ch * 0.92 / cap, 0.95)

    return schedule
