"""
V2G fleet scheduling strategy — the ONLY file the agent may modify.

Implements two required functions:
    fit(train_prices, train_sessions)  -> None
    plan_day(date, price_history, session_history, vehicle_states) -> np.ndarray

Optionally expose:
    last_price_forecast: np.ndarray | None  # shape (24,), logged as diagnostic

Metric optimized by the harness:
    val_revenue_per_kwh  (higher is better)
    = (gross V2G revenue - battery degradation cost) / total kWh transacted
    evaluated on fixed val period (2023), 30-vehicle workplace fleet.

Current baseline strategy:
  - Price forecast: persistence (yesterday's prices = tomorrow's prices)
  - Dispatch: greedy rule — charge in cheapest 8h, discharge in most expensive 4h
               subject to SoC and charger constraints
"""

import datetime
import numpy as np

# ---------------------------------------------------------------------------
# Module-level state (survives across plan_day calls within one backtest)
# ---------------------------------------------------------------------------

_train_prices: np.ndarray | None = None  # shape (N_train, 24)
last_price_forecast: np.ndarray | None = None  # exposed for diagnostic logging


# ---------------------------------------------------------------------------
# fit — called once before backtest begins
# ---------------------------------------------------------------------------

def fit(
    train_prices: np.ndarray,       # shape (N_train_days, 24), EUR/MWh
    train_sessions: list,           # N_train_days × list[dict]
) -> None:
    """Fit any models on training data. Store state in module globals."""
    global _train_prices
    _train_prices = train_prices.copy()


# ---------------------------------------------------------------------------
# plan_day — called once per evaluation day
# ---------------------------------------------------------------------------

def plan_day(
    date: datetime.date,
    price_history: np.ndarray,      # shape (H, 24), H ≤ 30 days of history
    session_history: list,          # H days of past sessions
    vehicle_states: list,           # today's vehicles (dicts with arrival/departure/SoC etc.)
) -> np.ndarray:
    """Produce 24h charge/discharge schedule for all vehicles.

    Returns np.ndarray of shape (N_vehicles, 24):
        Positive values = charging (kW)
        Negative values = discharging / V2G export (kW, only for V2G-capable vehicles)
    """
    global last_price_forecast

    n_vehicles = len(vehicle_states)
    if n_vehicles == 0:
        last_price_forecast = None
        return np.zeros((0, 24))

    # --- Price forecast: persistence (yesterday's prices) ---
    if len(price_history) > 0:
        price_forecast = price_history[-1].copy().astype(np.float64)
    else:
        price_forecast = np.full(24, 50.0)  # fallback: 50 EUR/MWh flat

    last_price_forecast = price_forecast

    # --- Greedy dispatch ---
    schedule = np.zeros((n_vehicles, 24), dtype=np.float64)

    # Rank hours by predicted price
    hour_rank = np.argsort(price_forecast)          # cheapest → most expensive
    cheap_hours = set(hour_rank[:8].tolist())       # 8 cheapest hours: charge
    expensive_hours = set(hour_rank[-4:].tolist())  # 4 most expensive hours: discharge

    for i, sess in enumerate(vehicle_states):
        arr      = sess["arrival_hour"]
        dep      = sess["departure_hour"]
        soc      = sess["soc_arrival"]
        cap      = sess["battery_capacity_kwh"]
        max_ch   = sess["max_charge_kw"]
        max_dis  = sess["max_discharge_kw"]
        soc_tgt  = 0.80  # must reach before departure

        # Simulate SoC forward to decide how aggressively to charge/discharge
        soc_sim = soc
        plugged_hours = list(range(arr, dep))

        for h in plugged_hours:
            if h in expensive_hours and max_dis > 0:
                # Discharge only if we have buffer above minimum safe SoC
                if soc_sim > soc_tgt + 0.05:
                    schedule[i, h] = -max_dis
                    energy_out = max_dis / 0.92
                    soc_sim = max(soc_sim - energy_out / cap, 0.10)
            elif h in cheap_hours:
                if soc_sim < 0.95:
                    schedule[i, h] = max_ch
                    energy_in = max_ch * 0.92
                    soc_sim = min(soc_sim + energy_in / cap, 0.95)

        # Safety pass: ensure we reach required SoC by departure.
        # If projected SoC at end of planned schedule is below target,
        # override remaining cheap hours with full charging.
        soc_check = soc
        for h in plugged_hours:
            kw = schedule[i, h]
            if kw > 0:
                soc_check = min(soc_check + kw * 0.92 / cap, 0.95)
            elif kw < 0:
                soc_check = max(soc_check - (-kw) / 0.92 / cap, 0.10)

        if soc_check < soc_tgt:
            for h in plugged_hours:
                if soc_check >= soc_tgt:
                    break
                if schedule[i, h] <= 0:
                    schedule[i, h] = max_ch
                    soc_check = min(soc_check + max_ch * 0.92 / cap, 0.95)

    return schedule
