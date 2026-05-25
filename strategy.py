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

Exp 3: same-weekday rolling mean forecast + priority-charging + feasibility-gated V2G
  - Forecast: weighted mean of same-weekday prices from last 4 weeks (most-recent 2×)
  - Charging: first guarantee departure SoC using cheapest available hours,
    then only allow V2G after confirming recovery is still feasible.
  - This significantly reduces violations versus Exp 2's threshold approach.
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
# Helpers
# ---------------------------------------------------------------------------

def _weekday_forecast(price_history: np.ndarray, date: datetime.date) -> np.ndarray:
    """Weighted mean of same-weekday hours from last 4 weeks in history.

    Gives 2× weight to the two most recent occurrences vs the two older ones.
    Falls back to yesterday's prices when fewer than 7 days of history exist.
    """
    n_days = len(price_history)
    if n_days < 7:
        return price_history[-1].copy() if n_days > 0 else np.full(24, 50.0)

    dow = date.weekday()
    # Find indices of same-weekday days in history (most recent last)
    same_dow = []
    for offset in range(1, n_days + 1):
        d = date - datetime.timedelta(days=offset)
        if d.weekday() == dow:
            same_dow.append(n_days - offset)  # index into price_history
            if len(same_dow) == 4:
                break

    if not same_dow:
        return price_history[-1].copy()

    # Weights: most recent = 2, second = 2, older = 1, oldest = 1
    weights = np.array([2.0, 2.0, 1.0, 1.0][: len(same_dow)])
    weighted = sum(w * price_history[idx] for w, idx in zip(weights, same_dow))
    return weighted / weights.sum()


def _can_discharge(h: int, dep: int, soc_after_ds: float, cap: float,
                   max_ch: float, needs_kwh: float) -> bool:
    """Return True if there's enough charge capacity in hours (h+1..dep-1) for needs_kwh.

    needs_kwh is the energy still needed to reach 80% SoC after the discharge.
    """
    if soc_after_ds < 0.10:
        return False
    remaining_hours = dep - h - 1
    max_recoverable_kwh = remaining_hours * max_ch * 0.92
    return max_recoverable_kwh >= needs_kwh


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

    # --- Same-weekday rolling mean forecast ---
    price_forecast = _weekday_forecast(price_history, date)
    last_price_forecast = price_forecast

    # Effective sell/buy prices
    if bal_up_history is not None and len(bal_up_history) > 0:
        bal_up_fc = _weekday_forecast(bal_up_history, date)
    else:
        bal_up_fc = price_forecast * 1.10

    if bal_dn_history is not None and len(bal_dn_history) > 0:
        bal_dn_fc = _weekday_forecast(bal_dn_history, date)
    else:
        bal_dn_fc = price_forecast * 0.90

    sell_fc = np.maximum(price_forecast, bal_up_fc)
    buy_fc  = np.minimum(price_forecast, bal_dn_fc)

    sell_threshold = np.percentile(sell_fc, 75)  # top 25% for V2G discharge

    schedule = np.zeros((n, 24), dtype=np.float64)

    for i, sess in enumerate(vehicle_states):
        arr    = int(sess["arrival_hour"])
        dep    = int(sess["departure_hour"])
        soc    = float(sess["soc_arrival"])
        cap    = float(sess["battery_capacity_kwh"])
        max_ch = float(sess["max_charge_kw"])
        max_ds = float(sess["max_discharge_kw"])

        plugged = list(range(arr, dep))
        if not plugged:
            continue

        # --- Step 1: Guarantee departure SoC using cheapest available hours ---
        energy_needed = max(0.0, (0.80 - soc) * cap / 0.92)  # kWh input needed
        hours_by_price = sorted(plugged, key=lambda h: buy_fc[h])
        soc_sim = soc
        for h in hours_by_price:
            if energy_needed <= 1e-6:
                break
            charge_kwh = min(max_ch, energy_needed / 0.92) if energy_needed > 0 else 0
            charge_kwh = min(charge_kwh, max_ch)
            if soc_sim < 0.95:
                schedule[i, h] = charge_kwh
                delivered = charge_kwh * 0.92
                soc_sim = min(soc_sim + delivered / cap, 0.95)
                energy_needed -= delivered

        # --- Step 2: V2G discharge in expensive hours (feasibility-gated) ---
        if max_ds > 0:
            # Recompute SoC trajectory after charging
            soc_sim = soc
            for h in plugged:
                if schedule[i, h] > 0:
                    soc_sim = min(soc_sim + schedule[i, h] * 0.92 / cap, 0.95)

            for h in sorted(plugged, key=lambda h: sell_fc[h], reverse=True):
                if sell_fc[h] < sell_threshold:
                    break  # sorted descending, no need to check cheaper hours
                # Margin above 80% SoC at this moment
                margin_kwh = (soc_sim - 0.80) * cap  # energy above target
                if margin_kwh < max_ds / 0.92:
                    continue  # not enough margin to discharge without V2G risk

                # After hypothetical discharge, how much kWh needed to recover?
                soc_after_ds = soc_sim - max_ds / 0.92 / cap
                needs_kwh = max(0.0, (0.80 - soc_after_ds) * cap / 0.92)

                if _can_discharge(h, dep, soc_after_ds, cap, max_ch, needs_kwh):
                    # Replace any existing charge with discharge
                    existing_charge = schedule[i, h] if schedule[i, h] > 0 else 0.0
                    if existing_charge > 0:
                        # Remove the charge we assigned in step 1, reclaim energy_needed
                        soc_sim = max(soc_sim - existing_charge * 0.92 / cap, 0.10)
                    schedule[i, h] = -max_ds
                    soc_sim = max(soc_sim - max_ds / 0.92 / cap, 0.10)

    return schedule
