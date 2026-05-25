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

Exp 4: LP optimal dispatch (cvxpy/CLARABEL) + same-weekday forecast
  - Maximize (discharge × sell_price − charge × buy_price − degradation_proxy)
    subject to: SoC evolution, power limits, departure SoC >= 0.80
  - Each vehicle solved as an independent LP (they're not coupled in harness)
  - Falls back to greedy on infeasibility or solver failure
  - Degradation proxy: 0.050 EUR/kWh discharged (conservative vs base 0.040)
    to penalise unnecessary cycling and keep SoC in healthy range
"""

import datetime
import numpy as np

try:
    import cvxpy as cp
    _CVXPY_OK = True
except ImportError:
    _CVXPY_OK = False

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_train_prices: np.ndarray | None = None
last_price_forecast: np.ndarray | None = None

_EFF = 0.92
_SOC_MIN = 0.10
_SOC_MAX = 0.95
_SOC_DEP = 0.80
_DEG_RATE = 0.050  # EUR/kWh discharged — proxy for NMC degradation


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
# Price forecast
# ---------------------------------------------------------------------------

def _weekday_forecast(price_history: np.ndarray, date: datetime.date) -> np.ndarray:
    """Weighted mean of same-weekday hours from last 4 occurrences in history."""
    n_days = len(price_history)
    if n_days < 7:
        return price_history[-1].copy() if n_days > 0 else np.full(24, 50.0)

    dow = date.weekday()
    same_dow = []
    for offset in range(1, n_days + 1):
        d = date - datetime.timedelta(days=offset)
        if d.weekday() == dow:
            same_dow.append(n_days - offset)
            if len(same_dow) == 4:
                break

    if not same_dow:
        return price_history[-1].copy()

    weights = np.array([2.0, 2.0, 1.0, 1.0][: len(same_dow)])
    weighted = sum(w * price_history[idx] for w, idx in zip(weights, same_dow))
    return weighted / weights.sum()


# ---------------------------------------------------------------------------
# LP dispatch — per vehicle
# ---------------------------------------------------------------------------

def _lp_vehicle(arr: int, dep: int, soc_init: float, cap: float,
                max_ch: float, max_ds: float,
                sell_fc: np.ndarray, buy_fc: np.ndarray) -> np.ndarray | None:
    """Solve the V2G scheduling LP for one vehicle. Returns (24,) kW schedule or None."""
    if not _CVXPY_OK:
        return None
    plugged_mask = np.zeros(24, dtype=bool)
    plugged_mask[arr:dep] = True

    charge    = cp.Variable(24, nonneg=True)
    discharge = cp.Variable(24, nonneg=True)
    soc       = cp.Variable(25)

    constraints = [
        soc[0] == soc_init,
        soc[dep] >= _SOC_DEP,
    ]
    for h in range(24):
        constraints.append(
            soc[h + 1] == soc[h]
            + charge[h] * _EFF / cap
            - discharge[h] / (_EFF * cap)
        )
        constraints.append(soc[h + 1] >= _SOC_MIN)
        constraints.append(soc[h + 1] <= _SOC_MAX)
        if plugged_mask[h]:
            constraints.append(charge[h] <= max_ch)
            constraints.append(discharge[h] <= max_ds)
        else:
            constraints.append(charge[h] == 0.0)
            constraints.append(discharge[h] == 0.0)

    revenue = cp.sum(cp.multiply(sell_fc, discharge) - cp.multiply(buy_fc, charge))
    degradation = _DEG_RATE * cp.sum(discharge)
    prob = cp.Problem(cp.Maximize(revenue - degradation), constraints)

    try:
        prob.solve(solver=cp.CLARABEL, warm_start=True)
    except Exception:
        return None

    if prob.status not in ("optimal", "optimal_inaccurate"):
        # Infeasible (physically can't reach 80% SoC) — relax departure constraint and retry
        constraints_relaxed = [c for c in constraints if c is not constraints[1]]
        prob2 = cp.Problem(cp.Maximize(revenue - degradation), constraints_relaxed)
        try:
            prob2.solve(solver=cp.CLARABEL)
        except Exception:
            return None
        if prob2.status not in ("optimal", "optimal_inaccurate"):
            return None
        ch = charge.value
        ds = discharge.value
    else:
        ch = charge.value
        ds = discharge.value

    if ch is None or ds is None:
        return None

    schedule = np.clip(ch, 0, max_ch) - np.clip(ds, 0, max_ds)
    return schedule


# ---------------------------------------------------------------------------
# Greedy fallback
# ---------------------------------------------------------------------------

def _greedy_vehicle(arr: int, dep: int, soc_init: float, cap: float,
                    max_ch: float, max_ds: float,
                    sell_fc: np.ndarray, buy_fc: np.ndarray) -> np.ndarray:
    """Simple greedy: charge cheapest hours to 80% SoC, discharge top-priced hours."""
    schedule = np.zeros(24)
    plugged = list(range(arr, dep))
    if not plugged:
        return schedule

    energy_needed = max(0.0, (_SOC_DEP - soc_init) * cap / _EFF)
    hours_by_buy = sorted(plugged, key=lambda h: buy_fc[h])
    soc_sim = soc_init
    for h in hours_by_buy:
        if energy_needed <= 1e-6:
            break
        schedule[h] = max_ch
        soc_sim = min(soc_sim + max_ch * _EFF / cap, _SOC_MAX)
        energy_needed -= max_ch * _EFF

    if max_ds > 0:
        sell_threshold = np.percentile(sell_fc, 75)
        for h in sorted(plugged, key=lambda h: sell_fc[h], reverse=True):
            if sell_fc[h] < sell_threshold:
                break
            margin = (soc_sim - _SOC_DEP) * cap
            remaining = dep - h - 1
            if margin >= max_ds / _EFF and remaining * max_ch * _EFF >= max_ds / _EFF:
                schedule[h] = -max_ds
                soc_sim = max(soc_sim - max_ds / (_EFF * cap), _SOC_MIN)

    return schedule


# ---------------------------------------------------------------------------
# plan_day — called once per evaluation day
# ---------------------------------------------------------------------------

def plan_day(
    date: datetime.date,
    price_history: np.ndarray,
    session_history: list,
    vehicle_states: list,
    *,
    bal_up_history: np.ndarray | None = None,
    bal_dn_history: np.ndarray | None = None,
) -> np.ndarray:
    """Return a (N_vehicles, 24) schedule in kW."""
    global last_price_forecast

    n = len(vehicle_states)
    if n == 0:
        last_price_forecast = None
        return np.zeros((0, 24))

    price_forecast = _weekday_forecast(price_history, date)
    last_price_forecast = price_forecast

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

    schedule = np.zeros((n, 24), dtype=np.float64)

    for i, sess in enumerate(vehicle_states):
        arr    = int(sess["arrival_hour"])
        dep    = int(sess["departure_hour"])
        soc    = float(sess["soc_arrival"])
        cap    = float(sess["battery_capacity_kwh"])
        max_ch = float(sess["max_charge_kw"])
        max_ds = float(sess["max_discharge_kw"])

        result = _lp_vehicle(arr, dep, soc, cap, max_ch, max_ds, sell_fc, buy_fc)
        if result is not None:
            schedule[i] = result
        else:
            schedule[i] = _greedy_vehicle(arr, dep, soc, cap, max_ch, max_ds, sell_fc, buy_fc)

    return schedule
