"""
One-time data preparation for V2G fleet autoresearch experiments.

Downloads electricity price data (Danish Energidataservice / Nord Pool DK1)
and generates reproducible synthetic EV fleet session data.

Usage:
    uv run prepare_v2g.py              # download + cache all data (one-time, ~1 min)
    uv run prepare_v2g.py --eval       # full backtest on val period (2023)
    uv run prepare_v2g.py --smoke      # quick smoke test (7 days)
    uv run prepare_v2g.py --test       # final evaluation on test period (2024)

Data stored in ~/.cache/autoresearch_v2g/
"""

import os
import sys
import time
import json
import argparse
import importlib
import importlib.util
import datetime
import random
from pathlib import Path

import requests
import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

# Fleet parameters
NUM_VEHICLES = 30              # fleet size (workplace depot)
V2G_FRACTION = 0.50            # fraction of fleet with V2G (discharge) capability
BATTERY_CAPACITY_KWH = 60.0    # kWh per vehicle (typical mid-size EV)
MAX_CHARGE_KW = 11.0           # max charge rate per vehicle (AC Type 2)
MAX_DISCHARGE_KW = 7.4         # max V2G discharge rate (inverter limit)
GRID_LIMIT_KW = 300.0          # total site grid connection limit (kW)

# SoC bounds (enforced by harness — agent cannot remove these)
SOC_MIN = 0.10                 # minimum allowed SoC (battery protection)
SOC_MAX = 0.95                 # maximum allowed SoC (battery protection)
SOC_DEPARTURE_MIN = 0.80       # vehicle must reach this SoC before departure

# Efficiency (round-trip losses)
CHARGE_EFF = 0.92
DISCHARGE_EFF = 0.92

# Battery degradation model (linear, fixed)
BATTERY_COST_EUR_PER_KWH = 150.0    # battery replacement cost
BATTERY_LIFETIME_CYCLES = 3000      # usable full-charge-equivalent cycles
DEGRADATION_EUR_PER_KWH = BATTERY_COST_EUR_PER_KWH / BATTERY_LIFETIME_CYCLES  # ~0.050 EUR/kWh

# V2G-capable vehicles: first N vehicles (deterministic subset)
NUM_V2G_VEHICLES = int(NUM_VEHICLES * V2G_FRACTION)  # 15

# Data splits (pinned — do not change after initial run)
TRAIN_START = datetime.date(2019, 1, 1)
TRAIN_END   = datetime.date(2022, 12, 31)
VAL_START   = datetime.date(2023, 1, 1)
VAL_END     = datetime.date(2023, 12, 31)
TEST_START  = datetime.date(2024, 1, 1)
TEST_END    = datetime.date(2024, 12, 31)

# Nord Pool price zone (Danish grid, West Denmark)
PRICE_AREA = "DK1"

# Reproducibility seed for synthetic session generation
SESSION_SEED = 42

# Context window: how many past days of history are passed to strategy
HISTORY_DAYS = 30

# Cache locations
CACHE_DIR   = Path(os.path.expanduser("~")) / ".cache" / "autoresearch_v2g"
PRICES_FILE = CACHE_DIR / "prices_eur_mwh.npy"   # float32, shape (N_days, 24)
SESSIONS_FILE = CACHE_DIR / "sessions.json"       # list[list[dict]]
DATES_FILE  = CACHE_DIR / "dates.json"            # list[str] "YYYY-MM-DD"

# ---------------------------------------------------------------------------
# Data download: electricity spot prices
# ---------------------------------------------------------------------------

def _date_range(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += datetime.timedelta(days=1)
    return out


def download_prices() -> tuple[np.ndarray, list[datetime.date]]:
    """Download day-ahead spot prices from Energidataservice (Danish TSO, open data).

    Returns prices shape (N_days, 24) in EUR/MWh and corresponding date list.
    """
    print("Downloading electricity prices from Energidataservice...")

    base = "https://api.energidataservice.dk/dataset/Elspotprices"
    start_str = TRAIN_START.strftime("%Y-%m-%dT00:00")
    end_str   = TEST_END.strftime("%Y-%m-%dT23:00")
    limit = 100_000  # full dataset fits in one request (~52,560 hourly records)

    params = {
        "start":  start_str,
        "end":    end_str,
        "filter": json.dumps({"PriceArea": [PRICE_AREA]}),
        "sort":   "HourUTC asc",
        "limit":  limit,
        "offset": 0,
    }

    resp = requests.get(base, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    records = data.get("records", [])
    print(f"  Downloaded {len(records)} hourly price records.")

    if len(records) < 1000:
        raise RuntimeError("Too few price records — check API or network connection.")

    # Build date → 24-hour price array
    date_prices: dict[datetime.date, list[float | None]] = {}
    for rec in records:
        # HourUTC is like "2019-01-01T00:00:00"
        ts_str = rec.get("HourUTC", "")
        price = rec.get("SpotPriceEUR")
        if not ts_str or price is None:
            continue
        dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        date = dt.date()
        hour = dt.hour
        if date not in date_prices:
            date_prices[date] = [None] * 24
        date_prices[date][hour] = float(price)

    # Fill any missing hours with daily mean (rare)
    all_dates = sorted(date_prices.keys())
    for date in all_dates:
        row = date_prices[date]
        valid = [v for v in row if v is not None]
        fill = float(np.mean(valid)) if valid else 50.0
        date_prices[date] = [v if v is not None else fill for v in row]

    dates = all_dates
    prices = np.array([date_prices[d] for d in dates], dtype=np.float32)
    return prices, dates


# ---------------------------------------------------------------------------
# Synthetic EV session generation
# ---------------------------------------------------------------------------

def generate_sessions(dates: list[datetime.date], seed: int = SESSION_SEED) -> list[list[dict]]:
    """Generate synthetic workplace EV charging sessions.

    Distributions based on published ACN-Data statistics:
    - Arrival: Gaussian N(8.5h, 1.2h) — morning commuter peak
    - Duration: Gaussian N(8h, 2h) — typical 9-to-5 workplace session
    - Energy: correlated with duration, 2 kWh/h ± noise
    - SoC on arrival: N(0.35, 0.15) — partially depleted battery
    - Weekday arrival probability 70%, weekend 20%
    """
    rng_py  = random.Random(seed)
    rng_np  = np.random.default_rng(seed)

    v2g_ids = set(range(NUM_V2G_VEHICLES))  # first 15 vehicles are V2G capable

    all_sessions: list[list[dict]] = []

    for date in dates:
        is_weekend = date.weekday() >= 5
        p_arrive = 0.20 if is_weekend else 0.70

        day: list[dict] = []
        for vid in range(NUM_VEHICLES):
            if rng_py.random() > p_arrive:
                continue

            # Arrival hour (rounded to integer, clipped)
            arr_h = int(np.clip(rng_np.normal(8.5, 1.2), 6, 12))
            # Session duration (hours)
            dur_h = int(np.clip(rng_np.normal(8.0, 2.0), 4, 13))
            dep_h = min(arr_h + dur_h, 21)  # latest departure 21:00
            if dep_h <= arr_h:
                dep_h = arr_h + 4

            # Energy needed (kWh) — correlated with session duration
            energy_kwh = float(np.clip(rng_np.normal(dur_h * 2.0, 3.0), 3.0, 40.0))

            # SoC on arrival
            soc_arr = float(np.clip(rng_np.normal(0.35, 0.15), SOC_MIN, 0.65))

            day.append({
                "vehicle_id":           vid,
                "arrival_hour":         arr_h,
                "departure_hour":       dep_h,
                "energy_needed_kwh":    energy_kwh,
                "soc_arrival":          soc_arr,
                "battery_capacity_kwh": BATTERY_CAPACITY_KWH,
                "max_charge_kw":        MAX_CHARGE_KW,
                "max_discharge_kw":     MAX_DISCHARGE_KW if vid in v2g_ids else 0.0,
            })

        all_sessions.append(day)

    return all_sessions


# ---------------------------------------------------------------------------
# Data caching
# ---------------------------------------------------------------------------

def prepare_data(force: bool = False) -> None:
    """Download prices and generate sessions; save to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and PRICES_FILE.exists() and SESSIONS_FILE.exists():
        print("Data already cached. Use --force to re-download.")
        return

    prices, dates = download_prices()

    print("Generating synthetic EV fleet sessions...")
    sessions = generate_sessions(dates)

    np.save(PRICES_FILE, prices)
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, separators=(",", ":"))
    with open(DATES_FILE, "w") as f:
        json.dump([d.isoformat() for d in dates], f)

    print(f"Cached {len(dates)} days of data ({prices.shape[0]} days × 24h prices).")
    print(f"  Price range: {prices.min():.1f} – {prices.max():.1f} EUR/MWh")
    _print_split_info(dates)


def load_data() -> tuple[np.ndarray, list[list[dict]], list[datetime.date]]:
    """Load cached data. Raises if not prepared yet."""
    if not PRICES_FILE.exists():
        print("Data not found. Run 'uv run prepare_v2g.py' first.")
        sys.exit(1)

    prices = np.load(PRICES_FILE)
    with open(SESSIONS_FILE) as f:
        sessions = json.load(f)
    with open(DATES_FILE) as f:
        dates = [datetime.date.fromisoformat(s) for s in json.load(f)]

    return prices, sessions, dates


def _print_split_info(dates: list[datetime.date]) -> None:
    splits = [
        ("train", TRAIN_START, TRAIN_END),
        ("val",   VAL_START,   VAL_END),
        ("test",  TEST_START,  TEST_END),
    ]
    for name, s, e in splits:
        n = sum(1 for d in dates if s <= d <= e)
        print(f"  {name}: {s} – {e} ({n} days)")


# ---------------------------------------------------------------------------
# Day simulation
# ---------------------------------------------------------------------------

def simulate_day(
    sessions: list[dict],
    schedule: np.ndarray,          # shape (N_sessions, 24), kW; + = charge, – = discharge
    actual_prices: np.ndarray,     # shape (24,), EUR/MWh
) -> dict:
    """Simulate one day of V2G dispatch against actual prices.

    Enforces all physical constraints (SoC bounds, charger limits, grid limit).
    Applies departure-SoC penalty for constraint violations.
    """
    n = len(sessions)
    if n == 0:
        return {k: 0.0 for k in (
            "net_revenue_eur", "gross_revenue_eur", "degradation_cost_eur",
            "violation_count", "total_charged_kwh", "total_discharged_kwh",
        )}

    schedule = np.asarray(schedule, dtype=np.float64)
    if schedule.shape != (n, 24):
        schedule = np.zeros((n, 24))

    gross_revenue  = 0.0
    degradation    = 0.0
    violations     = 0
    total_charged  = 0.0
    total_discharged = 0.0

    # Hourly net grid draw tracker (positive = consuming, negative = exporting)
    hourly_grid = np.zeros(24)

    for i, sess in enumerate(sessions):
        soc    = sess["soc_arrival"]
        cap    = sess["battery_capacity_kwh"]
        arr    = sess["arrival_hour"]
        dep    = sess["departure_hour"]
        max_ch = sess["max_charge_kw"]
        max_ds = sess["max_discharge_kw"]

        for h in range(24):
            if h < arr or h >= dep:
                continue

            raw_kw = float(schedule[i, h])

            if raw_kw >= 0:  # charging
                headroom = max(0.0, GRID_LIMIT_KW - hourly_grid[h])
                kw = min(raw_kw, max_ch, headroom)
                energy_in = kw * CHARGE_EFF
                new_soc = min(soc + energy_in / cap, SOC_MAX)
                actual_energy_in = (new_soc - soc) * cap
                actual_kw = actual_energy_in / CHARGE_EFF
                cost = actual_kw * actual_prices[h] / 1000.0
                gross_revenue -= cost
                hourly_grid[h] += actual_kw
                total_charged += actual_kw
                soc = new_soc

            else:  # discharging (V2G)
                if max_ds <= 0:
                    continue
                kw = min(-raw_kw, max_ds)
                energy_out = kw / DISCHARGE_EFF
                new_soc = max(soc - energy_out / cap, SOC_MIN)
                actual_energy_out = (soc - new_soc) * cap
                actual_kw_exported = actual_energy_out * DISCHARGE_EFF
                rev = actual_kw_exported * actual_prices[h] / 1000.0
                gross_revenue += rev
                degradation += actual_energy_out * DEGRADATION_EUR_PER_KWH
                hourly_grid[h] -= actual_kw_exported
                total_discharged += actual_energy_out
                soc = new_soc

        # Check departure SoC requirement
        if soc < SOC_DEPARTURE_MIN:
            missing_kwh = (SOC_DEPARTURE_MIN - soc) * cap
            # Penalty: 3× the cost of charging the missing energy at today's peak price
            peak_price = float(actual_prices[arr:dep].max()) if dep > arr else actual_prices.max()
            penalty = missing_kwh * peak_price / 1000.0 * 3.0
            gross_revenue -= penalty
            violations += 1

    net_revenue = gross_revenue - degradation
    return {
        "net_revenue_eur":      net_revenue,
        "gross_revenue_eur":    gross_revenue,
        "degradation_cost_eur": degradation,
        "violation_count":      violations,
        "total_charged_kwh":    total_charged,
        "total_discharged_kwh": total_discharged,
    }


# ---------------------------------------------------------------------------
# Evaluation harness (fixed — this is what the agent optimizes against)
# ---------------------------------------------------------------------------

def evaluate_v2g(
    strategy_path: str = "strategy.py",
    split: str = "val",
) -> dict:
    """Run full backtest of strategy.py against the fixed val (or test) period.

    Strategy interface (must be implemented in strategy.py):
        fit(train_prices: np.ndarray, train_sessions: list[list[dict]]) -> None
            Called once before evaluation. Fit any models on training data.

        plan_day(
            date: datetime.date,
            price_history: np.ndarray,   # shape (H, 24), H ≤ HISTORY_DAYS
            session_history: list[list], # H days of past sessions
            vehicle_states: list[dict],  # today's vehicles (arrival/departure/SoC)
        ) -> np.ndarray                  # shape (N_vehicles, 24), kW

        Optional attribute:
            last_price_forecast: np.ndarray | None  # shape (24,) — logged for diagnostics

    Primary metric:
        val_revenue_per_kwh  (higher is better)
    """
    prices, sessions, dates = load_data()
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # Determine eval period
    if split == "val":
        eval_start, eval_end = VAL_START, VAL_END
    elif split == "test":
        eval_start, eval_end = TEST_START, TEST_END
    elif split == "smoke":
        eval_start = datetime.date(2023, 6, 1)
        eval_end   = datetime.date(2023, 6, 7)
    else:
        raise ValueError(f"Unknown split: {split!r}")

    eval_dates   = [d for d in dates if eval_start <= d <= eval_end]
    eval_indices = [date_to_idx[d] for d in eval_dates]
    train_indices = [i for i, d in enumerate(dates) if d < eval_start]

    if not eval_dates:
        raise RuntimeError(f"No dates found for split={split!r}")

    # Load strategy
    spec = importlib.util.spec_from_file_location("strategy", strategy_path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"Error loading strategy: {e}")
        raise

    # Fit on training data
    train_prices   = prices[train_indices]
    train_sessions = [sessions[i] for i in train_indices]
    mod.fit(train_prices, train_sessions)

    # Backtest
    t0 = time.time()
    total_net_revenue  = 0.0
    total_charged      = 0.0
    total_discharged   = 0.0
    total_violations   = 0
    forecast_rmse_vals = []

    for day_num, (actual_idx, actual_date) in enumerate(zip(eval_indices, eval_dates)):
        hist_start    = max(0, actual_idx - HISTORY_DAYS)
        price_history   = prices[hist_start:actual_idx]
        session_history = sessions[hist_start:actual_idx]
        today_sessions  = sessions[actual_idx]
        actual_prices   = prices[actual_idx]

        schedule = mod.plan_day(
            date=actual_date,
            price_history=price_history,
            session_history=session_history,
            vehicle_states=today_sessions,
        )

        # Defensive coercion
        if schedule is None:
            schedule = np.zeros((len(today_sessions), 24))
        schedule = np.asarray(schedule, dtype=np.float64)
        if schedule.shape != (len(today_sessions), 24):
            schedule = np.zeros((len(today_sessions), 24))

        # Track optional price forecast diagnostic
        forecast = getattr(mod, "last_price_forecast", None)
        if forecast is not None:
            fc = np.asarray(forecast, dtype=np.float64)
            if fc.shape == (24,):
                rmse = float(np.sqrt(np.mean((fc - actual_prices.astype(np.float64)) ** 2)))
                forecast_rmse_vals.append(rmse)

        result = simulate_day(today_sessions, schedule, actual_prices)
        total_net_revenue  += result["net_revenue_eur"]
        total_charged      += result["total_charged_kwh"]
        total_discharged   += result["total_discharged_kwh"]
        total_violations   += result["violation_count"]

    total_seconds = time.time() - t0
    total_kwh_transacted = total_charged + total_discharged
    val_revenue_per_kwh  = total_net_revenue / max(1.0, total_kwh_transacted)

    return {
        "val_revenue_per_kwh":        val_revenue_per_kwh,
        "total_net_revenue_eur":       total_net_revenue,
        "total_charged_kwh":           total_charged,
        "total_discharged_kwh":        total_discharged,
        "total_constraint_violations": total_violations,
        "num_eval_days":               len(eval_dates),
        "total_seconds":               total_seconds,
        **({"forecast_rmse_eur_mwh": float(np.mean(forecast_rmse_vals))}
           if forecast_rmse_vals else {}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="V2G fleet autoresearch data prep + evaluation")
    parser.add_argument("--eval",  action="store_true", help="Run full val backtest")
    parser.add_argument("--smoke", action="store_true", help="Quick 7-day smoke test")
    parser.add_argument("--test",  action="store_true", help="Final test-set evaluation")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--strategy", default="strategy.py", help="Path to strategy.py")
    args = parser.parse_args()

    if args.eval or args.smoke or args.test:
        split = "test" if args.test else ("smoke" if args.smoke else "val")
        print(f"Running backtest on '{split}' split...")
        metrics = evaluate_v2g(strategy_path=args.strategy, split=split)

        # Print in the format the agent parses with grep
        print("---")
        print(f"val_revenue_per_kwh:          {metrics['val_revenue_per_kwh']:.6f}")
        print(f"total_net_revenue_eur:         {metrics['total_net_revenue_eur']:.2f}")
        print(f"total_charged_kwh:             {metrics['total_charged_kwh']:.1f}")
        print(f"total_discharged_kwh:          {metrics['total_discharged_kwh']:.1f}")
        print(f"total_constraint_violations:   {metrics['total_constraint_violations']}")
        print(f"num_eval_days:                 {metrics['num_eval_days']}")
        print(f"total_seconds:                 {metrics['total_seconds']:.1f}")
        if "forecast_rmse_eur_mwh" in metrics:
            print(f"forecast_rmse_eur_mwh:         {metrics['forecast_rmse_eur_mwh']:.2f}")
        return

    # Default: prepare data
    prepare_data(force=args.force)
    print("\nSetup complete. Run 'uv run prepare_v2g.py --smoke' to verify the pipeline.")


if __name__ == "__main__":
    main()
