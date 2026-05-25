"""
One-time data preparation for V2G fleet autoresearch experiments.

Data sources (all free, no API key required by default):
  - Electricity prices:   Energidataservice / Nord Pool DK1 day-ahead spot
  - Balancing prices:     Energidataservice / Danish TSO up/down regulation market
  - EV fleet sessions:    Synthetic, parameterized from published ACN-Data statistics

Optional enhanced sources (set environment variables to enable):
  - ENTSOE_TOKEN:  Use ENTSO-E Transparency Platform instead (broader European coverage)
  - ACN_TOKEN:     Use real Caltech ACN-Data sessions instead of synthetic

Usage:
    uv run prepare_v2g.py              # download + cache all data (one-time, ~1 min)
    uv run prepare_v2g.py --eval       # full backtest on val period (2023)
    uv run prepare_v2g.py --smoke      # quick smoke test (7 days)
    uv run prepare_v2g.py --test       # final evaluation on test period (2024)
    uv run prepare_v2g.py --force      # re-download even if cache exists

Data stored in ~/.cache/autoresearch_v2g/

--- DO NOT MODIFY THIS FILE ---
This is the fixed evaluation harness. All strategy research goes in strategy.py.
Modifying this file invalidates experiment comparisons across branches.
"""

import os
import sys
import time
import json
import argparse
import importlib.util
import datetime
import random
from pathlib import Path

import requests
import numpy as np

# ---------------------------------------------------------------------------
# Fleet constants (fixed — do not modify)
# ---------------------------------------------------------------------------

# Realistic EV fleet mix for a 2024-era European workplace depot.
# Each entry: (name, fleet_fraction, battery_kwh, max_charge_kw, max_discharge_kw)
# max_discharge_kw=0 means no V2G capability.
# Sources: manufacturer specs (Hyundai, Nissan, VW, Tesla, Renault).
EV_FLEET_TYPES = [
    ("Hyundai IONIQ 5",       0.20, 77.4, 11.0, 3.6),   # V2G via ISO 15118-20
    ("Nissan Leaf e+",        0.15, 62.0,  6.6, 3.3),   # V2G via CHAdeMO
    ("Renault Megane E-Tech", 0.15, 60.0, 11.0, 3.7),   # V2G capable (2024+)
    ("Volkswagen ID.4",       0.20, 77.0, 11.0, 0.0),   # no V2G (AC only)
    ("Tesla Model 3 LR",      0.15, 75.0, 11.0, 0.0),   # no V2G
    ("Renault Zoe",           0.15, 52.0, 22.0, 0.0),   # fast AC charge, no V2G
]
# V2G-capable fraction: 0.20 + 0.15 + 0.15 = 0.50

NUM_VEHICLES = 30              # total fleet size
GRID_LIMIT_KW = 300.0          # total site grid connection limit (kW)

# SoC bounds (enforced by harness — agent cannot override these)
SOC_MIN = 0.10                 # minimum SoC (battery protection, all models)
SOC_MAX = 0.95                 # maximum SoC (battery protection, all models)
SOC_DEPARTURE_MIN = 0.80       # vehicle must reach this before departure

# Efficiency (AC Type 2 round-trip, IEC 62196)
CHARGE_EFF    = 0.92
DISCHARGE_EFF = 0.92

# ---------------------------------------------------------------------------
# Battery degradation model (fixed — based on published NMC Li-ion data)
#
# Model: SoC-stress-weighted degradation per kWh discharged.
#   deg_cost(kWh, soc_mean) = BASE_RATE × soc_stress(soc_mean) × kWh
#
# Rationale:
#   - Base rate from Schmalstieg et al. (2014) and Wang et al. (2011):
#     NMC packs lose ~20% capacity after ~3000 full-equivalent cycles.
#   - SoC stress: Li-ion degrades faster at high SoC (lithium plating,
#     SEI growth) and slightly faster at very low SoC (copper dissolution).
#     Minimum degradation occurs around 40-60% SoC.
#   - Keeping SoC between 20-80% roughly doubles battery life.
# ---------------------------------------------------------------------------

BATTERY_COST_EUR_PER_KWH = 120.0   # 2024 NMC pack cost (BNEF estimate)
BATTERY_LIFETIME_FCE     = 3000    # full charge equivalents to 80% capacity
# Base degradation cost per kWh discharged at 50% SoC (sweet spot)
_BASE_DEG = BATTERY_COST_EUR_PER_KWH / BATTERY_LIFETIME_FCE   # ~0.040 EUR/kWh


def _soc_stress(soc_mean: float) -> float:
    """SoC-dependent degradation multiplier for NMC Li-ion.

    Returns a factor ≥ 1.0. Minimum (1.0) at 50% SoC.
    At 95% SoC: ~2.3×. At 10% SoC: ~1.3×.
    """
    # Asymmetric parabola: steeper penalty above 50% than below
    if soc_mean >= 0.5:
        return 1.0 + 5.0 * (soc_mean - 0.5) ** 2
    else:
        return 1.0 + 1.0 * (0.5 - soc_mean) ** 2


def degradation_eur(energy_kwh: float, soc_before: float, soc_after: float) -> float:
    """Compute degradation cost in EUR for one discharge event.

    Args:
        energy_kwh:  kWh taken from the battery (positive)
        soc_before:  SoC at start of discharge
        soc_after:   SoC at end of discharge
    """
    soc_mid = (soc_before + soc_after) / 2
    return energy_kwh * _BASE_DEG * _soc_stress(soc_mid)


# ---------------------------------------------------------------------------
# Data splits (pinned — never change these after first data download)
# ---------------------------------------------------------------------------

TRAIN_START = datetime.date(2019, 1, 1)
TRAIN_END   = datetime.date(2022, 12, 31)
VAL_START   = datetime.date(2023, 1, 1)    # agent optimizes against this
VAL_END     = datetime.date(2023, 12, 31)
TEST_START  = datetime.date(2024, 1, 1)    # held out until final evaluation
TEST_END    = datetime.date(2024, 12, 31)

PRICE_AREA   = "DK1"
SESSION_SEED = 42
HISTORY_DAYS = 30

# ---------------------------------------------------------------------------
# Cache locations
# ---------------------------------------------------------------------------

CACHE_DIR          = Path(os.path.expanduser("~")) / ".cache" / "autoresearch_v2g"
PRICES_FILE        = CACHE_DIR / "prices_eur_mwh.npy"        # (N_days, 24) spot
BALANCING_UP_FILE  = CACHE_DIR / "balancing_up_eur_mwh.npy"  # (N_days, 24) up-reg
BALANCING_DN_FILE  = CACHE_DIR / "balancing_dn_eur_mwh.npy"  # (N_days, 24) down-reg
SESSIONS_FILE      = CACHE_DIR / "sessions.json"
DATES_FILE         = CACHE_DIR / "dates.json"
FLEET_FILE         = CACHE_DIR / "fleet_specs.json"           # per-vehicle specs

# ---------------------------------------------------------------------------
# Fleet generation
# ---------------------------------------------------------------------------

def _build_fleet(seed: int = SESSION_SEED) -> list[dict]:
    """Assign each of NUM_VEHICLES a specific EV type from the fleet mix."""
    rng = random.Random(seed)
    fleet = []
    # Expand fleet_types by fraction to assign types
    assigned = []
    for name, frac, bat_kwh, ch_kw, dis_kw in EV_FLEET_TYPES:
        count = round(frac * NUM_VEHICLES)
        assigned.extend([(name, bat_kwh, ch_kw, dis_kw)] * count)
    # Pad or trim to exactly NUM_VEHICLES
    while len(assigned) < NUM_VEHICLES:
        assigned.append(assigned[-1])
    assigned = assigned[:NUM_VEHICLES]
    rng.shuffle(assigned)
    for vid, (name, bat_kwh, ch_kw, dis_kw) in enumerate(assigned):
        fleet.append({
            "vehicle_id":           vid,
            "model":                name,
            "battery_capacity_kwh": bat_kwh,
            "max_charge_kw":        ch_kw,
            "max_discharge_kw":     dis_kw,
        })
    return fleet

FLEET_SPECS = _build_fleet()  # deterministic, loaded at import time


def generate_sessions(dates: list[datetime.date], seed: int = SESSION_SEED) -> list[list[dict]]:
    """Generate synthetic EV fleet sessions for each date.

    Distributions derived from ACN-Data (Caltech EV dataset) statistics:
      Fri et al. (2021), "Electric Vehicle Charging in the United States"
      - Workplace arrival:  Gaussian μ=8.5h, σ=1.2h
      - Session duration:   Gaussian μ=8.0h, σ=2.0h
      - Energy requested:   ~2 kWh/h of session duration ± noise
      - SoC on arrival:     Gaussian μ=0.35, σ=0.15 (EV arrives ~35% charged)
      - Weekday prob:       70%  |  Weekend prob: 20%

    Returns: list of N_days lists, each containing session dicts.
    """
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    all_sessions: list[list[dict]] = []

    for date in dates:
        is_weekend = date.weekday() >= 5
        p_arrive   = 0.20 if is_weekend else 0.70

        day: list[dict] = []
        for spec in FLEET_SPECS:
            if rng_py.random() > p_arrive:
                continue

            arr_h = int(np.clip(rng_np.normal(8.5, 1.2), 6, 12))
            dur_h = int(np.clip(rng_np.normal(8.0, 2.0), 4, 13))
            dep_h = min(arr_h + dur_h, 21)
            if dep_h <= arr_h:
                dep_h = arr_h + 4

            energy_kwh = float(np.clip(rng_np.normal(dur_h * 2.0, 3.0), 3.0, 40.0))
            soc_arr    = float(np.clip(rng_np.normal(0.35, 0.15), SOC_MIN, 0.65))

            day.append({
                "vehicle_id":           spec["vehicle_id"],
                "model":                spec["model"],
                "arrival_hour":         arr_h,
                "departure_hour":       dep_h,
                "energy_needed_kwh":    energy_kwh,
                "soc_arrival":          soc_arr,
                "battery_capacity_kwh": spec["battery_capacity_kwh"],
                "max_charge_kw":        spec["max_charge_kw"],
                "max_discharge_kw":     spec["max_discharge_kw"],
            })

        all_sessions.append(day)

    return all_sessions


# ---------------------------------------------------------------------------
# Data download: spot prices
# ---------------------------------------------------------------------------

def _date_range(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    out, d = [], start
    while d <= end:
        out.append(d)
        d += datetime.timedelta(days=1)
    return out


def _energidataservice_get(dataset: str, start: datetime.date, end: datetime.date,
                            price_area_filter: bool = True) -> list[dict]:
    """Fetch all records from an Energidataservice dataset for the given date range."""
    base = f"https://api.energidataservice.dk/dataset/{dataset}"
    params: dict = {
        "start":  start.strftime("%Y-%m-%dT00:00"),
        "end":    end.strftime("%Y-%m-%dT23:00"),
        "sort":   "HourUTC asc",
        "limit":  100_000,
        "offset": 0,
    }
    if price_area_filter:
        params["filter"] = json.dumps({"PriceArea": [PRICE_AREA]})
    resp = requests.get(base, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json().get("records", [])


def _records_to_daily_array(records: list[dict], field: str,
                             dates: list[datetime.date],
                             fill_value: float = 0.0) -> np.ndarray:
    """Convert hourly records to (N_days, 24) float32 array aligned to dates."""
    date_to_idx = {d: i for i, d in enumerate(dates)}
    arr = np.full((len(dates), 24), fill_value, dtype=np.float32)

    for rec in records:
        ts_str = rec.get("HourUTC", "")
        val    = rec.get(field)
        if not ts_str or val is None:
            continue
        try:
            dt   = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            date = dt.date()
            hour = dt.hour
            if date in date_to_idx:
                arr[date_to_idx[date], hour] = float(val)
        except (ValueError, KeyError):
            continue

    # Fill any remaining zeros with daily mean (sparse records)
    for i in range(len(dates)):
        row = arr[i]
        valid = row[row != fill_value]
        if len(valid) > 0 and len(valid) < 24:
            arr[i, row == fill_value] = float(valid.mean())

    return arr


def download_spot_prices(dates: list[datetime.date]) -> np.ndarray:
    """Download Nord Pool DK1 day-ahead spot prices (EUR/MWh)."""
    print(f"  Downloading spot prices (Energidataservice / {PRICE_AREA})...")
    records = _energidataservice_get("Elspotprices", dates[0], dates[-1], price_area_filter=True)
    if len(records) < 100:
        raise RuntimeError(f"Too few spot price records ({len(records)}). Check connectivity.")
    arr = _records_to_daily_array(records, "SpotPriceEUR", dates, fill_value=50.0)
    print(f"    {len(records)} hourly records. Range: {arr.min():.1f}–{arr.max():.1f} EUR/MWh")
    return arr


def download_balancing_prices(dates: list[datetime.date]) -> tuple[np.ndarray, np.ndarray]:
    """Download DK1 balancing/regulation market prices (EUR/MWh).

    Up-regulation price:   V2G discharge revenue when grid needs more power.
    Down-regulation price: Smart charging cost when grid has surplus power.

    Sources tried in order:
      1. ENTSO-E Transparency Platform (if ENTSOE_TOKEN env var is set).
         Free token at: https://transparency.entsoe.eu/usrm/user/createPublicUser
         Returns real DK1 balancing energy prices (dataset: imbalance_prices_received).
      2. Energidataservice aFRR activation prices (Danish TSO, no auth).
         Frequency restoration reserve activation — a key V2G ancillary service.
      3. Fallback: spot ± 15% spread (reasonable approximation of DK1 avg spread).
    """
    print("  Downloading balancing market prices...")

    # --- Try ENTSO-E first if token is set ---
    entsoe_token = os.environ.get("ENTSOE_TOKEN", "")
    if entsoe_token:
        try:
            from entsoe import EntsoePandasClient
            import pandas as pd
            client = EntsoePandasClient(api_key=entsoe_token)
            # DK1 area code: 10YDK-1--------W
            area = "10YDK-1--------W"
            start_ts = pd.Timestamp(dates[0].isoformat(), tz="Europe/Copenhagen")
            end_ts   = pd.Timestamp(dates[-1].isoformat(), tz="Europe/Copenhagen") + pd.Timedelta(days=1)
            imb = client.query_imbalance_prices(area, start=start_ts, end=end_ts)
            if imb is not None and not imb.empty:
                # imb is a DataFrame with columns like 'Long', 'Short' (EUR/MWh)
                imb_utc = imb.tz_convert("UTC")
                up_arr  = np.full((len(dates), 24), 50.0, dtype=np.float32)
                dn_arr  = np.full((len(dates), 24), 50.0, dtype=np.float32)
                date_to_idx = {d: i for i, d in enumerate(dates)}
                long_col  = next((c for c in imb.columns if "Long"  in str(c) or "long"  in str(c)), None)
                short_col = next((c for c in imb.columns if "Short" in str(c) or "short" in str(c)), None)
                for ts, row in imb_utc.iterrows():
                    d, h = ts.date(), ts.hour
                    if d in date_to_idx:
                        i = date_to_idx[d]
                        if long_col and pd.notna(row[long_col]):
                            up_arr[i, h] = float(row[long_col])
                        if short_col and pd.notna(row[short_col]):
                            dn_arr[i, h] = float(row[short_col])
                print(f"    ENTSO-E imbalance prices: up {up_arr.mean():.1f}, dn {dn_arr.mean():.1f} EUR/MWh avg")
                return up_arr, dn_arr
        except Exception as e:
            print(f"    ENTSO-E failed ({type(e).__name__}). Trying Energidataservice aFRR...")

    # --- Try Energidataservice aFRR energy activation prices ---
    try:
        records = _energidataservice_get("AfrrEnergyActivation", dates[0], dates[-1],
                                         price_area_filter=False)
        if len(records) > 100:
            sample = records[0]
            up_f = next((k for k in sample if any(x in k for x in ["Up", "upward", "Up"])), None)
            dn_f = next((k for k in sample if any(x in k for x in ["Down", "downward", "Dn"])), None)
            if up_f and dn_f:
                up_arr = _records_to_daily_array(records, up_f, dates, 50.0)
                dn_arr = _records_to_daily_array(records, dn_f, dates, 50.0)
                print(f"    aFRR activation prices: up {up_arr.mean():.1f}, dn {dn_arr.mean():.1f} EUR/MWh")
                return up_arr, dn_arr
            raise ValueError(f"Unexpected field names: {list(sample.keys())[:6]}")
    except Exception as e:
        print(f"    aFRR data failed ({e}).")

    # --- Fallback: empirical spot spread ---
    # DK1 historical avg: up-regulation ~10-20% above spot, down ~10-15% below.
    # Using ±15% gives a conservative but directionally correct approximation.
    print("    Using fallback: spot ±15% spread (set ENTSOE_TOKEN for real balancing data).")
    spot = np.load(PRICES_FILE)
    return spot * 1.15, spot * 0.85


# ---------------------------------------------------------------------------
# Data caching
# ---------------------------------------------------------------------------

def prepare_data(force: bool = False) -> None:
    """Download all data and save to cache. Run once before experiments."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force and PRICES_FILE.exists() and SESSIONS_FILE.exists():
        print("Data already cached. Use --force to re-download.")
        return

    all_dates = _date_range(TRAIN_START, TEST_END)

    # Spot prices
    spot = download_spot_prices(all_dates)
    np.save(PRICES_FILE, spot)

    # Balancing prices (requires spot to be saved first for fallback)
    bal_up, bal_dn = download_balancing_prices(all_dates)
    np.save(BALANCING_UP_FILE, bal_up)
    np.save(BALANCING_DN_FILE, bal_dn)

    # EV sessions (synthetic, reproducible)
    print("  Generating synthetic EV fleet sessions...")
    sessions = generate_sessions(all_dates)
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, separators=(",", ":"))
    with open(DATES_FILE, "w") as f:
        json.dump([d.isoformat() for d in all_dates], f)
    with open(FLEET_FILE, "w") as f:
        json.dump(FLEET_SPECS, f, indent=2)

    print(f"\nCached {len(all_dates)} days:")
    for name, start, end in [("train", TRAIN_START, TRAIN_END),
                               ("val",   VAL_START,   VAL_END),
                               ("test",  TEST_START,  TEST_END)]:
        n = sum(1 for d in all_dates if start <= d <= end)
        print(f"  {name}: {start} – {end} ({n} days)")
    print(f"\nV2G-capable vehicles: {sum(1 for s in FLEET_SPECS if s['max_discharge_kw'] > 0)} / {NUM_VEHICLES}")
    print("Setup complete. Run 'uv run prepare_v2g.py --smoke' to verify.")


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[dict]], list[datetime.date]]:
    """Load all cached data. Returns (spot_prices, bal_up, bal_dn, sessions, dates)."""
    if not PRICES_FILE.exists():
        print("Data not found. Run: uv run prepare_v2g.py")
        sys.exit(1)

    spot    = np.load(PRICES_FILE)
    bal_up  = np.load(BALANCING_UP_FILE)  if BALANCING_UP_FILE.exists()  else spot * 1.10
    bal_dn  = np.load(BALANCING_DN_FILE)  if BALANCING_DN_FILE.exists()  else spot * 0.90
    with open(SESSIONS_FILE) as f:
        sessions = json.load(f)
    with open(DATES_FILE) as f:
        dates = [datetime.date.fromisoformat(s) for s in json.load(f)]

    return spot, bal_up, bal_dn, sessions, dates


# ---------------------------------------------------------------------------
# Day simulation (fixed physics — do not modify)
# ---------------------------------------------------------------------------

def simulate_day(
    sessions:      list[dict],
    schedule:      np.ndarray,    # (N_sessions, 24) kW; + charge, – discharge
    spot_prices:   np.ndarray,    # (24,) EUR/MWh
    bal_up_prices: np.ndarray | None = None,  # (24,) EUR/MWh — up-reg price
    bal_dn_prices: np.ndarray | None = None,  # (24,) EUR/MWh — down-reg price
) -> dict:
    """Simulate one day of V2G dispatch, return economics.

    Revenue model:
      - Discharging to grid: paid at max(spot, bal_up) per kWh exported
      - Charging from grid:  cost at min(spot, bal_dn) per kWh consumed
      - Battery degradation: SoC-stress-weighted per kWh discharged (see degradation_eur)
      - Departure violation: 3× peak-price penalty per kWh short of SOC_DEPARTURE_MIN

    Physical constraints enforced:
      - SoC ∈ [SOC_MIN, SOC_MAX] at all times
      - Power ≤ vehicle max_charge_kw / max_discharge_kw
      - Total site power ≤ GRID_LIMIT_KW per hour
    """
    n = len(sessions)
    if n == 0:
        return {k: 0.0 for k in ("net_revenue_eur", "gross_revenue_eur",
                                   "degradation_cost_eur", "violation_count",
                                   "total_charged_kwh", "total_discharged_kwh")}

    schedule     = np.asarray(schedule, dtype=np.float64)
    if schedule.shape != (n, 24):
        schedule = np.zeros((n, 24))

    if bal_up_prices is None:
        bal_up_prices = spot_prices * 1.10
    if bal_dn_prices is None:
        bal_dn_prices = spot_prices * 0.90

    bal_up_prices = np.asarray(bal_up_prices, dtype=np.float64)
    bal_dn_prices = np.asarray(bal_dn_prices, dtype=np.float64)

    gross_revenue    = 0.0
    total_degradation = 0.0
    violations        = 0
    total_charged     = 0.0
    total_discharged  = 0.0
    hourly_grid       = np.zeros(24)   # net site draw per hour (kW)

    for i, sess in enumerate(sessions):
        soc    = float(sess["soc_arrival"])
        cap    = float(sess["battery_capacity_kwh"])
        arr    = int(sess["arrival_hour"])
        dep    = int(sess["departure_hour"])
        max_ch = float(sess["max_charge_kw"])
        max_ds = float(sess["max_discharge_kw"])

        for h in range(24):
            if h < arr or h >= dep:
                continue

            raw_kw = float(schedule[i, h])

            if raw_kw >= 0:  # charging
                headroom = max(0.0, GRID_LIMIT_KW - hourly_grid[h])
                kw       = min(raw_kw, max_ch, headroom)
                soc_new  = min(soc + kw * CHARGE_EFF / cap, SOC_MAX)
                kwh_in   = (soc_new - soc) * cap          # kWh added to battery
                kw_drawn = kwh_in / CHARGE_EFF             # kWh drawn from grid

                # Cost: charged at the lower of spot or down-regulation price
                charge_price = min(float(spot_prices[h]), float(bal_dn_prices[h]))
                gross_revenue -= kw_drawn * charge_price / 1000.0
                hourly_grid[h] += kw_drawn
                total_charged  += kw_drawn
                soc = soc_new

            else:  # discharging (V2G)
                if max_ds <= 0:
                    continue
                kw      = min(-raw_kw, max_ds)
                soc_new = max(soc - kw / DISCHARGE_EFF / cap, SOC_MIN)
                kwh_out = (soc - soc_new) * cap            # kWh taken from battery
                kwh_exp = kwh_out * DISCHARGE_EFF          # kWh exported to grid

                # Revenue: discharged at the higher of spot or up-regulation price
                discharge_price = max(float(spot_prices[h]), float(bal_up_prices[h]))
                gross_revenue  += kwh_exp * discharge_price / 1000.0
                total_degradation += degradation_eur(kwh_out, soc, soc_new)
                hourly_grid[h]    -= kwh_exp
                total_discharged  += kwh_out
                soc = soc_new

        # Departure SoC check — 3× peak price penalty per kWh short
        if soc < SOC_DEPARTURE_MIN:
            short_kwh    = (SOC_DEPARTURE_MIN - soc) * cap
            peak_price   = float(spot_prices[arr:dep].max()) if dep > arr else spot_prices.max()
            gross_revenue -= short_kwh * peak_price / 1000.0 * 3.0
            violations   += 1

    net_revenue = gross_revenue - total_degradation
    return {
        "net_revenue_eur":      net_revenue,
        "gross_revenue_eur":    gross_revenue,
        "degradation_cost_eur": total_degradation,
        "violation_count":      violations,
        "total_charged_kwh":    total_charged,
        "total_discharged_kwh": total_discharged,
    }


# ---------------------------------------------------------------------------
# Evaluation harness (fixed ground truth — do not modify)
# ---------------------------------------------------------------------------

def evaluate_v2g(strategy_path: str = "strategy.py", split: str = "val") -> dict:
    """Run the full backtest of strategy.py against a fixed data split.

    Strategy interface (implement both in strategy.py):

        fit(
            train_prices:   np.ndarray,  # (N_train, 24) spot EUR/MWh
            train_sessions: list,        # N_train lists of session dicts
            *,
            train_bal_up:   np.ndarray,  # (N_train, 24) up-reg EUR/MWh
            train_bal_dn:   np.ndarray,  # (N_train, 24) down-reg EUR/MWh
        ) -> None

        plan_day(
            date:            datetime.date,
            price_history:   np.ndarray,   # (H, 24) spot prices, H ≤ HISTORY_DAYS
            session_history: list,         # H lists of past sessions
            vehicle_states:  list[dict],   # today's vehicles (arrival/departure/SoC)
            *,
            bal_up_history:  np.ndarray,   # (H, 24) up-reg prices
            bal_dn_history:  np.ndarray,   # (H, 24) down-reg prices
        ) -> np.ndarray                    # (N_vehicles, 24) kW schedule

        Optional attribute:
            last_price_forecast: np.ndarray | None  # (24,) → logged as diagnostic

    Primary metric (higher is better):
        val_revenue_per_kwh = net_revenue_eur / max(1, total_kwh_transacted)
    """
    spot, bal_up, bal_dn, sessions, dates = load_data()
    date_to_idx = {d: i for i, d in enumerate(dates)}

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
    train_idx    = [i for i, d in enumerate(dates) if d < eval_start]

    # Load strategy module
    spec = importlib.util.spec_from_file_location("strategy", strategy_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Fit on training data
    try:
        mod.fit(
            spot[train_idx],
            [sessions[i] for i in train_idx],
            train_bal_up=bal_up[train_idx],
            train_bal_dn=bal_dn[train_idx],
        )
    except TypeError:
        # Backward compat: strategy.fit() doesn't accept balancing kwargs
        mod.fit(spot[train_idx], [sessions[i] for i in train_idx])

    # Backtest
    t0 = time.time()
    total_net   = 0.0
    total_ch    = 0.0
    total_dis   = 0.0
    total_viol  = 0
    fc_rmse_vals = []

    for actual_idx, actual_date in zip(eval_indices, eval_dates):
        h0 = max(0, actual_idx - HISTORY_DAYS)
        ph  = spot[h0:actual_idx]
        sh  = sessions[h0:actual_idx]
        buh = bal_up[h0:actual_idx]
        bdh = bal_dn[h0:actual_idx]
        today_sess  = sessions[actual_idx]
        today_spot  = spot[actual_idx]
        today_bup   = bal_up[actual_idx]
        today_bdn   = bal_dn[actual_idx]

        try:
            schedule = mod.plan_day(
                date=actual_date,
                price_history=ph,
                session_history=sh,
                vehicle_states=today_sess,
                bal_up_history=buh,
                bal_dn_history=bdh,
            )
        except TypeError:
            # Backward compat: strategy doesn't accept balancing kwargs
            schedule = mod.plan_day(
                date=actual_date,
                price_history=ph,
                session_history=sh,
                vehicle_states=today_sess,
            )

        if schedule is None:
            schedule = np.zeros((len(today_sess), 24))
        schedule = np.asarray(schedule, dtype=np.float64)
        if schedule.shape != (len(today_sess), 24):
            schedule = np.zeros((len(today_sess), 24))

        fc = getattr(mod, "last_price_forecast", None)
        if fc is not None:
            fc = np.asarray(fc, dtype=np.float64)
            if fc.shape == (24,):
                fc_rmse_vals.append(float(np.sqrt(np.mean((fc - today_spot.astype(np.float64)) ** 2))))

        result = simulate_day(today_sess, schedule, today_spot, today_bup, today_bdn)
        total_net  += result["net_revenue_eur"]
        total_ch   += result["total_charged_kwh"]
        total_dis  += result["total_discharged_kwh"]
        total_viol += result["violation_count"]

    elapsed = time.time() - t0
    kwh_total = total_ch + total_dis
    val_revenue_per_kwh = total_net / max(1.0, kwh_total)

    out = {
        "val_revenue_per_kwh":        val_revenue_per_kwh,
        "total_net_revenue_eur":       total_net,
        "total_charged_kwh":           total_ch,
        "total_discharged_kwh":        total_dis,
        "total_constraint_violations": total_viol,
        "num_eval_days":               len(eval_dates),
        "total_seconds":               elapsed,
    }
    if fc_rmse_vals:
        out["forecast_rmse_eur_mwh"] = float(np.mean(fc_rmse_vals))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="V2G fleet autoresearch — data prep & eval")
    parser.add_argument("--eval",     action="store_true")
    parser.add_argument("--smoke",    action="store_true")
    parser.add_argument("--test",     action="store_true")
    parser.add_argument("--force",    action="store_true", help="Re-download data")
    parser.add_argument("--strategy", default="strategy.py")
    args = parser.parse_args()

    if args.eval or args.smoke or args.test:
        split = "test" if args.test else ("smoke" if args.smoke else "val")
        print(f"Running backtest on '{split}' split...")
        m = evaluate_v2g(strategy_path=args.strategy, split=split)
        print("---")
        print(f"val_revenue_per_kwh:          {m['val_revenue_per_kwh']:.6f}")
        print(f"total_net_revenue_eur:         {m['total_net_revenue_eur']:.2f}")
        print(f"total_charged_kwh:             {m['total_charged_kwh']:.1f}")
        print(f"total_discharged_kwh:          {m['total_discharged_kwh']:.1f}")
        print(f"total_constraint_violations:   {m['total_constraint_violations']}")
        print(f"num_eval_days:                 {m['num_eval_days']}")
        print(f"total_seconds:                 {m['total_seconds']:.1f}")
        if "forecast_rmse_eur_mwh" in m:
            print(f"forecast_rmse_eur_mwh:         {m['forecast_rmse_eur_mwh']:.2f}")
        return

    prepare_data(force=args.force)


if __name__ == "__main__":
    main()
