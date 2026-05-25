# Contributing to autoresearch_v2g

Thanks for your interest! This project is deliberately minimal — contributions
that keep it small, focused, and correct are most welcome.

---

## What we welcome

| Type | Examples |
|------|---------|
| **Better strategies** | Improved `strategy.py` baselines (LP dispatch, ML forecasters, RL policies) |
| **Better data** | New data sources, additional price zones, real EV session datasets |
| **Better degradation models** | More accurate battery physics, temperature effects |
| **New markets** | Adapting `prepare_v2g.py` for other price zones (NO1, SE3, DE, etc.) |
| **Bug fixes** | Correctness issues in the harness or simulation |
| **Documentation** | Clearer explanations, better examples |
| **Analysis notebooks** | Better visualizations, strategy comparison tools |

## What we do NOT want in PRs

- Changes to `prepare_v2g.py` that modify the **evaluation harness** (`evaluate_v2g`, `simulate_day`) without a very strong reason — this is the fixed ground truth
- New dependencies unless clearly justified
- Complex abstractions — three simple lines beat a premature abstraction
- Features not used by the core experiment loop

---

## How to contribute

### 1. Fork and branch

```bash
git clone https://github.com/thesairam/autoresearch_v2g
cd autoresearch_v2g
git checkout -b feat/your-feature-name
```

### 2. Set up

```bash
uv sync
uv run prepare_v2g.py          # download data (~1 min, one-time)
uv run prepare_v2g.py --smoke  # verify baseline works
```

### 3. Make your changes

- If you're contributing a new **strategy**, replace `strategy.py` with your implementation
  and show the `val_revenue_per_kwh` improvement over the baseline in your PR description.
- If you're contributing a **data source**, add it to `prepare_v2g.py` and document
  what it provides and why it's reliable.
- If you're fixing a **bug**, include a short description of what was wrong and how to verify the fix.

### 4. Verify

All PRs must pass:

```bash
uv run prepare_v2g.py --smoke   # must complete without error
uv run prepare_v2g.py --eval    # must print a valid val_revenue_per_kwh
```

If your change modifies data or fleet constants, re-run `--force` first:

```bash
uv run prepare_v2g.py --force && uv run prepare_v2g.py --eval
```

### 5. Open a PR

- **Title**: short and specific (e.g. `Add LP dispatch baseline`, `Add ENTSO-E price source`)
- **Description**: what you changed and why, with before/after metrics if applicable
- **Keep it small**: one logical change per PR

---

## Data sources

### Adding a new price zone

The harness supports any hourly day-ahead price series in EUR/MWh.
To add a new zone (e.g., German DE-LU):
1. Add a download function to `prepare_v2g.py`
2. Add a `--zone` CLI flag
3. Keep the default as DK1 so existing experiments aren't broken

### ENTSO-E Transparency Platform (optional, free token required)

If you want broader European market data, you can use the ENTSO-E API
as an alternative price source. Set your free token:

```bash
export ENTSOE_TOKEN=your_token_here  # get free at transparency.entsoe.eu
uv run prepare_v2g.py --source entsoe
```

Note: the default source (Energidataservice / Danish TSO) requires no auth.

### Real EV session data (ACN-Data)

To use real Caltech EV charging sessions instead of synthetic:
1. Register free at [ev.caltech.edu](https://ev.caltech.edu)
2. `pip install acnportal`
3. Set `ACN_TOKEN=your_token_here`
4. Run `uv run prepare_v2g.py --ev-source acn`

---

## Code style

- No type annotation enforcement, but add them where they help readability
- Keep `prepare_v2g.py` self-contained (no imports outside stdlib + numpy + requests)
- Strategy files can use any package already in `pyproject.toml`
- No comments explaining what code does — only comments explaining **why** if it's non-obvious
- Prefer editing existing files over creating new ones

---

## Conduct

Be respectful. This is a research project. Disagreements about modeling choices
are fine and expected — back them with data or citations.
