"""
GSS Happiness Simulation: 2% Annual Income Growth
===================================================
Freezes the income→happiness curve at 1972–1975 levels, then simulates
what steady 2% real annual income growth would do to aggregate happiness,
starting from the 1972 income distribution.

Within-bracket income distributions
-------------------------------------
The GSS income question is categorical. `coninc` is the bracket midpoint in
constant dollars. To build a continuous income distribution we need to assume
a distribution within each bracket:

  Interior brackets  → Uniform(lower_bound, upper_bound)
  Top bracket        → Pareto(shape=α, scale=lower_bound)
                       i.e. income = lower_bound * U^(-1/α), U ~ Uniform(0,1)

Pareto is standard for income tails; we test α ∈ {1.5, 2.0, 3.0} to show
sensitivity. α=2 (heavy tail, infinite variance) is the base case.

The top-bracket lower bound is the midpoint between the two highest midpoints.

Outputs: output/simulation/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

OUT_DIR      = "output/simulation"
DATA_PATH    = "data/GSS.xlsx"
CURVE_YEARS  = [1972, 1973, 1974, 1975]   # years used to fit the happiness curve
SIM_START    = 1972
SIM_END      = 2024
GROWTH_RATE  = 0.02                         # real annual income growth
PARETO_ALPHAS = [1.5, 2.0, 3.0]            # sensitivity range for top-bracket shape

import os; os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and clean data
# ---------------------------------------------------------------------------

print("Loading data…")
df = pd.read_excel(DATA_PATH, sheet_name="Data")
df.columns = df.columns.str.strip().str.lower()
df["year"]   = pd.to_numeric(df["year"],   errors="coerce").astype("Int64")
df["weight"] = pd.to_numeric(df["wtssps"], errors="coerce")
df["coninc"] = pd.to_numeric(df["coninc"], errors="coerce")
df.loc[df["coninc"] <= 0, "coninc"] = np.nan

VALID_HAPPY = {"Very happy", "Pretty happy", "Not too happy"}
df["happy_clean"] = df["happy"].where(df["happy"].isin(VALID_HAPPY))
df["very_happy"]  = (df["happy_clean"] == "Very happy").astype(float)

df_clean = df.dropna(subset=["year", "coninc", "happy_clean", "weight"]).copy()
df_clean  = df_clean[df_clean["weight"] > 0].copy()

# ---------------------------------------------------------------------------
# 2. Fit happiness curve on pooled 1972–1975 data
# ---------------------------------------------------------------------------

print(f"Fitting happiness curve on {CURVE_YEARS}…")
pool = df_clean[df_clean["year"].isin(CURVE_YEARS)].copy()
X_pool = np.log(pool["coninc"].values).reshape(-1, 1)
y_pool = pool["very_happy"].values
w_pool = pool["weight"].values

curve = LogisticRegression(max_iter=1000, solver="lbfgs")
curve.fit(X_pool, y_pool, sample_weight=w_pool)

intercept = curve.intercept_[0]
slope     = curve.coef_[0][0]
print(f"  Curve: log-odds = {intercept:.3f} + {slope:.3f} × log(income)")
print(f"  Implied: doubling income changes log-odds by {slope * np.log(2):.3f} "
      f"({slope * np.log(2) / 4 * 100:.1f} pp near p=0.5)")

# ---------------------------------------------------------------------------
# 3. Build continuous income distribution from 1972 respondents
# ---------------------------------------------------------------------------

print("\nBuilding continuous 1972 income distribution…")
df_72 = df_clean[df_clean["year"] == 1972].copy().reset_index(drop=True)

# Bracket midpoints unique to 1972, sorted
midpoints = np.array(sorted(df_72["coninc"].unique()))
print(f"  {len(midpoints)} brackets, midpoints: {midpoints.astype(int).tolist()}")

# Infer bracket boundaries: midpoint between consecutive midpoints
boundaries = np.concatenate([
    [0.0],
    (midpoints[:-1] + midpoints[1:]) / 2,
    [np.inf],
])
top_lower = boundaries[-2]   # lower bound of the top bracket
print(f"  Top-bracket lower bound: ${top_lower:,.0f}")
print(f"  Share in top bracket (weighted): "
      f"{(df_72[df_72['coninc'] == midpoints[-1]]['weight'].sum() / df_72['weight'].sum()):.1%}")


def sample_continuous_incomes(df_yr, midpoints, boundaries, alpha, rng):
    """
    For each respondent, draw a continuous income from within their bracket.
    Interior brackets: Uniform(lower, upper).
    Top bracket:       Pareto(alpha, scale=lower_bound).
    """
    incomes = np.empty(len(df_yr))
    top_idx = len(midpoints) - 1
    top_lower = boundaries[-2]

    for i, (_, row) in enumerate(df_yr.iterrows()):
        m = row["coninc"]
        # find which bracket this midpoint belongs to
        k = np.searchsorted(midpoints, m, side="left")
        lo, hi = boundaries[k], boundaries[k + 1]

        if k == top_idx:
            # Pareto draw: income = top_lower * u^(-1/alpha), u ~ U(0,1)
            u = rng.uniform(0, 1)
            inc = top_lower * (u ** (-1.0 / alpha))
            # cap at 50× lower bound to prevent infinite-variance blowup
            inc = min(inc, 50 * top_lower)
        else:
            inc = rng.uniform(lo, hi)
        incomes[i] = inc

    return incomes


# ---------------------------------------------------------------------------
# 4. Simulate 2% annual growth for each Pareto α
# ---------------------------------------------------------------------------

print("\nRunning simulations…")

sim_years = np.arange(SIM_START, SIM_END + 1)

def simulate_growth(df_yr, midpoints, boundaries, alpha, rng):
    """Return dict year → predicted P(very happy) under 2% annual growth."""
    incomes_0 = sample_continuous_incomes(df_yr, midpoints, boundaries, alpha, rng)
    weights   = df_yr["weight"].values
    results   = {}
    for yr in sim_years:
        t = yr - SIM_START
        incomes_t = incomes_0 * (1 + GROWTH_RATE) ** t
        # clamp to avoid log(0)
        incomes_t = np.maximum(incomes_t, 1.0)
        log_inc = np.log(incomes_t).reshape(-1, 1)
        probs   = curve.predict_proba(log_inc)[:, 1]
        results[yr] = np.average(probs, weights=weights)
    return results

sim_results = {}
for alpha in PARETO_ALPHAS:
    print(f"  α = {alpha}…")
    sim_results[alpha] = simulate_growth(df_72, midpoints, boundaries, alpha, rng)

# Also record simulated median income by year (using α=2.0 for reference)
incomes_ref = sample_continuous_incomes(df_72, midpoints, boundaries, 2.0, rng)
weights_72  = df_72["weight"].values
def weighted_median(vals, wts):
    order = np.argsort(vals)
    vals, wts = vals[order], wts[order]
    cumw = np.cumsum(wts)
    return vals[np.searchsorted(cumw, cumw[-1] / 2.0)]

sim_income = {
    yr: {
        "mean":   np.average(incomes_ref * (1.02 ** (yr - SIM_START)), weights=weights_72),
        "median": weighted_median(incomes_ref * (1.02 ** (yr - SIM_START)), weights_72),
    }
    for yr in sim_years
}

# ---------------------------------------------------------------------------
# 5. Actual happiness rates for comparison
# ---------------------------------------------------------------------------

actual = (
    df_clean
    .groupby("year")
    .apply(lambda g: np.average(g["very_happy"], weights=g["weight"]))
    .reset_index(name="actual_happiness")
)

# ---------------------------------------------------------------------------
# 6. Save results table
# ---------------------------------------------------------------------------

out_rows = []
for yr in sim_years:
    row = {"year": yr,
           "simulated_income_mean":   sim_income[yr]["mean"],
           "simulated_income_median": sim_income[yr]["median"]}
    for alpha in PARETO_ALPHAS:
        row[f"sim_happiness_alpha{alpha}"] = sim_results[alpha][yr]
    out_rows.append(row)

results_df = pd.DataFrame(out_rows)
results_df = results_df.merge(actual.rename(columns={"year": "year"}),
                              left_on="year", right_on="year", how="left")
results_df.to_csv(f"{OUT_DIR}/results_simulation.csv", index=False, float_format="%.4f")
print(f"\nSaved {OUT_DIR}/results_simulation.csv")
print(results_df[["year",
                   f"sim_happiness_alpha2.0",
                   "actual_happiness",
                   "simulated_income_median"]].to_string(index=False,
                   float_format=lambda x: f"{x:.3f}"))

# ---------------------------------------------------------------------------
# 7. Plot 1: Simulated vs actual happiness
# ---------------------------------------------------------------------------

filename = f"{OUT_DIR}/simulation_vs_actual.png"
fig, ax  = plt.subplots(figsize=(12, 6))

# Actual survey data
ax.plot(actual["year"], actual["actual_happiness"] * 100,
        color="#2c7bb6", lw=2.2, marker="o", ms=4, zorder=5,
        label="Actual 'very happy' rate")

# Base simulation (α=2)
base = [sim_results[2.0][yr] * 100 for yr in sim_years]
ax.plot(sim_years, base,
        color="#d7191c", lw=2.2, ls="-",
        label="Simulated: 2% annual income growth,\nfrozen 1972–75 happiness curve  (Pareto α=2)")

# Sensitivity band (α=1.5 to α=3)
lo_band = [sim_results[3.0][yr] * 100 for yr in sim_years]   # α=3 = thinner tail = lower
hi_band = [sim_results[1.5][yr] * 100 for yr in sim_years]   # α=1.5 = heavier tail = higher
ax.fill_between(sim_years, lo_band, hi_band,
                color="#d7191c", alpha=0.15,
                label="Sensitivity: Pareto α ∈ {1.5, 3.0}")

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Share 'Very happy' (%)", fontsize=12)
ax.set_title("What would 2% annual income growth have done to happiness?\n"
             "Frozen 1972–75 happiness curve applied to growing 1972 income distribution",
             fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(axis="y", ls="--", alpha=0.4)
ax.set_xlim(SIM_START - 1, SIM_END + 1)
plt.tight_layout()
plt.savefig(filename, dpi=150)
plt.close()
print(f"\nSaved {filename}")

# ---------------------------------------------------------------------------
# 8. Plot 2: Simulated income growth (sanity check)
# ---------------------------------------------------------------------------

filename = f"{OUT_DIR}/simulated_income_growth.png"
fig, ax  = plt.subplots(figsize=(11, 5))

means   = [sim_income[yr]["mean"]   / 1000 for yr in sim_years]
medians = [sim_income[yr]["median"] / 1000 for yr in sim_years]

ax.plot(sim_years, means,   color="#2c7bb6", lw=2.2, label="Simulated mean income")
ax.plot(sim_years, medians, color="#2c7bb6", lw=2.2, ls="--", label="Simulated median income")

# Overlay actual GSS means for comparison
actual_income = (
    df_clean
    .groupby("year")
    .apply(lambda g: np.average(g["coninc"], weights=g["weight"]))
    .reset_index(name="mean_income")
)
ax.plot(actual_income["year"], actual_income["mean_income"] / 1000,
        color="#d7191c", lw=1.8, marker="s", ms=3, alpha=0.8,
        label="Actual GSS mean income\n(top-coded — understates true growth)")

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Constant dollars (thousands)", fontsize=12)
ax.set_title("Simulated income under 2% annual growth vs. actual GSS income\n"
             "(gap illustrates top-coding problem in GSS data)",
             fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(axis="y", ls="--", alpha=0.4)
ax.set_xlim(SIM_START - 1, SIM_END + 1)
plt.tight_layout()
plt.savefig(filename, dpi=150)
plt.close()
print(f"Saved {filename}")

# ---------------------------------------------------------------------------
# 9. Plot 3: Simulation vs frozen-expectations counterfactual
#    The gap between the two reveals the cost of top-coding.
#
#    Simulation          : frozen 1972-75 curve + 2% growth on 1972 population
#                          (income grows as it should in reality)
#    Frozen expectations : frozen 1972 curve + actual observed GSS incomes
#                          (income growth attenuated by top-coding)
#
#    Note: the two curves use slightly different baseline curves
#    (1972-75 pool vs 1972 only) so they won't start at exactly the same
#    point, but the gap over time is the meaningful comparison.
# ---------------------------------------------------------------------------

print("\nGenerating comparison plot…")
frozen_exp = pd.read_csv("output/happiness/results_unadjusted.csv")

filename = f"{OUT_DIR}/simulation_vs_frozen_expectations.png"
fig, ax  = plt.subplots(figsize=(12, 6))

# Actual happiness
ax.plot(actual["year"], actual["actual_happiness"] * 100,
        color="#2c7bb6", lw=2.2, marker="o", ms=4, zorder=5,
        label="Actual 'very happy' rate")

# Simulation: 2% growth, Pareto α=2 base + sensitivity band
base    = [sim_results[2.0][yr] * 100 for yr in sim_years]
lo_band = [sim_results[3.0][yr] * 100 for yr in sim_years]
hi_band = [sim_results[1.5][yr] * 100 for yr in sim_years]
ax.plot(sim_years, base, color="#d7191c", lw=2.2,
        label="Simulated: 2% annual growth, frozen 1972–75 curve  (Pareto α=2)")
ax.fill_between(sim_years, lo_band, hi_band,
                color="#d7191c", alpha=0.15, label="Sensitivity: Pareto α ∈ {1.5, 3.0}")

# Frozen-expectations counterfactual (actual top-coded GSS incomes)
ax.plot(frozen_exp["year"], frozen_exp["frozen_expectations_happiness"] * 100,
        color="#1a9641", lw=2.2, ls="--", marker="^", ms=4,
        label="Frozen 1972 curve + actual GSS incomes\n(top-coding attenuates income growth)")

# Shade the gap between the two counterfactuals
merged = pd.DataFrame({"year": sim_years,
                        "sim": base}).merge(
         frozen_exp[["year","frozen_expectations_happiness"]],
         on="year", how="left")
ax.fill_between(merged["year"],
                merged["frozen_expectations_happiness"] * 100,
                merged["sim"],
                color="#f0a500", alpha=0.2,
                label="Gap = effect of top-coding on measured income growth")

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Share 'Very happy' (%)", fontsize=12)
ax.set_title("How much does GSS income top-coding matter?\n"
             "Comparing simulated 2% growth to frozen-expectations counterfactual",
             fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.legend(fontsize=9.5, framealpha=0.9, loc="upper right")
ax.grid(axis="y", ls="--", alpha=0.4)
ax.set_xlim(SIM_START - 1, SIM_END + 1)
plt.tight_layout()
plt.savefig(filename, dpi=150)
plt.close()
print(f"Saved {filename}")

print("\nDone.")
