"""
Non-parametric threshold analysis
==================================
Robustness check for gss_threshold.py and gss_finrela_worst.py.

Instead of fitting a logistic regression of P(bad outcome) on log(income),
this script computes the weighted proportion of bad outcomes within each
distinct coninc bracket, then linearly interpolates between adjacent
brackets to find the income at which that proportion crosses 25%.

This is fully non-parametric: no functional form assumption on the
income-wellbeing relationship, no log transformation of income, and no
sensitivity to how brackets are coded.

Outputs comparison charts (non-parametric vs. logistic regression) to
output/non_parametric/, reading the logistic results from:
  output/threshold/threshold_adjusted.csv
  output/finrela_worst/threshold_adjusted.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

warnings.filterwarnings("ignore")

DATA_PATH = "data/GSS.xlsx"
OUT_DIR   = "output/non_parametric"
THRESHOLD = 0.25

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------

print("Loading data…")
df = pd.read_excel(DATA_PATH, sheet_name="Data")
df.columns = df.columns.str.strip().str.lower()
df["year"]   = pd.to_numeric(df["year"],   errors="coerce").astype("Int64")
df["weight"] = pd.to_numeric(df["wtssps"], errors="coerce")
df["coninc"] = pd.to_numeric(df["coninc"], errors="coerce")
df.loc[df["coninc"] <= 0, "coninc"] = np.nan

def parse_hompop(val):
    s = str(val).strip()
    if s.startswith("14"):
        return 14.0
    try:
        return float(s)
    except ValueError:
        return np.nan

df["hompop"]    = df["hompop"].apply(parse_hompop)
df.loc[df["hompop"] <= 0, "hompop"] = np.nan
df["coninc_eq"] = df["coninc"] / np.sqrt(df["hompop"])

# satfin outcome
VALID_SATFIN = {"Pretty well satisfied", "More or less satisfied", "Not satisfied at all"}
df["satfin_clean"] = df["satfin"].where(df["satfin"].isin(VALID_SATFIN))
df["dissatisfied"] = (df["satfin_clean"] == "Not satisfied at all").astype(float)
df.loc[df["satfin_clean"].isna(), "dissatisfied"] = np.nan

# finrela outcome
VALID_FINRELA = {"FAR ABOVE AVERAGE", "ABOVE AVERAGE", "AVERAGE",
                 "BELOW AVERAGE", "FAR BELOW AVERAGE"}
df["finrela_clean"] = df["finrela"].where(df["finrela"].isin(VALID_FINRELA))
df["below_avg"]     = df["finrela_clean"].isin({"BELOW AVERAGE", "FAR BELOW AVERAGE"}).astype(float)
df.loc[df["finrela_clean"].isna(), "below_avg"] = np.nan

# ---------------------------------------------------------------------------
# 2. Non-parametric threshold
#
# For each year:
#   a. Group respondents by their coninc_eq bracket (distinct values).
#   b. Compute weighted proportion with bad outcome per bracket.
#   c. Sort brackets by income.
#   d. Find the first adjacent pair where proportion descends through 25%.
#   e. Linear interpolation between those two bracket midpoints.
# ---------------------------------------------------------------------------

def weighted_median(vals, wts):
    order = np.argsort(vals)
    vals, wts = vals[order], wts[order]
    cumw = np.cumsum(wts)
    return vals[np.searchsorted(cumw, cumw[-1] / 2.0)]

def nonparametric_threshold(sub, income_col, outcome_col, target=THRESHOLD):
    sub = sub.dropna(subset=[income_col, outcome_col, "weight"])
    sub = sub[sub["weight"] > 0]
    if len(sub) < 10:
        return np.nan

    # Weighted proportion of bad outcome per bracket
    incomes, props = [], []
    for inc_val, group in sub.groupby(income_col, observed=True):
        prop = np.average(group[outcome_col], weights=group["weight"])
        incomes.append(inc_val)
        props.append(prop)

    # Sort ascending by income
    order   = np.argsort(incomes)
    incomes = np.array(incomes)[order]
    props   = np.array(props)[order]

    # Need at least one bracket above and one below the target
    if props[0] < target or props[-1] > target:
        return np.nan

    # First downward crossing through target
    for i in range(len(props) - 1):
        p0, p1 = props[i], props[i + 1]
        if p0 >= target >= p1 and p0 != p1:
            t = (target - p0) / (p1 - p0)
            return incomes[i] + t * (incomes[i + 1] - incomes[i])

    return np.nan

def run_analysis(df_in, income_col, outcome_col):
    df_c  = df_in.dropna(subset=[income_col, outcome_col, "weight"]).copy()
    years = sorted(df_c["year"].unique())
    rows  = []
    for yr in years:
        sub    = df_c[df_c["year"] == yr]
        thresh = nonparametric_threshold(sub, income_col, outcome_col)
        vals   = sub[income_col].dropna().values
        wts    = sub.loc[sub[income_col].notna(), "weight"].values
        rows.append({
            "year":             int(yr),
            "threshold_income": thresh,
            "median_income":    weighted_median(vals, wts),
            "n":                len(sub),
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 3. Smoothing (same ±2 calendar year window as logistic scripts)
# ---------------------------------------------------------------------------

def add_smooth(res, half_window=2):
    r = res.copy().sort_values("year").reset_index(drop=True)
    for col in ["threshold_income", "median_income"]:
        smoothed = []
        for yr in r["year"]:
            mask = (r["year"] >= yr - half_window) & (r["year"] <= yr + half_window)
            smoothed.append(r.loc[mask, col].mean())
        r[f"{col}_smooth"] = smoothed
    return r

# ---------------------------------------------------------------------------
# 4. Run both analyses
# ---------------------------------------------------------------------------

df_satfin  = df.dropna(subset=["year", "satfin_clean",  "weight"]).copy()
df_satfin  = df_satfin[df_satfin["weight"] > 0]

df_finrela = df.dropna(subset=["year", "finrela_clean", "weight"]).copy()
df_finrela = df_finrela[df_finrela["weight"] > 0]

print("Running satfin (HH-adjusted)…")
res_satfin  = add_smooth(run_analysis(df_satfin,  "coninc_eq", "dissatisfied"))

print("Running finrela (HH-adjusted)…")
res_finrela = add_smooth(run_analysis(df_finrela, "coninc_eq", "below_avg"))

res_satfin.to_csv( f"{OUT_DIR}/satfin_threshold_adjusted.csv",  index=False, float_format="%.1f")
res_finrela.to_csv(f"{OUT_DIR}/finrela_threshold_adjusted.csv", index=False, float_format="%.1f")
print(f"Saved CSVs to {OUT_DIR}/")

# ---------------------------------------------------------------------------
# 5. Load logistic regression results for comparison
# ---------------------------------------------------------------------------

logit_satfin  = add_smooth(pd.read_csv("output/threshold/threshold_adjusted.csv"))
logit_finrela = add_smooth(pd.read_csv("output/finrela_worst/threshold_adjusted.csv"))

# ---------------------------------------------------------------------------
# 6. Comparison plots
# ---------------------------------------------------------------------------

def plot_comparison(res_np, res_logit, title, filename):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw non-parametric dots
    ax.plot(res_np["year"], res_np["threshold_income"] / 1000,
            color="#d7191c", lw=0, marker="o", ms=3, alpha=0.25)

    # Smoothed non-parametric
    ax.plot(res_np["year"], res_np["threshold_income_smooth"] / 1000,
            color="#d7191c", lw=2.4,
            label="Non-parametric threshold  (5-yr rolling avg)")

    # Smoothed logistic regression
    ax.plot(res_logit["year"], res_logit["threshold_income_smooth"] / 1000,
            color="#e08214", lw=2.0, ls="--",
            label="Logistic regression threshold  (5-yr rolling avg)")

    # Weighted median income
    ax.plot(res_np["year"], res_np["median_income_smooth"] / 1000,
            color="#2c7bb6", lw=2.2,
            label="Weighted median income  (5-yr rolling avg)")

    # Shade gap between non-parametric threshold and median
    t = res_np["threshold_income_smooth"]
    m = res_np["median_income_smooth"]
    ax.fill_between(res_np["year"], m / 1000, t / 1000,
                    where=t > m,  color="#d7191c", alpha=0.10,
                    label="Median below NP threshold (at-risk)")
    ax.fill_between(res_np["year"], m / 1000, t / 1000,
                    where=t <= m, color="#1a9641", alpha=0.10,
                    label="Median above NP threshold (comfortable)")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Constant dollars (thousands)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
    ax.legend(fontsize=9.5, framealpha=0.9, loc="upper left")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_xlim(res_np["year"].min() - 1, res_np["year"].max() + 1)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


print("\nGenerating plots…")
plot_comparison(
    res_satfin, logit_satfin,
    "Robustness check: non-parametric vs. logistic regression  —  satfin\n"
    "Income at 25% P(not satisfied at all)  (faint dots = raw NP values)",
    f"{OUT_DIR}/satfin_comparison.png"
)
plot_comparison(
    res_finrela, logit_finrela,
    "Robustness check: non-parametric vs. logistic regression  —  finrela\n"
    "Income at 25% P(below average)  (faint dots = raw NP values)",
    f"{OUT_DIR}/finrela_comparison.png"
)
print("\nDone.")
