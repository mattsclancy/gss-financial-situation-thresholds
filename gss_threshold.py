"""
GSS Financial Dissatisfaction Threshold Analysis
=================================================
For each year, fits a logistic regression of P(not satisfied at all) on
log(income), then solves analytically for the income level at which there
is a 25% probability of reporting financial dissatisfaction.

That threshold income is plotted alongside the weighted mean and median
actual respondent income, so we can see how the "danger zone" boundary
has moved relative to where people actually stand.

Both unadjusted (coninc) and HH-adjusted (coninc / sqrt(hompop)) variants
are shown in a side-by-side comparison.

Outputs: output/threshold/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

DATA_PATH = "data/GSS.xlsx"
OUT_DIR   = "output/threshold"
THRESHOLD = 0.25    # P(not satisfied at all) at which we compute the income threshold

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------

print("Loading data…")
try:
    df = pd.read_excel(DATA_PATH, sheet_name="Data")
except ValueError as e:
    raise SystemExit(
        f"Could not read sheet 'Data' from {DATA_PATH}. "
        f"Verify the sheet name matches exactly. Original error: {e}"
    )
df.columns = df.columns.str.strip().str.lower()

REQUIRED_COLS = {"year", "wtssps", "coninc", "hompop", "satfin"}
missing = REQUIRED_COLS - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns in {DATA_PATH}: {missing}")
df["year"]   = pd.to_numeric(df["year"],   errors="coerce").astype("Int64")
df["weight"] = pd.to_numeric(df["wtssps"], errors="coerce")
df["coninc"] = pd.to_numeric(df["coninc"], errors="coerce")
df.loc[df["coninc"] <= 0, "coninc"] = np.nan

def parse_hompop(val):
    s = str(val).strip()
    if s.startswith("14"):   # GSS top-codes "14 or more" persons in household as "14+"
        return 14.0
    try:
        return float(s)
    except ValueError:
        return np.nan

df["hompop"]    = df["hompop"].apply(parse_hompop)
df.loc[df["hompop"] <= 0, "hompop"] = np.nan
df["coninc_eq"] = df["coninc"] / np.sqrt(df["hompop"])

_SATFIN_MAP = {
    "pretty well satisfied": "Pretty well satisfied",
    "more or less satisfied": "More or less satisfied",
    "not satisfied at all": "Not satisfied at all",
}
_satfin_norm = df["satfin"].astype(str).str.strip().str.lower()
unknown_satfin = {v for v in _satfin_norm.unique()
                  if v not in _SATFIN_MAP and v not in {"nan", ""}}
if unknown_satfin:
    print(f"  Warning: unrecognized satfin values (will be dropped): {unknown_satfin}")
df["satfin_clean"] = _satfin_norm.map(_SATFIN_MAP)
df["dissatisfied"] = (df["satfin_clean"] == "Not satisfied at all").astype(float)

df_base = df.dropna(subset=["year", "satfin_clean", "weight"]).copy()
df_base  = df_base[df_base["weight"] > 0].copy()

# ---------------------------------------------------------------------------
# 2. Threshold calculation
#
# Logistic model: log-odds = b0 + b1 * log(income)
# Solve for income at P = THRESHOLD:
#   log(p / (1-p)) = b0 + b1 * log(income)
#   log(income)    = (log(p/(1-p)) - b0) / b1
#   income         = exp((log(p/(1-p)) - b0) / b1)
# ---------------------------------------------------------------------------

target_logodds = np.log(THRESHOLD / (1 - THRESHOLD))   # ≈ -1.099 for p=0.25
print(f"Target log-odds for p={THRESHOLD}: {target_logodds:.4f}")

def weighted_median(vals, wts):
    order = np.argsort(vals)
    vals, wts = vals[order], wts[order]
    cumw = np.cumsum(wts)
    return vals[np.searchsorted(cumw, cumw[-1] / 2.0)]

def run_threshold_analysis(df_in, income_col):
    df_c   = df_in.dropna(subset=[income_col]).copy()
    years  = sorted(df_c["year"].unique())
    rows   = []

    for yr in years:
        sub = df_c[df_c["year"] == yr]
        X   = np.log(sub[income_col].values).reshape(-1, 1)
        y   = sub["dissatisfied"].values
        w   = sub["weight"].values

        if len(np.unique(y)) < 2 or len(sub) < 10:
            continue

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X, y, sample_weight=w)

        b0 = clf.intercept_[0]
        b1 = clf.coef_[0][0]

        # b1 must be negative (higher income → lower dissatisfaction) and
        # the solution must be positive income
        if b1 >= 0:
            print(f"  Warning: year {yr} has b1={b1:.4f} >= 0 (unexpected sign); threshold set to NaN")
            threshold_income = np.nan
        else:
            log_inc_threshold = (target_logodds - b0) / b1
            threshold_income  = np.exp(log_inc_threshold)

        vals = sub[income_col].values
        wts  = sub["weight"].values
        rows.append({
            "year":             int(yr),
            "threshold_income": threshold_income,
            "mean_income":      np.average(vals, weights=wts),
            "median_income":    weighted_median(vals, wts),
            "n":                len(sub),
            "b0":               b0,
            "b1":               b1,
        })

    return pd.DataFrame(rows)

print("\nFitting models (unadjusted)…")
df_unadj = df_base.dropna(subset=["coninc"]).copy()
res_unadj = run_threshold_analysis(df_unadj, "coninc")

print("Fitting models (HH-adjusted)…")
df_adj = df_base.dropna(subset=["coninc_eq"]).copy()
res_adj = run_threshold_analysis(df_adj, "coninc_eq")

# Save
res_unadj.to_csv(f"{OUT_DIR}/threshold_unadjusted.csv", index=False, float_format="%.1f")
res_adj.to_csv(  f"{OUT_DIR}/threshold_adjusted.csv",   index=False, float_format="%.1f")
print(f"\nSaved CSVs to {OUT_DIR}/")

# Quick summary
print("\nUnadjusted — selected years:")
cols = ["year", "threshold_income", "mean_income", "median_income"]
snap = res_unadj[res_unadj["year"].isin([1972, 1980, 1990, 2000, 2010, 2018, 2024])][cols].copy()
print(snap.applymap(lambda x: f"${x:,.0f}" if isinstance(x, float) else str(x)).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Smooth: 5-wave centred rolling average
#    The GSS is not evenly spaced (annual through 1993, biennial after), so
#    "5-year" is interpreted as 5 adjacent survey waves, centred.
# ---------------------------------------------------------------------------

def add_smooth(res, half_window=2):
    """
    Centred 5-calendar-year rolling average (±2 years).
    For each survey year Y, averages over all survey waves where
    Y - half_window <= wave_year <= Y + half_window.
    Post-1994 biennial waves contribute fewer observations per window
    than annual pre-1994 waves, but the calendar span is consistent.
    """
    r = res.copy().sort_values("year").reset_index(drop=True)
    for col in ["threshold_income", "mean_income", "median_income"]:
        smoothed = []
        for yr in r["year"]:
            mask = (r["year"] >= yr - half_window) & (r["year"] <= yr + half_window)
            smoothed.append(r.loc[mask, col].mean())
        r[f"{col}_smooth"] = smoothed
    return r

res_unadj = add_smooth(res_unadj)
res_adj   = add_smooth(res_adj)

# ---------------------------------------------------------------------------
# 4. Plotting helper
# ---------------------------------------------------------------------------

def plot_threshold(res, income_label, filename):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw series — faint background dots
    ax.plot(res["year"], res["threshold_income"] / 1000,
            color="#d7191c", lw=0, marker="o", ms=3, alpha=0.25)
    ax.plot(res["year"], res["median_income"] / 1000,
            color="#2c7bb6", lw=0, marker="^", ms=3, alpha=0.25)

    # Smoothed series — primary lines
    ax.plot(res["year"], res["threshold_income_smooth"] / 1000,
            color="#d7191c", lw=2.4,
            label=f"25% dissatisfaction threshold  (5-yr rolling avg)")
    ax.plot(res["year"], res["median_income_smooth"] / 1000,
            color="#2c7bb6", lw=2.2,
            label="Weighted median income  (5-yr rolling avg)")

    # Shade gap between smoothed threshold and smoothed median
    t = res["threshold_income_smooth"]
    m = res["median_income_smooth"]
    ax.fill_between(res["year"], m / 1000, t / 1000,
                    where=t > m,  color="#d7191c", alpha=0.12,
                    label="Median below threshold (at-risk)")
    ax.fill_between(res["year"], m / 1000, t / 1000,
                    where=t <= m, color="#1a9641", alpha=0.12,
                    label="Median above threshold (comfortable)")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Constant dollars (thousands)", fontsize=12)
    ax.set_title(
        f"The rising cost of financial security  —  {income_label}\n"
        f"Income needed for <25% chance of 'not satisfied at all'  (faint dots = raw values)",
        fontsize=13, fontweight="bold"
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
    ax.legend(fontsize=9.5, framealpha=0.9, loc="upper left")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_xlim(res["year"].min() - 1, res["year"].max() + 1)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_comparison(res_u, res_a, filename):
    """Side-by-side: unadjusted vs HH-adjusted."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    for ax, res, title in zip(
        axes,
        [res_u, res_a],
        ["Unadjusted income (coninc)",
         "HH-equivalised income (coninc ÷ √hompop)"]
    ):
        # Raw — faint dots
        ax.plot(res["year"], res["threshold_income"] / 1000,
                color="#d7191c", lw=0, marker="o", ms=3, alpha=0.25)
        ax.plot(res["year"], res["median_income"] / 1000,
                color="#2c7bb6", lw=0, marker="^", ms=3, alpha=0.25)
        # Smoothed — primary lines
        ax.plot(res["year"], res["threshold_income_smooth"] / 1000,
                color="#d7191c", lw=2.4, label="25% dissatisfaction threshold")
        ax.plot(res["year"], res["median_income_smooth"] / 1000,
                color="#2c7bb6", lw=2.2, label="Weighted median income")
        t = res["threshold_income_smooth"]
        m = res["median_income_smooth"]
        ax.fill_between(res["year"], m / 1000, t / 1000,
                        where=t > m,  color="#d7191c", alpha=0.10)
        ax.fill_between(res["year"], m / 1000, t / 1000,
                        where=t <= m, color="#1a9641", alpha=0.10)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Year", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.set_xlim(res["year"].min() - 1, res["year"].max() + 1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Income needed for <25% chance of financial dissatisfaction vs. actual incomes\n"
        "(red shading = median income falls short of the threshold)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


# ---------------------------------------------------------------------------
# 5. Generate plots
# ---------------------------------------------------------------------------

print("\nGenerating plots…")
plot_threshold(res_unadj, "Unadjusted income",
               f"{OUT_DIR}/threshold_unadjusted.png")
plot_threshold(res_adj,   "HH-equivalised income",
               f"{OUT_DIR}/threshold_adjusted.png")
plot_comparison(res_unadj, res_adj,
                f"{OUT_DIR}/threshold_comparison.png")

print("\nDone.")
