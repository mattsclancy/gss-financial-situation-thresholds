"""
GSS finrela Worst-Outcome Threshold Analysis
=============================================
Mirrors gss_finrela_threshold.py but predicts the probability of the WORST
outcome: "BELOW AVERAGE" or "FAR BELOW AVERAGE".

For each year, finds the income at which there is a 25% chance of reporting
feeling below average — the same 25% threshold used in the financial
dissatisfaction analysis (gss_threshold.py), making the two directly comparable.

Binary outcome: BELOW AVERAGE / FAR BELOW AVERAGE = 1,  everything else = 0
Target P:       0.25  →  log-odds = log(0.25/0.75) ≈ -1.099
Threshold:      income = exp((log_odds_target - b0) / b1)

Both unadjusted (coninc) and HH-adjusted (coninc / sqrt(hompop)) variants.
Smoothing: ±2 calendar year centred window.

Outputs: output/finrela_worst/
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
OUT_DIR   = "output/finrela_worst"
THRESHOLD = 0.25

os.makedirs(OUT_DIR, exist_ok=True)
TARGET_LOGODDS = np.log(THRESHOLD / (1 - THRESHOLD))   # ≈ -1.099

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

REQUIRED_COLS = {"year", "wtssps", "coninc", "hompop", "finrela"}
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

_FINRELA_MAP = {
    "far above average": "FAR ABOVE AVERAGE",
    "above average":     "ABOVE AVERAGE",
    "average":           "AVERAGE",
    "below average":     "BELOW AVERAGE",
    "far below average": "FAR BELOW AVERAGE",
}
_finrela_norm = df["finrela"].astype(str).str.strip().str.lower()
unknown_finrela = {v for v in _finrela_norm.unique()
                   if v not in _FINRELA_MAP and v not in {"nan", ""}}
if unknown_finrela:
    print(f"  Warning: unrecognized finrela values (will be dropped): {unknown_finrela}")
df["finrela_clean"] = _finrela_norm.map(_FINRELA_MAP)
df["below_avg"]     = df["finrela_clean"].isin(
    {"BELOW AVERAGE", "FAR BELOW AVERAGE"}
).astype(float)
df.loc[df["finrela_clean"].isna(), "below_avg"] = np.nan

df_base = df.dropna(subset=["year", "finrela_clean", "weight"]).copy()
df_base  = df_base[df_base["weight"] > 0].copy()
print(f"  Base rows: {len(df_base):,}")
print(f"  Overall share 'below average': "
      f"{(df_base['below_avg'] * df_base['weight']).sum() / df_base['weight'].sum():.1%}")

# ---------------------------------------------------------------------------
# 2. Fit per-year models and compute threshold
# ---------------------------------------------------------------------------

def weighted_median(vals, wts):
    order = np.argsort(vals)
    vals, wts = vals[order], wts[order]
    cumw = np.cumsum(wts)
    return vals[np.searchsorted(cumw, cumw[-1] / 2.0)]

def run_threshold_analysis(df_in, income_col):
    df_c  = df_in.dropna(subset=[income_col, "below_avg"]).copy()
    years = sorted(df_c["year"].unique())
    rows  = []

    for yr in years:
        sub = df_c[df_c["year"] == yr]
        X   = np.log(sub[income_col].values).reshape(-1, 1)
        y   = sub["below_avg"].values
        w   = sub["weight"].values

        if len(np.unique(y)) < 2 or len(sub) < 10:
            continue

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X, y, sample_weight=w)

        b0 = clf.intercept_[0]
        b1 = clf.coef_[0][0]

        # Higher income → lower P(below average), so b1 should be negative
        if b1 >= 0:
            print(f"  Warning: year {yr} has b1={b1:.4f} >= 0 (unexpected sign); threshold set to NaN")
            threshold_income = np.nan
        else:
            threshold_income = np.exp((TARGET_LOGODDS - b0) / b1)

        vals = sub[income_col].values
        wts  = sub["weight"].values
        rows.append({
            "year":             int(yr),
            "threshold_income": threshold_income,
            "mean_income":      np.average(vals, weights=wts),
            "median_income":    weighted_median(vals, wts),
            "n":                len(sub),
            "b0": b0, "b1": b1,
        })

    return pd.DataFrame(rows)

print(f"\nFitting models (unadjusted)…")
res_unadj = run_threshold_analysis(df_base, "coninc")
print(f"Fitting models (HH-adjusted)…")
res_adj   = run_threshold_analysis(df_base, "coninc_eq")

# ---------------------------------------------------------------------------
# 3. Smooth: ±2 calendar year centred window
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

res_unadj.to_csv(f"{OUT_DIR}/threshold_unadjusted.csv", index=False, float_format="%.1f")
res_adj.to_csv(  f"{OUT_DIR}/threshold_adjusted.csv",   index=False, float_format="%.1f")
print(f"\nSaved CSVs to {OUT_DIR}/")

print("\nHH-adjusted — selected years:")
snap = res_adj[res_adj["year"].isin([1972, 1980, 1990, 2000, 2010, 2018, 2024])].copy()
for _, row in snap.iterrows():
    ratio = row["threshold_income"] / row["median_income"]
    print(f"  {int(row['year'])}: threshold=${row['threshold_income']:>8,.0f}  "
          f"median=${row['median_income']:>8,.0f}  ratio={ratio:.2f}")

# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

def plot_threshold(res, income_label, filename):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(res["year"], res["threshold_income"] / 1000,
            color="#d7191c", lw=0, marker="o", ms=3, alpha=0.25)
    ax.plot(res["year"], res["median_income"] / 1000,
            color="#2c7bb6", lw=0, marker="^", ms=3, alpha=0.25)

    ax.plot(res["year"], res["threshold_income_smooth"] / 1000,
            color="#d7191c", lw=2.4,
            label="Income where P(feel below average) = 25%  (5-yr rolling avg)")
    ax.plot(res["year"], res["median_income_smooth"] / 1000,
            color="#2c7bb6", lw=2.2,
            label="Weighted median income  (5-yr rolling avg)")

    t = res["threshold_income_smooth"]
    m = res["median_income_smooth"]
    ax.fill_between(res["year"], m / 1000, t / 1000,
                    where=t > m,  color="#d7191c", alpha=0.12,
                    label="Median below threshold (at risk of feeling below average)")
    ax.fill_between(res["year"], m / 1000, t / 1000,
                    where=t <= m, color="#1a9641", alpha=0.12,
                    label="Median above threshold (safely above 25% risk)")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Constant dollars (thousands)", fontsize=12)
    ax.set_title(
        f"The income needed to avoid feeling below average  —  {income_label}\n"
        f"25% threshold on P(below average finrela) vs. actual median income  "
        f"(faint dots = raw values)",
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
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    for ax, res, title in zip(
        axes,
        [res_u, res_a],
        ["Unadjusted income (coninc)",
         "HH-equivalised income (coninc ÷ √hompop)"]
    ):
        ax.plot(res["year"], res["threshold_income"] / 1000,
                color="#d7191c", lw=0, marker="o", ms=3, alpha=0.25)
        ax.plot(res["year"], res["median_income"] / 1000,
                color="#2c7bb6", lw=0, marker="^", ms=3, alpha=0.25)
        ax.plot(res["year"], res["threshold_income_smooth"] / 1000,
                color="#d7191c", lw=2.4,
                label="25% 'below average' threshold")
        ax.plot(res["year"], res["median_income_smooth"] / 1000,
                color="#2c7bb6", lw=2.2,
                label="Weighted median income")
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
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Income needed for <25% chance of feeling below average vs. actual median income\n"
        "(red shading = typical household at elevated risk of feeling below average)",
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
