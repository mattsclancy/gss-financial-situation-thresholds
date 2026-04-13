"""
Logistic curve illustrations for anchor years
==============================================
For six anchor years, plots the weighted proportion of bad outcomes at each
income bracket alongside the fitted logistic regression curve. Shows visually
what the model is doing: fitting a sigmoid to the bracket-level data and
reading off the income at which the curve crosses 25%.

Outputs: output/logistic_curves/satfin_curves.png
         output/logistic_curves/finrela_curves.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
import os

warnings.filterwarnings("ignore")

DATA_PATH    = "data/GSS.xlsx"
OUT_DIR      = "output/logistic_curves"
THRESHOLD    = 0.25
ANCHOR_YEARS = [1974, 1984, 1994, 2004, 2012, 2022]

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

VALID_SATFIN = {"Pretty well satisfied", "More or less satisfied", "Not satisfied at all"}
df["satfin_clean"] = df["satfin"].where(df["satfin"].isin(VALID_SATFIN))
df["dissatisfied"] = (df["satfin_clean"] == "Not satisfied at all").astype(float)
df.loc[df["satfin_clean"].isna(), "dissatisfied"] = np.nan

VALID_FINRELA = {"FAR ABOVE AVERAGE", "ABOVE AVERAGE", "AVERAGE",
                 "BELOW AVERAGE", "FAR BELOW AVERAGE"}
df["finrela_clean"] = df["finrela"].where(df["finrela"].isin(VALID_FINRELA))
df["below_avg"]     = df["finrela_clean"].isin({"BELOW AVERAGE", "FAR BELOW AVERAGE"}).astype(float)
df.loc[df["finrela_clean"].isna(), "below_avg"] = np.nan

available_years = sorted(df["year"].dropna().unique())
anchor_years    = sorted({min(available_years, key=lambda y: abs(y - t)) for t in ANCHOR_YEARS})
print(f"Anchor years: {anchor_years}")

# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

def bracket_props(sub, income_col, outcome_col):
    sub = sub.dropna(subset=[income_col, outcome_col, "weight"])
    sub = sub[sub["weight"] > 0]
    incomes, props, weights = [], [], []
    for inc_val, group in sub.groupby(income_col, observed=True):
        props.append(np.average(group[outcome_col], weights=group["weight"]))
        incomes.append(inc_val)
        weights.append(group["weight"].sum())
    order = np.argsort(incomes)
    return (np.array(incomes)[order],
            np.array(props)[order],
            np.array(weights)[order])

def fit_logistic(sub, income_col, outcome_col):
    sub = sub.dropna(subset=[income_col, outcome_col, "weight"])
    sub = sub[sub["weight"] > 0]
    if len(np.unique(sub[outcome_col])) < 2:
        return None, None, np.nan
    X   = np.log(sub[income_col].values).reshape(-1, 1)
    y   = sub[outcome_col].values
    w   = sub["weight"].values
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X, y, sample_weight=w)
    b0, b1 = clf.intercept_[0], clf.coef_[0][0]
    threshold = np.exp((np.log(THRESHOLD / (1 - THRESHOLD)) - b0) / b1) if b1 < 0 else np.nan

    def curve(x_range):
        return clf.predict_proba(np.log(x_range).reshape(-1, 1))[:, 1]

    return curve, (b0, b1), threshold

# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------

def make_curves(outcome_col, outcome_label, df_base, filename):
    n  = len(anchor_years)
    nc = 3
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(15, 4.5 * nr))
    axes = axes.flatten()

    for ax, yr in zip(axes, anchor_years):
        sub = df_base[df_base["year"] == yr]
        incomes, props, wts = bracket_props(sub, "coninc_eq", outcome_col)

        if len(incomes) < 3:
            ax.set_visible(False)
            continue

        curve_fn, coeffs, lg_thresh = fit_logistic(sub, "coninc_eq", outcome_col)

        # Bracket proportion dots, sized by weight
        wt_norm = wts / wts.max() * 160
        ax.scatter(incomes / 1000, props, s=wt_norm,
                   color="#4d4d4d", alpha=0.65, zorder=3)

        # Logistic curve
        if curve_fn is not None:
            x_range = np.linspace(incomes.min() * 0.9, incomes.max() * 1.05, 400)
            ax.plot(x_range / 1000, curve_fn(x_range),
                    color="#d7191c", lw=2.2, zorder=2)

        # 25% reference line
        ax.axhline(THRESHOLD, color="grey", lw=1, ls=":", alpha=0.7)

        # Logistic threshold vertical
        if not np.isnan(lg_thresh):
            ax.axvline(lg_thresh / 1000, color="#d7191c", lw=1.4,
                       ls="--", alpha=0.8,
                       label=f"Threshold: ${lg_thresh/1000:.0f}k")
            ax.legend(fontsize=8, framealpha=0.85, loc="upper right")

        ax.set_title(str(yr), fontsize=12, fontweight="bold")
        ax.set_xlabel("Equiv. household income", fontsize=9)
        ax.set_ylabel("Share reporting bad outcome", fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(axis="y", ls="--", alpha=0.3)

    # Hide any unused axes
    for ax in axes[len(anchor_years):]:
        ax.set_visible(False)

    fig.suptitle(
        f"{outcome_label}\n"
        "Each dot is one income bracket; size reflects total survey weight. "
        "Red curve: logistic fit. Dashed line: 25% threshold income.",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


df_satfin  = df.dropna(subset=["year", "satfin_clean",  "weight"]).copy()
df_satfin  = df_satfin[df_satfin["weight"] > 0]

df_finrela = df.dropna(subset=["year", "finrela_clean", "weight"]).copy()
df_finrela = df_finrela[df_finrela["weight"] > 0]

print("\nGenerating charts…")
make_curves(
    "dissatisfied",
    "Financial dissatisfaction (satfin) — P(not satisfied at all) by income bracket",
    df_satfin,
    f"{OUT_DIR}/satfin_curves.png"
)
make_curves(
    "below_avg",
    "Relative standing (finrela) — P(below average) by income bracket",
    df_finrela,
    f"{OUT_DIR}/finrela_curves.png"
)
print("\nDone.")
