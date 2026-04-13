"""
Non-parametric diagnostic: bracket-level curves for anchor years
=================================================================
For a set of anchor years, plots the weighted proportion of bad outcomes
at each income bracket, the 25% interpolation threshold, and the fitted
logistic regression curve — so you can see exactly where and why the two
methods agree or diverge.

Outputs: output/non_parametric/diagnostic_satfin.png
         output/non_parametric/diagnostic_finrela.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

DATA_PATH   = "data/GSS.xlsx"
OUT_DIR     = "output/non_parametric"
THRESHOLD   = 0.25
ANCHOR_YEARS = [1974, 1984, 1994, 2004, 2012, 2022]

# ---------------------------------------------------------------------------
# 1. Load and clean (same as other scripts)
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

# Snap anchor years to nearest available survey year
available_years = sorted(df["year"].dropna().unique())

def nearest_year(target, available):
    return min(available, key=lambda y: abs(y - target))

anchor_years = [nearest_year(y, available_years) for y in ANCHOR_YEARS]
anchor_years = sorted(set(anchor_years))
print(f"Anchor years: {anchor_years}")

# ---------------------------------------------------------------------------
# 2. Per-year bracket data + logistic curve
# ---------------------------------------------------------------------------

def bracket_props(sub, income_col, outcome_col):
    """Weighted proportion of bad outcome per income bracket."""
    sub = sub.dropna(subset=[income_col, outcome_col, "weight"])
    sub = sub[sub["weight"] > 0]
    incomes, props, weights = [], [], []
    for inc_val, group in sub.groupby(income_col, observed=True):
        prop = np.average(group[outcome_col], weights=group["weight"])
        incomes.append(inc_val)
        props.append(prop)
        weights.append(group["weight"].sum())
    order   = np.argsort(incomes)
    return (np.array(incomes)[order],
            np.array(props)[order],
            np.array(weights)[order])

def interpolated_threshold(incomes, props, target=THRESHOLD):
    """Linear interpolation through the first downward crossing."""
    if len(props) < 2 or props[0] < target or props[-1] > target:
        return np.nan
    for i in range(len(props) - 1):
        p0, p1 = props[i], props[i + 1]
        if p0 >= target >= p1 and p0 != p1:
            t = (target - p0) / (p1 - p0)
            return incomes[i] + t * (incomes[i + 1] - incomes[i])
    return np.nan

def logistic_curve(sub, income_col, outcome_col, x_range):
    """Fitted logistic P(bad outcome | income) over x_range."""
    sub = sub.dropna(subset=[income_col, outcome_col, "weight"])
    sub = sub[sub["weight"] > 0]
    if len(np.unique(sub[outcome_col])) < 2:
        return None, None
    X = np.log(sub[income_col].values).reshape(-1, 1)
    y = sub[outcome_col].values
    w = sub["weight"].values
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X, y, sample_weight=w)
    x_log = np.log(x_range).reshape(-1, 1)
    return x_range, clf.predict_proba(x_log)[:, 1]

def logistic_threshold(sub, income_col, outcome_col, target=THRESHOLD):
    """Analytically inverted logistic threshold income."""
    sub = sub.dropna(subset=[income_col, outcome_col, "weight"])
    sub = sub[sub["weight"] > 0]
    if len(np.unique(sub[outcome_col])) < 2:
        return np.nan
    X = np.log(sub[income_col].values).reshape(-1, 1)
    y = sub[outcome_col].values
    w = sub["weight"].values
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X, y, sample_weight=w)
    b0, b1 = clf.intercept_[0], clf.coef_[0][0]
    if b1 >= 0:
        return np.nan
    return np.exp((np.log(target / (1 - target)) - b0) / b1)

# ---------------------------------------------------------------------------
# 3. Diagnostic plot
# ---------------------------------------------------------------------------

def make_diagnostic(outcome_col, outcome_label, income_col, income_label,
                    df_base, filename):
    n_years = len(anchor_years)
    fig, axes = plt.subplots(2, n_years // 2, figsize=(16, 8), sharey=False)
    axes = axes.flatten()

    for ax, yr in zip(axes, anchor_years):
        sub = df_base[df_base["year"] == yr]
        incomes, props, wts = bracket_props(sub, income_col, outcome_col)

        if len(incomes) == 0:
            ax.set_title(str(yr))
            continue

        n_brackets = len(incomes)
        wt_norm = wts / wts.max() * 120

        ax.scatter(incomes / 1000, props, s=wt_norm,
                   color="#d7191c", alpha=0.7, zorder=3,
                   label=f"Bracket proportion (n={n_brackets})")

        # Logistic curve
        x_range = np.linspace(incomes.min(), incomes.max(), 300)
        lx, lp  = logistic_curve(sub, income_col, outcome_col, x_range)
        if lx is not None:
            ax.plot(lx / 1000, lp, color="#e08214", lw=1.8, ls="--",
                    alpha=0.9, label="Logistic fit")

        ax.axhline(THRESHOLD, color="grey", lw=1, ls=":", alpha=0.8)

        np_thresh = interpolated_threshold(incomes, props)
        if not np.isnan(np_thresh):
            ax.axvline(np_thresh / 1000, color="#d7191c", lw=1.5, alpha=0.7,
                       label=f"NP: ${np_thresh/1000:.0f}k")

        lg_thresh = logistic_threshold(sub, income_col, outcome_col)
        if not np.isnan(lg_thresh):
            ax.axvline(lg_thresh / 1000, color="#e08214", lw=1.5, ls="--",
                       alpha=0.7, label=f"Logit: ${lg_thresh/1000:.0f}k")

        ax.set_title(str(yr), fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{income_label} ($k)", fontsize=9)
        ax.set_ylabel("P(bad outcome)", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
        ax.legend(fontsize=7, framealpha=0.8)
        ax.grid(axis="y", ls="--", alpha=0.3)

    fig.suptitle(
        f"Non-parametric bracket curves — {outcome_label}  [{income_label}]\n"
        "Dot size = total bracket weight. Red vertical = NP threshold, "
        "orange dashed = logistic threshold.",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


df_satfin  = df.dropna(subset=["year", "satfin_clean",  "weight"]).copy()
df_satfin  = df_satfin[df_satfin["weight"] > 0]

df_finrela = df.dropna(subset=["year", "finrela_clean", "weight"]).copy()
df_finrela = df_finrela[df_finrela["weight"] > 0]

print("\nGenerating diagnostic plots…")

# HH-adjusted (original)
make_diagnostic("dissatisfied", "satfin",  "coninc_eq", "HH-equiv. income",
                df_satfin,  f"{OUT_DIR}/diagnostic_satfin.png")
make_diagnostic("below_avg",   "finrela", "coninc_eq", "HH-equiv. income",
                df_finrela, f"{OUT_DIR}/diagnostic_finrela.png")

# Unadjusted
make_diagnostic("dissatisfied", "satfin",  "coninc", "Unadjusted income",
                df_satfin,  f"{OUT_DIR}/diagnostic_satfin_unadjusted.png")
make_diagnostic("below_avg",   "finrela", "coninc", "Unadjusted income",
                df_finrela, f"{OUT_DIR}/diagnostic_finrela_unadjusted.png")

print("\nDone.")
