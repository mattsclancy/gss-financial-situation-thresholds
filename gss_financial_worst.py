"""
GSS Financial Satisfaction Analysis — Worst Outcome
=====================================================
Identical decomposition to gss_financial.py, but predicting the probability
of the WORST response: "Not satisfied at all" (rather than "Pretty well satisfied").

Runs both unadjusted (coninc) and HH-adjusted (coninc / sqrt(hompop)) variants.

Weight: wtssps. Data source: data/GSS.xlsx. Outputs: output/financial_worst/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

DATA_PATH    = "data/GSS.xlsx"
ANCHOR_YEARS = [1972, 1985, 2000, 2010, 2022]
REF_YEAR     = 1972
OUT_DIR      = "output/financial_worst"

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
df["satfin_clean"]  = df["satfin"].where(df["satfin"].isin(VALID_SATFIN))
df["dissatisfied"]  = (df["satfin_clean"] == "Not satisfied at all").astype(float)

df_base = df.dropna(subset=["year", "satfin_clean", "weight"]).copy()
df_base  = df_base[df_base["weight"] > 0].copy()
print(f"  Base rows: {len(df_base):,}")

# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def run_analysis(df_in, income_col, label):
    print(f"\n{'='*60}")
    print(f"  Variant: {label}  (income = {income_col})")
    print(f"{'='*60}")

    df_c = df_in.dropna(subset=[income_col]).copy()
    print(f"  Rows after income filter: {len(df_c):,}")

    years = sorted(df_c["year"].unique())
    models, year_stats = {}, {}

    for yr in years:
        sub = df_c[df_c["year"] == yr]
        X = np.log(sub[income_col].values).reshape(-1, 1)
        y = sub["dissatisfied"].values
        w = sub["weight"].values
        if len(np.unique(y)) < 2 or len(sub) < 10:
            continue
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X, y, sample_weight=w)
        models[yr] = clf
        year_stats[yr] = {"actual": np.average(y, weights=w), "n": len(sub)}

    fitted_years = sorted(models.keys())
    print(f"  Fitted {len(fitted_years)} years.")

    ref_df  = df_c[df_c["year"] == REF_YEAR]
    ref_inc = ref_df[income_col].values
    ref_w   = ref_df["weight"].values
    log_ref = np.log(ref_inc).reshape(-1, 1)
    print(f"  1972 reference: {len(ref_df):,} respondents, "
          f"median {income_col} = ${np.median(ref_inc):,.0f}")

    model_72 = models[REF_YEAR]
    frozen_income, frozen_expect = {}, {}

    for yr in fitted_years:
        probs = models[yr].predict_proba(log_ref)[:, 1]
        frozen_income[yr] = np.average(probs, weights=ref_w)

        sub = df_c[df_c["year"] == yr]
        log_inc = np.log(sub[income_col].values).reshape(-1, 1)
        probs72 = model_72.predict_proba(log_inc)[:, 1]
        frozen_expect[yr] = np.average(probs72, weights=sub["weight"].values)

    results = pd.DataFrame({
        "year":                               fitted_years,
        "actual_dissatisfaction":             [year_stats[yr]["actual"] for yr in fitted_years],
        "frozen_income_dissatisfaction":      [frozen_income[yr]        for yr in fitted_years],
        "frozen_expectations_dissatisfaction":[frozen_expect[yr]        for yr in fitted_years],
        "n_respondents":                      [year_stats[yr]["n"]      for yr in fitted_years],
    })

    print("\n  Per-year summary:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return results, models, df_c

# ---------------------------------------------------------------------------
# 2. Run both variants
# ---------------------------------------------------------------------------

res_unadj, models_unadj, df_unadj = run_analysis(df_base, "coninc",    "Unadjusted")
res_adj,   models_adj,   df_adj   = run_analysis(df_base, "coninc_eq", "HH-adjusted (÷√hompop)")

res_unadj.to_csv(f"{OUT_DIR}/results_unadjusted.csv", index=False, float_format="%.4f")
res_adj.to_csv(  f"{OUT_DIR}/results_adjusted.csv",   index=False, float_format="%.4f")
print(f"\nSaved CSVs to {OUT_DIR}/")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {"actual": "#2c7bb6", "frozen_income": "#d7191c", "frozen_expect": "#1a9641"}

def plot_main_trends(results, title_suffix, filename):
    fig, ax = plt.subplots(figsize=(11, 6))
    yr = results["year"]
    ax.plot(yr, results["actual_dissatisfaction"] * 100,
            color=COLORS["actual"], lw=2.2, marker="o", ms=4,
            label="Actual 'not satisfied at all' rate")
    ax.plot(yr, results["frozen_income_dissatisfaction"] * 100,
            color=COLORS["frozen_income"], lw=2.2, ls="--", marker="s", ms=4,
            label="Counterfactual: 1972 incomes, shifting expectations\n"
                  "(rising → dissatisfaction grew at any given income)")
    ax.plot(yr, results["frozen_expectations_dissatisfaction"] * 100,
            color=COLORS["frozen_expect"], lw=2.2, ls="-.", marker="^", ms=4,
            label="Counterfactual: current incomes, 1972 expectations\n"
                  "(falling → income growth would have reduced dissatisfaction)")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Share 'Not satisfied at all' (%)", fontsize=12)
    ax.set_title(f"Financial Dissatisfaction in the US (1972–2024)\n{title_suffix}",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(fontsize=9.5, framealpha=0.9)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_xlim(results["year"].min() - 1, results["year"].max() + 1)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_curves(df_c, income_col, models, title_suffix, filename):
    p5, p95  = np.percentile(df_c[income_col], [5, 95])
    inc_grid = np.linspace(p5, p95, 300)
    log_grid = np.log(inc_grid).reshape(-1, 1)
    anchors  = [yr for yr in ANCHOR_YEARS if yr in models]
    cmap     = plt.cm.get_cmap("RdYlGn_r", len(anchors))   # reversed: red = bad

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, yr in enumerate(anchors):
        probs = models[yr].predict_proba(log_grid)[:, 1]
        ax.plot(inc_grid / 1000, probs * 100, color=cmap(i), lw=2.2, label=str(yr))

    x_label = ("Equivalised family income, constant dollars (thousands)"
               if "eq" in income_col else
               "Family income, constant dollars (thousands)")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("P(Not satisfied at all) (%)", fontsize=12)
    ax.set_title(f"Income–dissatisfaction curve by year\n{title_suffix}",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}k"))
    ax.legend(title="Year", fontsize=10, title_fontsize=10)
    ax.grid(ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def plot_comparison(res_unadj, res_adj, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, res, title in zip(
        axes,
        [res_unadj, res_adj],
        ["Unadjusted income (coninc)",
         "HH-equivalised income (coninc ÷ √hompop)"]
    ):
        yr = res["year"]
        ax.plot(yr, res["actual_dissatisfaction"] * 100,
                color=COLORS["actual"], lw=2.2, marker="o", ms=3,
                label="Actual dissatisfaction")
        ax.plot(yr, res["frozen_income_dissatisfaction"] * 100,
                color=COLORS["frozen_income"], lw=2.2, ls="--", marker="s", ms=3,
                label="Frozen 1972 incomes")
        ax.plot(yr, res["frozen_expectations_dissatisfaction"] * 100,
                color=COLORS["frozen_expect"], lw=2.2, ls="-.", marker="^", ms=3,
                label="Frozen 1972 expectations")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Year", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.set_xlim(res["year"].min() - 1, res["year"].max() + 1)

    axes[0].set_ylabel("Share 'Not satisfied at all' (%)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=9.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Financial dissatisfaction: does household-size adjustment matter?",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


# ---------------------------------------------------------------------------
# 3. Generate plots
# ---------------------------------------------------------------------------

print("\nGenerating plots…")
plot_main_trends(res_unadj, "Unadjusted income",
                 f"{OUT_DIR}/main_trends_unadjusted.png")
plot_main_trends(res_adj,   "HH-equivalised income (÷ √household size)",
                 f"{OUT_DIR}/main_trends_adjusted.png")
plot_curves(df_unadj, "coninc",    models_unadj, "Unadjusted income",
            f"{OUT_DIR}/curves_unadjusted.png")
plot_curves(df_adj,   "coninc_eq", models_adj,   "HH-equivalised income",
            f"{OUT_DIR}/curves_adjusted.png")
plot_comparison(res_unadj, res_adj, f"{OUT_DIR}/comparison_adj_vs_unadj.png")

print("\nDone.")
