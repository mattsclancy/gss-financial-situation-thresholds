# GSS Financial Satisfaction Analysis

## Goal

Track how the relationship between income and financial satisfaction has changed in the United States from 1972 to 2024, using data from the General Social Survey (GSS). The core idea is to decompose changes in overall financial satisfaction into two components: (1) shifting expectations — people needing more income to feel satisfied — and (2) rising incomes — people actually having more money.

## Data

The input file is an Excel workbook downloaded from the GSS Data Explorer (gssdataexplorer.norc.org). It contains one row per respondent across all available GSS years (1972–2024) with the following variables:

- **`satfin`**: Satisfaction with financial situation. Categorical: `"Pretty well satisfied"` / `"More or less satisfied"` / `"Not at all satisfied"`. Treat as binary: satisfied = `"Pretty well satisfied"`, not satisfied = the other two.
- **`coninc`**: Family income in constant (inflation-adjusted) dollars. Continuous.
- **`year`**: GSS survey year.
- **`wtssall`**: Cross-sectional survey weight. Always apply this weight in all calculations.

The Excel file has two sheets: the data sheet and a metadata sheet. Load the data sheet. Drop rows where `satfin` or `coninc` is missing.

Note: the GSS was annual through 1993, then biennial. The 2021 wave switched to web-based collection — treat it as a minor discontinuity but include it.

## Analysis

### Step 1: Fit per-year satisfaction curves

For each year in the data:
1. Fit a weighted logistic regression of `satisfied` (binary) on `log(coninc)`, using `wtssall` as weights.
2. Store the fitted model (or just the coefficients — intercept and slope on log income) for each year.
3. Also compute the weighted mean of `satisfied` for each year (the raw satisfaction rate).

This gives you, for each year, a function `P(satisfied | income, year)` — the probability of being satisfied at a given income level in that year.

### Step 2: Build a reference income distribution

Use the 1972 income distribution as the reference. Specifically, extract all `coninc` values and their associated `wtssall` weights from 1972 respondents. This weighted sample of incomes is the "frozen" baseline distribution.

### Step 3: Analysis 1 — Shifting expectations

**Question:** If incomes had stayed at their 1972 distribution, how would overall satisfaction have changed over time purely due to shifting expectations?

For each year Y:
1. Take the 1972 income distribution (from Step 2).
2. Apply the Year Y satisfaction curve (from Step 1) to each income value in that distribution.
3. Compute the weighted average predicted probability of satisfaction across the 1972 income distribution.

This gives a time series: `counterfactual_satisfaction_frozen_income[Y]`. Because incomes are held constant at 1972 levels, any change in this series reflects only the shifting satisfaction function — i.e., changing expectations about what income is "enough."

**Expected result:** A declining trend, indicating that at any given income level, people became less satisfied over time — meaning expectations rose.

### Step 4: Analysis 2 — Effect of income growth

**Question:** If expectations had stayed at their 1972 levels, how would overall satisfaction have changed over time purely due to rising incomes?

For each year Y:
1. Take the Year Y income distribution (all respondents in year Y with their weights).
2. Apply the **1972** satisfaction curve (from Step 1) to each income value in that distribution.
3. Compute the weighted average predicted probability of satisfaction across the Year Y income distribution.

This gives a time series: `counterfactual_satisfaction_frozen_expectations[Y]`. Because the satisfaction curve is held constant at 1972 levels, any change in this series reflects only changing incomes.

**Expected result:** A rising trend, indicating that income growth alone would have made people more satisfied if expectations had not shifted.

### Step 5: Compile and plot results

Produce a single DataFrame with one row per year containing:
- `year`
- `actual_satisfaction`: weighted share satisfied in that year (from Step 1)
- `frozen_income_satisfaction`: counterfactual from Analysis 1
- `frozen_expectations_satisfaction`: counterfactual from Analysis 2

Plot all three series on the same chart, labeled clearly. Use a line chart with year on the x-axis and share satisfied (0–1 or 0–100%) on the y-axis.

Also produce a secondary plot showing, for selected anchor years (e.g. 1972, 1985, 2000, 2010, 2022), the fitted satisfaction curve — P(satisfied) vs. income — to visualise directly how the curve has shifted.

## Implementation notes

- Use `pandas`, `numpy`, `sklearn.linear_model.LogisticRegression` (or `statsmodels.formula.api.logit` for easier weighted fitting), and `matplotlib`.
- `statsmodels` handles weighted logistic regression more cleanly: use `smf.logit("satisfied ~ np.log(coninc)", data=df_year, freq_weights=...).fit()`. Note `freq_weights` expects integer counts — use `wtssall` scaled to sum to the sample size, or use `weights` with `WLS`-style fitting if available.
- Alternatively, use `sklearn` with `sample_weight` in `.fit()`.
- For the income grid in the curve plots, use a range from roughly the 5th to 95th percentile of `coninc` across all years.
- Save all plots as PNG files.
- Print a summary table of the results DataFrame to stdout.

## Output files

- `results.csv`: the per-year summary DataFrame
- `main_trends.png`: the three-series trend chart
- `satisfaction_curves.png`: the per-year fitted curves for anchor years
