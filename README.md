# Happiness is Reality minus Expectations

An analysis of how the relationship between income and subjective wellbeing has changed in the United States from 1972 to 2024, using data from the [General Social Survey (GSS)](https://gss.norc.org/).

The core idea: decompose changes in reported happiness and financial satisfaction into two forces — rising incomes and rising expectations. If expectations rise faster than incomes, wellbeing stagnates even as people get richer.

## Key findings

- The income required to have less than a 25% chance of feeling financially dissatisfied has risen substantially since 1972, outpacing growth in median household income.
- A cleaner measure — the income needed to avoid feeling "below average" relative to others (`finrela`) — tells the same story without the top-coding problems that plague the GSS income variable: in 1972 the typical household sat comfortably above this threshold; by the 2010s–2020s it sits below it.
- Income growth alone, simulated at 2% per year from the 1972 distribution, would have raised happiness by only ~6 percentage points over 50 years — a small effect, because income is a weak predictor of general happiness even at 1972 norms.

## Data

Data comes from the [GSS Data Explorer](https://gssdataexplorer.norc.org/). The analysis uses the following variables:

| Variable | Description |
|----------|-------------|
| `satfin` | Satisfaction with financial situation (3 categories) |
| `happy`  | General happiness (3 categories) |
| `finrela`| Opinion of family income relative to others (5 categories) |
| `coninc` | Family income in constant dollars (top-coded bracket midpoints) |
| `hompop` | Number of people in household |
| `wtssps` | Post-stratification survey weight (recommended for analyses spanning the 2021 web-mode transition) |

**The data is not included in this repository.** Download your own extract from [gssdataexplorer.norc.org](https://gssdataexplorer.norc.org/), include the variables above, and save as `data/GSS.xlsx`.

> **Note on `wtssps`:** this weight must be added explicitly to your GSS extract — it is not included by default.

> **Note on income top-coding:** `coninc` is derived from categorical brackets. The top bracket captures ~2–3% of respondents in 1972 but ~15% by 2024, which attenuates measured income growth. The `finrela`-based analyses avoid this problem entirely.

## Scripts

Run scripts from the project root (`python3 <script>.py`). Each script is self-contained and saves outputs to `output/`.

| Script | What it does |
|--------|-------------|
| `gss_financial.py` | Decomposes financial satisfaction into shifting expectations vs. rising incomes. Unadjusted and HH-equivalised income. |
| `gss_financial_worst.py` | Same decomposition but predicting the worst outcome: "not satisfied at all". |
| `gss_happiness.py` | Same decomposition for general happiness ("very happy"). |
| `gss_happiness_worst.py` | Same decomposition for the worst happiness outcome: "not too happy". |
| `gss_simulation.py` | Simulates what 2% annual income growth would have done to happiness if the 1972–75 curve had held fixed. Uses Pareto-distributed within-bracket incomes. |
| `gss_threshold.py` | For each year, finds the income at which there is a 25% chance of financial dissatisfaction. Plots this threshold against median income. |
| `gss_finrela_threshold.py` | For each year, finds the income at which there is a 50% chance of feeling "average or above" on `finrela`. |
| `gss_finrela_worst.py` | For each year, finds the income at which there is a 25% chance of feeling "below average" on `finrela`. The cleanest measure of rising reference points. |

## Output structure

```
output/
  financial/          finrela decomposition — financial satisfaction
  financial_worst/    worst-outcome decomposition — financial dissatisfaction
  happiness/          decomposition — general happiness
  happiness_worst/    worst-outcome decomposition — unhappiness
  simulation/         2% growth simulation vs. actual happiness
  threshold/          25% financial dissatisfaction income threshold
  finrela/            50% "average or above" finrela threshold
  finrela_worst/      25% "below average" finrela threshold  ← key result
```

## Requirements

```
pip install pandas openpyxl scikit-learn matplotlib statsmodels
```

Python 3.9+. No GPU required.

## Methodology notes

- **Household-size adjustment:** equivalised income = `coninc / sqrt(hompop)` (OECD square-root scale). All threshold analyses use the adjusted measure for comparison against median income.
- **Survey weights:** `wtssps` is used throughout. It is the GSS-recommended weight for analyses that span the 2021 web-mode transition.
- **Logistic regression:** fitted with `sklearn.linear_model.LogisticRegression` using `sample_weight`. Income enters as `log(coninc)`.
- **Smoothing:** threshold series use a ±2 calendar year centred window (consistent calendar span regardless of whether the GSS was fielded annually or biennially).
- **Threshold inversion:** for a logistic model with log-odds = b₀ + b₁·log(income), the income at target probability p is `exp((log(p/(1−p)) − b₀) / b₁)`.
