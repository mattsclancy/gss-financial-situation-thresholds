# gss-financial-situation-thresholds

Code for the charts in [this post](https://mattsclancy.github.io/2026/04/12/happiness-is-reality-minus-expectations.html) on how the income required to feel financially secure has changed in the United States since 1972.

## Scripts

| Script | Chart |
|--------|-------|
| `gss_threshold.py` | Income at which there is a 25% chance of reporting "not satisfied at all" with one's financial situation (`satfin`) |
| `gss_finrela_worst.py` | Income at which there is a 25% chance of reporting "below average" or "far below average" on relative income (`finrela`) |

## Data

Download a GSS extract from [gssdataexplorer.norc.org](https://gssdataexplorer.norc.org/) with variables `satfin`, `finrela`, `coninc`, `hompop`, and `wtssps`. Save it as `data/GSS.xlsx`. Note that `wtssps` must be added explicitly — it is not included by default.

## Requirements

```
pip install pandas openpyxl scikit-learn matplotlib statsmodels
```

Python 3.9+. Run scripts from the project root.
