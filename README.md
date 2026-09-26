# Credit Risk Strategy — LendingClub

Credit risk analysis and strategy built on the LendingClub loan dataset.

## Data

Dataset is pulled from Kaggle via `kagglehub` (see `download_data.py`):

```
wordsforthewise/lending-club
```

## Notebooks

- `notebooks/eda.ipynb` — exploratory analysis of loan volume, default rates by grade/term/purpose, and key risk drivers.
- `notebooks/scorecard.ipynb` — WOE/logistic-regression credit scorecard (v1) with out-of-time validation.
- `notebooks/scorecard_adv.ipynb` — scorecard v2: expanded feature set (adds `loan_to_income`, credit history length, delinquency recency, geography) on top of v1's methodology.
- `notebooks/cutoff_strategy.ipynb` — turns the scorecard into an approval-cutoff policy: realized-profit curves, a bad-capture gains chart, and a swap-set table.

## Scorecard design decisions

- **Out-of-time validation, not random split.** Model trains on loans issued 2007–2015 and is validated on loans issued in 2016. 2017–2018 vintages are excluded entirely: LendingClub loans that are still current or freshly charged off skew the resolved-loan mix for recent vintages (many haven't had time to reach a final outcome yet), which would bias any out-of-sample read on those periods.

- **No LendingClub-assigned risk fields.** `grade`, `sub_grade`, `int_rate`, and `installment` are excluded from the feature set — those are LendingClub's own underwriting output, and including them would leak the answer into the model rather than building an independent risk view.

- **FICO is included as a rolling percentile, not a raw score.** Bureau scoring models get revamped periodically (e.g. FICO 8 → 9 → 10), and the dataset has no field indicating which score version was used or when it changed — so there's no way to detect a revamp from the data directly. Using a fixed score cutoff (e.g. "FICO < 667") risks going stale silently if the bureau shifts its scale.

  The mitigation: each loan's FICO is converted to its **percentile rank against a trailing 6-month window of prior originations** (`fico_percentile`) before it's WOE-binned into the model. Score revamps typically preserve relative ordering between borrowers even when they shift the absolute point scale, so a percentile rank stays meaningful across a version change in a way a fixed bin edge does not.

  This came at effectively no performance cost — the percentile version matched (out-of-sample, even slightly beat) the raw-FICO version:

  | | Raw FICO | No FICO | FICO-percentile |
  |---|---|---|---|
  | Train Gini | 0.3795 | 0.3475 | 0.3785 |
  | Test Gini (out-of-time, 2016) | 0.3520 | 0.3186 | **0.3527** |

  Trade-off worth tracking: this requires maintaining a live rolling reference distribution at scoring time, and because the percentile is always relative to a moving population, it would mask a genuine population-wide credit-quality shift (everyone getting riskier just re-centers the percentiles). The raw score's own trend/PSI should still be monitored separately as a signal, even though it isn't a model input.

- **`revol_util` has an unstable sign.** It clears the IV threshold (~0.02, weak) but its logistic regression coefficient comes out positive (wrong direction) in every version of the model built so far — likely multicollinearity with other features rather than a real effect. Worth reviewing or dropping in a refinement pass.

- **v2 feature expansion: `loan_to_income` earned its place, `bc_util` and six other candidates didn't.** Tested adding `credit_history_years`, `loan_to_income`, three delinquency-recency fields, `acc_now_delinq`, `tax_liens`, `addr_state`, and `bc_util` (bankcard utilization %). Only `loan_to_income` (IV = 0.125, the second-strongest predictor after `term_months`) and `bc_util` (IV = 0.029) cleared the IV bar. `loan_to_income` lifted out-of-time test Gini from 0.3527 (v1) to 0.3578 — a real but modest ~1.4% relative gain. `bc_util` cleared IV on a univariate basis but added **zero** measurable lift on top (test Gini unchanged at 0.3578) and made `revol_util`'s existing sign instability worse (its coefficient moved from +0.36 to +0.56) since the two utilization measures overlap almost entirely — **recommendation is to keep `loan_to_income`, drop `bc_util`.** Separately, `loan_to_income` introduced its own sign flip: `loan_amnt`'s coefficient turned positive, because `loan_to_income = loan_amnt / annual_inc` shares a component with `loan_amnt` and the two are highly correlated (same kind of multicollinearity as `revol_util`, not a reversed real-world relationship). `revol_bal_to_limit`, a credit-mix ratio, `zip_code`, and `application_type` were considered but excluded or scoped out — the first two because their underlying fields (`total_rev_hi_lim`, `num_rev_accts`, `num_il_tl`) are 100% missing for loans issued 2007–2011 and only fully populated from 2013 onward, so any signal from them would partly reflect loan vintage rather than credit risk.
