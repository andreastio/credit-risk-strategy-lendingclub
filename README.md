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

- **v2 also rebuilt on Individual applications only, dropping Joint App loans.** A joint application combines two people's finances — `annual_inc`/`dti` mean something structurally different for a joint applicant than an individual one, and LendingClub even provides separate `annual_inc_joint`/`dti_joint` fields for that population — so modeling both together with one feature set risks blurring what those fields represent. Excluding the 120,710 Joint App loans (~5.3% of the dataset) made almost no difference to performance: out-of-time test Gini actually nudged up slightly, from 0.3578 (v1 features + `loan_to_income`, all applications) to 0.3588 (same features, Individual-only). That's the expected result for a small, cleanly-separable segment — it confirms the segmentation is conceptually correct without costing anything in raw performance. Joint App loans would be a reasonable candidate for their own separate scorecard using the joint-specific fields, but that's out of scope here.

- **Vintage-driven missingness: several bureau "trended" fields are really just a proxy for loan recency.** LendingClub added a large block of these fields partway through its history — a real risk signal shouldn't be near-100% missing before some year and then ~0% after.

  | Field | Missing 2007–2011 | Missing 2012 | Missing 2013+ | Status |
  |---|---|---|---|---|
  | `total_rev_hi_lim`, `num_rev_accts`, `num_il_tl` | 100% | 52% | 0% | Excluded on theory (not tested) |
  | `open_act_il` | 100% (through 2014) | — | 94.9% in 2015, 0% from 2016 | Excluded — would be entirely absent in train, fully present in test |
  | `il_util` | 100% (through 2014) | — | 95.6% in 2015, 13–16% in 2016–2018 | Excluded — essentially no real training data even after 2013 |
  | `bc_util` | 100% | 15% | ~1% | **Tested** (see above) — clears IV but adds no lift |
  | `mo_sin_old_il_acct` | 100% | 54% | ~3% | **Tested anyway** — see below |
  | `num_sats` | 100% | 30% | 0% | **Tested anyway** — see below |

  `mo_sin_old_il_acct` and `num_sats` were tested despite the caveat, on the theory that a real signal might still be worth the vintage noise. Both failed the IV bar regardless (0.008 and 0.012, both well under 0.02), so the question was moot in practice — but the underlying vintage-sanity check is worth recording: each field's "Missing" bin (i.e. pre-2013 loans) showed a *lower* bad rate than the population overall (16.18% and 14.80% vs. 18.42%), plausible either as a real vintage/economic-cycle effect or as the missingness artifact itself. Two related candidates were also tested and found to carry essentially zero signal for a different reason — extreme sparsity, not vintage: `delinq_to_loan` (`delinq_amnt` / `loan_amnt`, IV ≈ 0.000000, only 0.32% of loans have any current delinquent amount) and `chargeoff_within_12_mths` (IV ≈ 0.000018, ~0.76% nonzero).

- **Major finding: a milder-vintage-tier batch of bankcard/inquiry-recency fields produced the biggest single improvement in this project.** Systematically swept ~35 remaining bureau fields and classified each by missingness-by-year into unusable (100% missing through 2014–2015, e.g. `open_acc_6m`, `open_il_12m/24m`, `total_bal_il`, `all_util`, `inq_fi`, `total_cu_tl`, `inq_last_12m`), severe-vintage (100%/52%/~0% pattern, same tier as `mo_sin_old_il_acct`/`num_sats` above — includes `pct_tl_nvr_dlq`, `num_actv_bc_tl`, `num_bc_tl`, `tot_hi_cred_lim`, and ~10 others), and mild-vintage (100%/~15%/~1%, same tier as `bc_util`). Tested the 7 mild-vintage candidates: `acc_open_past_24mths`, `bc_open_to_buy`, `mths_since_recent_bc`, `percent_bc_gt_75`, `total_bal_ex_mort`, `total_bc_limit`, `mths_since_recent_inq`.

  **6 of 7 cleared the IV bar** (`total_bal_ex_mort` at IV=0.007 was the only miss), led by `acc_open_past_24mths` (accounts opened in the past 24 months, IV=0.083 — the single strongest new predictor added this session). Out-of-time test Gini jumped from 0.3588 to **0.3847** — a ~7.2% relative gain, roughly 5x bigger than the `loan_to_income` improvement. As a bonus, adding these bankcard-detail fields finally resolved `revol_util`'s chronic sign instability (coefficient is now correctly negative at -0.031, for the first time in this project) — the extra fields apparently absorbed whatever collinear signal was confusing it. `loan_amnt` remains positive (worse than before, +0.252) and `percent_bc_gt_75` is now barely positive too (+0.032) — both flagged, neither large.
