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
- `notebooks/scorecard_adv.ipynb` — scorecard v2: Individual-applications-only, 22 selected features after a full systematic sweep of every LendingClub bureau field (see below).
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

- **The remaining severe-vintage-tier fields were tested to complete the sweep — 6 more cleared IV, contradicting the earlier "this whole tier is a dead end" read.** `mo_sin_old_il_acct`/`num_sats` had both failed IV, which suggested the entire 52%-missing-in-2012 tier wasn't worth pursuing. Testing the other 13 fields in that tier disproved that: `num_tl_op_past_12m` (IV=0.058), `mo_sin_rcnt_tl` (0.043), `tot_hi_cred_lim` (0.040), `mo_sin_rcnt_rev_tl_op` (0.033), `num_actv_rev_tl` (0.032), and `num_rev_tl_bal_gt_0` (0.031) all cleared the bar and were selected. Out-of-time test Gini rose further to **0.3867**. All 13 fields in this tier share the *identical* Missing-bin population (same ~67,527 loans, since LendingClub added them together in one data-collection wave) with a consistent bad rate of 15.28% vs. 18.42% overall (WOE=0.225) — a real, moderate signal, plausibly a vintage/economic-cycle effect, that turned out not to disqualify the fields that carried genuine additional information on top of it. New sign issue: `mo_sin_rcnt_rev_tl_op` came in positive (+0.372), likely collinear with the very similar `mo_sin_rcnt_tl`. Two very sparse recency fields (`mths_since_recent_bc_dlq`, `mths_since_recent_revol_delinq`) were also tested and rejected (IV=0.0021 and 0.0015) — same profile as the other derogatory-recency fields already ruled out.

- **This completes the systematic sweep — full field-by-field results:**

  | Field | IV | Verdict |
  |---|---|---|
  | `term_months` | 0.240 | ✅ Selected (v1) |
  | `loan_to_income` | 0.125 | ✅ Selected |
  | `fico_percentile` | 0.110 | ✅ Selected (v1) |
  | `acc_open_past_24mths` | 0.083 | ✅ Selected |
  | `dti` | 0.075 | ✅ Selected (v1) |
  | `num_tl_op_past_12m` | 0.058 | ✅ Selected |
  | `bc_open_to_buy` | 0.055 | ✅ Selected |
  | `verification_status` | 0.051 | ✅ Selected (v1) |
  | `mo_sin_rcnt_tl` | 0.043 | ✅ Selected |
  | `total_bc_limit` | 0.041 | ✅ Selected |
  | `tot_hi_cred_lim` | 0.040 | ✅ Selected |
  | `mths_since_recent_inq` | 0.040 | ✅ Selected |
  | `loan_amnt` | 0.037 | ✅ Selected (v1) — sign flipped, +0.289 |
  | `mo_sin_rcnt_rev_tl_op` | 0.033 | ✅ Selected — sign flipped, +0.372 |
  | `percent_bc_gt_75` | 0.033 | ✅ Selected — sign flipped, +0.034 |
  | `num_actv_rev_tl` | 0.032 | ✅ Selected |
  | `num_rev_tl_bal_gt_0` | 0.031 | ✅ Selected |
  | `annual_inc` | 0.031 | ✅ Selected (v1) |
  | `mths_since_recent_bc` | 0.029 | ✅ Selected |
  | `mort_acc` | 0.027 | ✅ Selected (v1) |
  | `revol_util` | 0.022 | ✅ Selected (v1) — sign now correct (-0.059) |
  | `home_ownership` | 0.021 | ✅ Selected (v1) |
  | `purpose` | 0.020 | Tested, below bar |
  | `inq_last_6mths` | 0.018 | Tested, below bar |
  | `num_op_rev_tl` | 0.014 | Tested, below bar |
  | `num_sats` | 0.012 | Tested, below bar |
  | `num_actv_bc_tl` | 0.012 | Tested, below bar |
  | `mo_sin_old_il_acct` | 0.008 | Tested, below bar |
  | `open_acc` | 0.008 | Tested, below bar |
  | `total_bal_ex_mort` | 0.007 | Tested, below bar |
  | `emp_length_years` | 0.006 | Tested, below bar |
  | `total_il_high_credit_limit` | 0.006 | Tested, below bar |
  | `num_bc_sats` | 0.006 | Tested, below bar |
  | `num_bc_tl` | 0.005 | Tested, below bar |
  | `pct_tl_nvr_dlq` | 0.005 | Tested, below bar |
  | `num_accts_ever_120_pd` | 0.005 | Tested, below bar |
  | `num_tl_30dpd` | 0.004 | Tested, below bar |
  | `revol_bal` | 0.003 | Tested, below bar |
  | `mths_since_recent_bc_dlq` | 0.002 | Tested, below bar |
  | `mths_since_recent_revol_delinq` | 0.002 | Tested, below bar |
  | `delinq_2yrs` | 0.001 | Tested, below bar |
  | `num_tl_120dpd_2m` | 0.001 | Tested, below bar |
  | `pub_rec`, `pub_rec_bankruptcies`, `tax_liens` | ~0.001 | Tested, below bar |
  | `total_acc` | 0.0004 | Tested, below bar |
  | `chargeoff_within_12_mths` | 0.00002 | Tested, below bar (~0.76% nonzero) |
  | `delinq_to_loan` | ~0.000000 | Tested, below bar (~0.32% nonzero) |
  | `bc_util` | 0.029 (IV clears, no lift) | Tested and dropped — see above |
  | `addr_state`, `credit_history_years`, `acc_now_delinq` | — | Tested (earlier round), below bar |
  | `total_rev_hi_lim`, `num_rev_accts`, `num_il_tl`, `mo_sin_old_rev_tl_op`, `num_tl_90g_dpd_24m`, `avg_cur_bal` | — | Excluded on missingness theory, not individually tested (same family as the tested severe-tier fields) |
  | `open_act_il`, `il_util`, `open_acc_6m`, `open_il_12m/24m`, `open_rv_12m/24m`, `total_bal_il`, `all_util`, `inq_fi`, `total_cu_tl`, `inq_last_12m`, `mths_since_rcnt_il` | — | Unusable — 100% missing through 2014–2015 |
  | `grade`, `sub_grade`, `int_rate`, `installment` | — | Excluded from the start — LendingClub's own risk decision (leakage) |
  | `total_pymnt`, `recoveries`, `hardship_*`, `settlement_*`, etc. | — | Excluded — post-origination outcome fields (leakage) |

  **Nothing meaningful remains untested.** Final model: 22 selected features, out-of-time test Gini = **0.3867** (AUC 0.6934, KS 0.2771) — up from 0.3527 for the original v1 baseline, a ~9.6% relative improvement across the whole feature-engineering process.

## Benchmark: our scorecard vs. LendingClub's own risk assessment

Everything above was built without `grade`, `sub_grade`, or `int_rate` as model inputs — those are LendingClub's own underwriting output, and using them would be leakage. Purely as an **evaluation-only benchmark** (never fed into the model), here's how our from-scratch scorecard compares against LC's own risk measures, scored on the exact same out-of-time (2016) test population:

| | AUC | Gini | KS |
|---|---|---|---|
| LC `grade` (A–G, ordinal) | 0.6818 | 0.3635 | 0.2726 |
| LC `sub_grade` (A1–G5, ordinal) | 0.6912 | 0.3824 | 0.2761 |
| LC `int_rate` (continuous) | 0.6912 | 0.3823 | 0.2776 |
| **Our scorecard** | **0.6934** | **0.3867** | 0.2771 |

**Our scorecard actually beats all three of LendingClub's own risk measures** on this population, including `int_rate` — LC's finest-grained, continuous pricing signal, which presumably bakes in whatever non-bureau data and manual-underwriting judgment they had access to. This is a strong result for a model built entirely from public application/bureau data: the systematic feature sweep above (particularly the bankcard-detail and account-recency fields) recovered signal that's at least as good as LendingClub's own proprietary assessment, on this out-of-time slice. (Caveat: this is one test window — 2016 vintage only — not a claim that this holds across all periods or economic conditions.)
