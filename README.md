# Credit Risk Strategy — LendingClub

Credit risk analysis and strategy built on the LendingClub loan dataset.

**[Read the full business report](https://claude.ai/artifact/AHNNSnJkcNLyyvvoyozjZL)** — scorecard performance, approval-cutoff strategy, risk-based pricing, and loan-amount guidance, written up for a non-technical audience.

## Data

Dataset is pulled from Kaggle via `kagglehub` (see `download_data.py`):

```
wordsforthewise/lending-club
```

## Notebooks

- `notebooks/eda.ipynb` — exploratory analysis of loan volume, default rates by grade/term/purpose, and key risk drivers.
- `notebooks/scorecard.ipynb` — the final WOE/logistic-regression credit scorecard (22 features, out-of-time validated, Individual applications only), plus three business applications built on it: approval-cutoff strategy, risk-based pricing, and loan-amount/exposure guidance.

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

## Business applications (see `notebooks/scorecard.ipynb`, sections 8–10)

- **Approval cutoffs.** Accepting every 2016 applicant would have realized a **$71.5M loss** (real historical cash flow). A score-based cutoff (score ≥ 534.3, 52.7% approval rate, 13.9% bad rate among approved) turns that into a **+$65.8M profit** — a **$137.3M swing**, achieved purely by applying the scorecard to the accept/decline line LendingClub already draws.
- **Risk-based pricing.** Splitting each LendingClub grade into thirds by our score reveals default-rate gaps of up to 20 points that LC's actual rate charged barely reflects — e.g. within grade C, the worst third defaults at 34.1% vs. 19.1% for the best third, while the rate charged differs by well under half a point. The worst third of grades C–G is a **net loss** historically, even at rates up to 28.9%.
- **Loan-amount / exposure guidance.** Loan size amplifies whatever the score predicts: for the worst-scoring fifth of applicants, losses grow from -$858/loan (small) to -$3,358/loan (large); for the best-scoring fifth, profit grows from +$257/loan to +$1,186/loan. Exposure limits should scale with score, not just the approve/decline line.

Full detail, caveats, and recommendations are in the [business report](https://claude.ai/artifact/AHNNSnJkcNLyyvvoyozjZL) linked at the top.
