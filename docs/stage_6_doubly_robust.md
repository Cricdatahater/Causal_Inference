# Stage 6: Doubly robust learning and observational policy evaluation

Stage 6 extends the S/T-learner baseline with a cross-fitted DR-learner and
observational policy evaluation. It uses existing NumPy, pandas, and scikit-learn
dependencies; no EconML package is required.

## Scope and artifacts

- Implementation: `src/doubly_robust.py`
- Executed analysis: `notebooks/04_doubly_robust_policy_evaluation.ipynb`
- Reproduction: `scripts/run_stage6.py`
- Behavioral tests: `tests/test_doubly_robust.py`
- Five saved figures: `reports/figures/stage6_*.png`
- Seven saved result tables: `reports/tables/stage6_*.csv`

Run from the repository with its Python environment:

```powershell
.venv\Scripts\python.exe scripts/run_stage6.py
.venv\Scripts\python.exe -m pytest -q
```

The runner sets the kernel working directory to the repository, executes every
cell, and persists notebook outputs after successful execution. For interactive
use, start Jupyter in the repository or notebooks directory, or set
`CAUSAL_INFERENCE_ROOT` before executing the notebook. Setup validates the
repository before creating any output directories. Figures and result CSVs are
tracked in Git; the separate simulated `data/*.csv` files remain ignored.

## Data separation and leakage prevention

The seed-42 simulation contains 20,000 customers. Its shuffled indices recreate
Stage 5's 14,000-customer training cohort. The remaining customers are divided
into 3,000 calibration customers and 3,000 final evaluation customers.

- Only training treatment, retention, and pre-treatment covariates fit models.
- Calibration uses covariates and predicted scores to set thresholds, without
  accessing calibration outcomes or simulator truth.
- Evaluation outcomes are used only for AIPW evaluation, never fitting or tuning.
- Simulator truth is accessed only to calculate effect errors and oracle profit.
- Explicit feature selection excludes customer IDs, outcomes, treatment from
  nuisance covariates, oracle columns, and arbitrary post-treatment columns.

Stage 6 uses a smaller final evaluation cohort than Stage 5. The new S/T/DR
comparison is on a common cohort, but its total profits are not directly
comparable to Stage 5's 6,000-customer totals.

## Cross-fitted nuisance estimation

Three shuffled treatment-stratified folds partition the training cohort. In
each fold, fit propensity `e(X)`, control response `mu0(X)`, and treated response
`mu1(X)` on the other folds. Predict all three functions on the held-out fold.
Each row receives exactly one out-of-fold prediction. Fold indices are retained
so disjointness and complete coverage can be checked.

The nuisance estimators are unweighted random-forest classifiers with 200 trees,
minimum leaf size 25, and all transformed features eligible at each split.
Categorical covariates are one-hot encoded with unknown categories ignored.
Outcome classifiers are fitted separately in each treatment arm. Single-class
outcome folds return the appropriate constant probability.

Propensity scores used as denominators are clipped to `[0.05, 0.95]`. Raw
propensity predictions, clipping rates, Brier scores, observed inverse-weight
extremes, and effective sample size are retained as diagnostics. Effective
sample size here describes the combined observed-treatment inverse weights;
it is not a policy-specific precision or coverage guarantee.

For out-of-sample evaluation, average the probability predictions from the
retained fold models. For training rows, use `oof_nuisance_` rather than the
ensemble, since the ensemble includes models that saw those rows.

## DR effect scores and final CATE model

For binary treatment and binary observed retention:

$$
\phi_1 = \hat\mu_1(X) + \frac{T(Y-\hat\mu_1(X))}{\hat e(X)},\qquad
\phi_0 = \hat\mu_0(X) + \frac{(1-T)(Y-\hat\mu_0(X))}{1-\hat e(X)}.
$$

The effect pseudo-outcome is `phi1 - phi0`. Regress these cross-fitted scores on
pre-treatment covariates using a 200-tree random-forest regressor with minimum
leaf size 50. Final CATE predictions are bounded to `[-1, 1]` because retention
is binary. Raw pseudo-outcomes and policy-evaluation scores are not bounded;
clipping them would change the estimator and its target.

The population AIPW identity is doubly robust: absent propensity-clipping bias,
it recovers the effect if either the propensity model is correct or both outcome
models are correct, given identification assumptions. This is not a guarantee
of finite-sample accuracy, valid intervals, or superior DR-learner predictions.
Both nuisance families can be wrong; weak overlap and unobserved confounding
remain risks. Score regression introduces further approximation error.

Stage 5's S/T settings, including their balanced outcome classifiers, are
retained for comparison. DR uses unweighted probabilities because class
balancing can change probability calibration. The comparison therefore includes
model-configuration differences and cannot isolate the estimator alone.

## Policy calibration and capacity

For each learned policy, calculate `80 * predicted_CATE - 10` on calibration
customers. Choose positive scores in the top 20% and retain the lowest selected
score as a fixed economic threshold. If no customer qualifies, treat nobody.
On evaluation customers, require positive score and that prechosen threshold.
If more than 20% qualify, rank eligible customers by score and enforce the hard
capacity cap. Ties are resolved stably. Persist thresholds, candidate counts,
treatment counts, and whether the cap was active.

Comparators are treat-none and a seed-fixed random 20% allocation. The oracle
policy is shown only as a simulation-truth upper bound, not as an observational
policy or model trained from deployable data.

## Observational policy evaluation

For policy decisions `d_i`, paired incremental-profit contributions relative to
treat-none are:

$$
z_i = d_i\{80(\phi_{1i}-\phi_{0i})-10\}.
$$

Report their mean (profit per eligible customer) and sum (profit for an
`n`-customer campaign). Variance is based on the paired differences, rather than
subtracting independent policy-value variances. AIPW retention rate is also
returned. Sampling can put unbounded AIPW estimates outside the physical outcome
range; these estimates are not truncated.

Pointwise 95% normal intervals use `sd(z, ddof=1)/sqrt(n)` for the mean and
multiply by `n` for campaign-scale totals. For a fixed policy, inference requires
iid sampling, overlap, suitable nuisance convergence rates, and identification.
The strict cohort-level allocation cap creates dependence between decisions, so
these intervals are illustrative approximations rather than certified intervals
for that constrained allocation rule. They omit nuisance/calibration fitting
variability and policy-selection uncertainty, and are not simultaneous intervals.
Conditioning on an independently calibrated threshold does not remove dependence
introduced by applying a cohort-level capacity cap.

Final test data are not used to choose a winning policy, nuisance hyperparameters,
or clipping level. Oracle metrics and observational-vs-truth comparisons are
diagnostics, not a replacement for a future prospective experiment.

## Clipping sensitivity and validation

Vary evaluation-only propensity clipping across 0.01, 0.025, 0.05, and 0.10,
holding fitted models and decisions fixed. This checks weight sensitivity; it
does not rerun DR training or tune clipping on the final cohort.

Tests enumerate the joint treatment/outcome distribution to check the population
double-robust identity with either nuisance family misspecified. Other tests
cover raw unbounded scores, fold coverage, oracle/post-treatment exclusion,
reproducibility, covariate-only prediction, single-class outcomes, failed refits,
input validation, and paired policy variance. Notebook completion additionally
checks all policy capacities, prediction bounds, oracle regret, finite results,
and saved files. No test requires DR to beat S/T on a particular seed.

## References

- [EconML DR-learner specification](https://econml.azurewebsites.net/spec/estimation/dr.html)
- [EconML DR-learner estimating equations](https://econml.azurewebsites.net/_autosummary/econml.dr.DRLearner.html)

These references describe the methodology. This project implements its own
binary-treatment version and does not claim equivalence to EconML's fitted models.
