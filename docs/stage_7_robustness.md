# Stage 7: Robustness and business sensitivity

Stage 7 tests whether Stage 6's model and policy comparisons persist across
independent simulated cohorts and prespecified business assumptions. It is an
initial Monte Carlo robustness study, not validation on real customers.

## Prespecified experiment

| Setting | Value |
|---|---|
| Simulation seeds | 42 through 51 inclusive (10 cohorts) |
| Customers per cohort | 20,000 |
| Training / calibration / evaluation | 14,000 / 3,000 / 3,000 |
| Learners | S-learner, T-learner, three-fold cross-fitted DR-learner |
| Algorithm random seed | Fixed at 42 for every cohort |
| Retention value | $40, $80, $120 |
| Treatment cost | $5, $10, $20 |
| Capacity | 10%, 20%, 30% |
| Business scenarios | Full factorial: 27 |
| Policies per scenario | Treat-none, random, S, T, DR, oracle |
| Policy/scenario/seed evaluations | 1,620 |

Each seed regenerates the cohort and random split and refits every learner.
Model settings remain those used in Stage 6: S/T use 300-tree balanced outcome
forests; DR uses 200-tree unweighted nuisance/final forests with three folds.
Fixing algorithm randomness isolates variation in cohort generation and splitting;
it does not measure extra variability from changing estimator random seeds.
No final-evaluation result is used to select seeds, models, hyperparameters,
business settings, or favorable runs.

For each seed, fit only on pre-treatment covariates, observed treatment, and
observed retention in the training cohort. The model interface is checked by
leakage tests; simulator truth is never given to fitting or prediction functions.
As in Stage 6, the S/T and DR model configurations differ, so comparisons do not
isolate the estimator independently of those settings.

The simulator standardizes some covariates within each generated cohort. Cohorts
are independently simulated across seeds, but rows within a cohort are not
perfectly independent draws from a fixed feature-generating distribution. This
study treats the entire simulated seed cohort as the independent replicate.

## Policy recalibration under each business scenario

CATE predictions are reused across the 27 scenarios; economic settings change
which customers are profitable, not the fitted retention response functions.
For each learner, recompute `retention_value * CATE - treatment_cost` on the
separate calibration cohort. Select positive calibration scores up to capacity
and set a threshold equal to the lowest selected score. Apply that threshold to
evaluation covariates. If too many qualify, choose the highest eligible scores
up to the strict evaluation capacity cap, with stable tie resolution.

Calibration uses no outcomes, treatment assignments, or oracle effects. If no
calibration customers are profitable, the policy treats nobody, even if some
later evaluation predictions are positive. Capacity zero also treats nobody.

Random targeting treats exactly the allowed fraction without a profitability
filter. Oracle targeting uses true effects solely for evaluation. Treat-none is
always feasible. Recalibration uses a different threshold for each business
scenario but never selects scenarios after seeing evaluation outcomes.

Costs and values are accounting assumptions about an unchanged intervention.
Changing the discount itself could change treatment effects and requires a new
simulation mechanism. That is outside this stage.

## Outcomes, uncertainty, and paired comparisons

Save CATE PEHE, absolute ATE error, CATE MAE, and correlation once per learner
and seed. Save simulator-truth expected profit, profit per customer, regret,
treatment rate, and oracle-profit fraction for each policy/scenario/seed.
A zero-profit oracle yields a missing oracle-profit ratio, not Infinity or zero.
The summary records how many nonmissing replicates contribute to each metric.

Summarize independent seeds within each scenario:

- mean, median, and sample SD;
- Monte Carlo standard error of the seed average: `SD / sqrt(n)`;
- minimum, maximum, and empirical 2.5%/97.5% quantiles.

The SD describes variability between seed results. MCSE describes numerical
precision of the estimated average under this simulation, not uncertainty about
real-world business profit. Quantiles from ten seeds are descriptive; they are
not 95% confidence intervals or reliable estimates of rare tails. The capacity
plot shades one seed SD, clearly labeled as such.

Compute DR-minus-S, DR-minus-T, and S-minus-T profit differences on the same
seed, scenario, and evaluation cohort. Save their raw differences, averages,
win fractions, and ties. Do not pool dependent scenarios or customer rows as
independent simulation replicates. Positive difference frequency is descriptive;
it is not a hypothesis test or proof of a universally best learner.

Evaluate the learned and baseline policies observationally with AIPW using
training-only nuisance models. Persist AIPW point estimates, approximate
pointwise intervals, and estimation errors versus simulator truth. These
intervals inherit Stage 6's limitations, including omitted fitting uncertainty
and potential dependence from cohort-level capacity allocation. Stage 7 reports
estimation-error variability rather than claiming nominal interval coverage.

## Reproduction and provenance

Implementation: `src/robustness.py`. Runner: `scripts/run_stage7.py`.
Analysis: `notebooks/05_robustness_and_sensitivity.ipynb`.
Tests: `tests/test_robustness.py`.

Run the entire stage from the repository:

```powershell
.venv\Scripts\python.exe scripts/run_stage7.py
.venv\Scripts\python.exe -m pytest -q
```

The runner publishes eight CSV tables and a JSON manifest in `reports/tables/`,
then executes and saves the notebook, which renders six PNG figures in
`reports/figures/`. All published tables, the manifest, notebook, and figures
are tracked by Git. Cached per-seed computations in `reports/stage7_cache/` are
ignored; they can be regenerated.

Checkpoints are keyed by a fingerprint covering seed list, population size,
scenario grid, model seed, source hashes for the simulation/model/policy/experiment
modules, and Python/NumPy/pandas/scikit-learn versions. A completed seed is saved
atomically. Interrupted runs can resume under the identical fingerprint. The
manifest records configuration, versions, source hashes, completion timestamp,
and table row counts.

For a smaller local smoke run, use `--seeds 42 43 --population-size 1000`.
**This overwrites the published Stage 7 tables and notebook with that configuration.**
Rerun the default command to restore the ten-seed report. `--skip-notebook` saves
only experiment tables. The notebook reads the manifest and labels the actual
seed count, cohort size, and scenarios; the published default report uses the
prespecified ten-seed experiment.

The notebook itself renders saved results and does not launch expensive model
fits. Start Jupyter in the repository or notebooks directory, or set
`CAUSAL_INFERENCE_ROOT` before running it from another location.

## Validation and remaining scope

Tests cover scenario-grid completeness/validation, deterministic disjoint
splits that ignore labels, calibration-only threshold selection, empty policies,
zero/full capacity, profitability checks, strict caps and stable ties, exact
sample SD/MCSE, missing ratios, seed pairing and ties, and model input whitelists.

Completion checks require every prespecified seed and scenario, no duplicate
rows, six policies per scenario, finite key metrics, feasible allocations,
nonnegative oracle regret, paired-difference consistency, treat-none zero profit,
manifest row-count consistency, and six saved figures. No assertion requires
one learner to beat another.

The study does not address unobserved confounding, manipulated poor overlap,
structural changes to treatment effects, longitudinal interference, estimator
random-seed sensitivity, or real data. Poor-overlap and hidden-confounding stress
tests are Stage 8; final synthesis and delivery are Stage 9.

## Published results

The default ten-seed run completed successfully. All 59 tests and notebook
completion checks passed. Baseline S profit averages $7,087.76, DR $6,954.56,
and T $6,630.85. DR beats S on 3/10 seeds and T on 9/10; its average absolute
ATE error is 0.0074 versus S 0.0553 and T 0.0619. The Stage 6 single-seed
DR profit advantage is not a stable general ranking. Complete numerical
results and artifact links are in Section 16 of
[causal_design.md](causal_design.md).
