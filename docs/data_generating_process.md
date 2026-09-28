# Data-Generating Process

## 1. Purpose

This simulator generates observational customer-retention data with:

- non-random treatment assignment;
- measured confounding;
- heterogeneous treatment effects;
- known counterfactual response probabilities; and
- adequate treatment overlap.

The simulator's oracle variables are retained only for evaluation and must not
be used as model inputs.

## 2. Business assumptions

- Treatment is a standardized $10 retention offer.
- Treatment cost is treated as a fixed expected cost per targeted customer.
- Retaining a customer for 90 days is valued at $80.
- A deployment policy may treat at most 20% of eligible customers.
- Historical treatment assignment is not limited to 20%. The 20% constraint
  applies to the policy developed later.

## 3. Observed covariates

| Variable | Type | Range/categories | Role |
|---|---|---|---|
| `tenure_months` | Integer | 1-120 | Confounder/effect modifier |
| `plan_type` | Category | basic, standard, premium | Confounder/effect modifier |
| `monthly_price` | Continuous | Positive dollars | Confounder |
| `login_days_30d` | Integer | 0-30 | Confounder/effect modifier |
| `usage_trend_90d` | Continuous | -1 to 1 | Confounder/effect modifier |
| `support_tickets_30d` | Integer | 0-10 | Confounder/effect modifier |
| `late_payments_12m` | Integer | 0-12 | Confounder |
| `region` | Category | north, south, east, west | Possible confounder |
| `device_type` | Category | mobile, desktop, tablet | Effect modifier |

## 4. Causal mechanisms

The simulator contains three primary mechanisms:

1. Baseline retention: `mu0(x) = P(Y(0)=1 | X=x)`
2. Retention under treatment: `mu1(x) = P(Y(1)=1 | X=x)`
3. Historical treatment assignment: `e(x) = P(T=1 | X=x)`

The true conditional treatment effect is:

`tau(x) = mu1(x) - mu0(x)`

## 5. Treatment-effect heterogeneity

The offer is designed to work best for moderately disengaged customers.

- Highly engaged customers are likely to renew without treatment.
- Moderately disengaged customers can be persuaded by the offer.
- Severely disengaged customers may not respond.
- Basic-plan customers are more price sensitive.
- Premium customers respond less to a generic discount.
- Customers with many support issues respond less to a discount-only treatment.

## 6. Treatment assignment

Historical targeting depends on measured churn-risk signals:

- falling usage;
- fewer login days;
- more support tickets;
- more late payments; and
- subscription plan.

This creates confounding while preserving conditional exchangeability because
all common causes used by the assignment and outcome mechanisms are observed.

## 7. Model-visible columns

- Customer covariates
- `treatment`
- `retained_90d`

## 8. Oracle-only columns

- `true_propensity`
- `true_mu0`
- `true_mu1`
- `true_cate`

Oracle columns must be removed before fitting any predictive or causal model.

## 9. Initial validation targets

- Treatment rate between 20% and 45%
- Overall retention between 45% and 85%
- Propensity scores between 0.05 and 0.80
- Non-zero variation in true CATE
- Both profitable and unprofitable treatment candidates
- A visible difference between naive and true treatment effects
